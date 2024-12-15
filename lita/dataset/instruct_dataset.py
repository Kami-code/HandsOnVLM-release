# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import json
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from lita.dataset.base_dataset import BaseDataset


class LlavaDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        """
        __getitem__ returns a dictionary with the following keys:
        - input_ids: tensor (247)
        - labels: tensor (247)
        - image: tensor (100, 3, 224, 224)
        """


        super(LlavaDataset, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'coco', 'train2017')

    def get_sources(self, i):
        return self.list_data_dict[i]
    
    def get_visual(self, sources):
        image_path = os.path.join(self.image_folder, sources['image'])
        image = self.load_image(image_path)
        return torch.stack([image] * self.data_args.num_frames, dim=0)
    
    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'LLaVA-Instruct-150K', 'llava_instruct_150k.json')
        self.list_data_dict = json.load(open(data_path, "r"))
        print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        print(f"Sample: {self.list_data_dict[0]}")


if __name__ == '__main__':
    """
    python ./lita/dataset/instruct_dataset.py --vision_tower openai/clip-vit-large-patch14 --task_sample_rate 1 --dvc_sample_rate 1 --event_loc_sample_rate 1 --imgqa_sample_rate 1 --vidqa_sample_rate 1 --temporal_reasoning_sample_rate 1 --output_dir . --data_path /ocean/projects/cis240031p/cbao/datasets/LITA
    
    """
    import os
    import logging
    import pathlib
    import warnings
    from typing import Dict, Optional, Sequence, List

    import torch

    import transformers

    from llava.train.llava_trainer import LLaVATrainer
    from llava import conversation as conversation_lib
    from lita.model import *
    from lita.dataset.hybrid_dataset import HybridDataset, DataCollatorForSupervisedDataset
    from lita.arguments import ModelArguments, DataArguments, TrainingArguments

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        model = LitaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)


            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            from lita.train.train import smart_tokenizer_and_embedding_resize
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        # time tokens
        if model_args.mm_use_im_start_end or model_args.mm_use_im_patch_token:
            warnings.warn("compatability with im_start_end and im_patch tokens are not checked for time tokens")
        model.config.num_frames = data_args.num_frames = model_args.num_frames
        model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
        training_args.num_time_tokens = model_args.num_time_tokens
        model.initialize_time_tokenizer(model_args, tokenizer=tokenizer)
        # video related configs
        model.config.input_type = model_args.input_type
        model.config.video_arch = model_args.video_arch

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    """Make dataset and collator for supervised fine-tuning."""
    print("start making llava dataset")
    llava_dataset = LlavaDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)


    for key, value in llava_dataset[0].items():
        print(f"{key}: {value.shape}")