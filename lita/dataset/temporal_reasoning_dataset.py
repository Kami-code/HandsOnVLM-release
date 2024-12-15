# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import glob
import json
import numpy as np
import re

from lita.dataset.base_dataset import BaseDataset
from lita.constants import DEFAULT_IMAGE_TOKEN, TIME_TOKEN_TEMPLATE


class TemporalReasoningDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset, self).__init__(data_path, tokenizer, data_args)
        
    def get_sources(self, i):
        vqas = self.list_data_dict[i]
        return self.format_temporal_reasoning(vqas)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
        
    def format_temporal_reasoning(self, vqas):
        out = {}
        vid = vqas['id']
        out['id'] = vid
        
        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*'+ self.ext)))
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif self.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, captions['image'])
            
        convo = []
        duration = vqas['duration']
        max_offset = float(self.data_args.num_time_tokens - 1)
        for i, vqa in enumerate(vqas['QA']):
            if i == 0:
                gpt_prompt = DEFAULT_IMAGE_TOKEN + '\n'
            else:
                gpt_prompt = ""
                
            question = vqa['q']
            answer = vqa['a']
            
            gpt_prompt += question.strip()
            
            # process answer
            timestamp_pattern = '\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>'
            rx = re.compile(timestamp_pattern, re.VERBOSE)
            timestamp_tokens = []
            new_answer = ""
            prev_end = 0

            # convert timestamp to time token
            for m in rx.finditer(answer):
                start = m.start(0)
                end = m.end(0)
                timestamp = float(m.group(0)[1:-1])
                timestamp_time = int(np.round(max_offset * (timestamp / duration)))
                timestamp_token = TIME_TOKEN_TEMPLATE.format(t=timestamp_time)
                new_answer += answer[prev_end:start]
                new_answer += timestamp_token
                prev_end = end
            new_answer += answer[prev_end:]
          
            gpt_value = new_answer.strip()
            convo.append({"from": "human", "value": gpt_prompt.strip()})
            convo.append({"from": "gpt", "value": gpt_value.strip()})
            
        out['conversations'] = convo
        # print("out.items(): ", out.items())
        """
        out is a dictionary with the following keys:
        - id: str
        - image: list of image absolute paths
        - conversations: list of dictionaries with keys 'from' and 'value'
        """
        return out
                
                
class TemporalReasoningDataset_activitynet(TemporalReasoningDataset):
    """
    <image>
    When is the process of finalizing the newly placed tiles being done?

    <t79> <t85> The process of finalizing the newly placed tiles is done between <t79> and <t85>. During this time, a heavy weight roller is rolled over the newly placed tiles to ensure they are firmly in place.
    """
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset_activitynet, self).__init__(data_path, tokenizer, data_args)
    
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'activitynet-captions', 'activitynet_frames')
        self.visual_data_type = 'video_frames'
        self.ext = '.jpg'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'temporal_reasoning', 'activitynet_train_gpt-4-0613_temp_6_f10009.json')
        data_dict = json.load(open(data_path, "r"))
        for vid in data_dict:
            data = data_dict[vid]            
            for vqa in data['QA']:
                out = {}
                out['id'] = vid
                out['duration'] = data['duration']
                out['QA'] = [vqa]
                self.list_data_dict.append(out)


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
    print("start making activitynet dataset")
    activity_dataset = TemporalReasoningDataset_activitynet(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    for key, value in activity_dataset[0].items():
        print(f"{key}: {value.shape}")