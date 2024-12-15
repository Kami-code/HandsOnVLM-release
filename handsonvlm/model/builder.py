# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from handsonvlm.model import *
from handsonvlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, TIME_TOKEN_TEMPLATE, HAND_TOKEN_TEMPLATE


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"):
    kwargs = {"device_map": device_map, 'is_inference': True}
    print("model_name: ", model_name, "model_base: ", model_base, "model_path: ", model_path)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if 'mpt' in model_name.lower():
        warnings.warn("mpt is currently not supported for LITA")
    if 'lora' in model_name.lower() and model_base is not None:
        # todo: support lora for liha
        warnings.warn("lora is currently not supported for LITA")
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')

        lora_cfg_pretrained = HandsOnVLMConfig.from_pretrained(model_path)
        print("lora_cfg_pretrained: ", lora_cfg_pretrained)
        print("model_base: ", model_base, " model_path: ", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading LLaVA from base model...')
        model = HandsOnVLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            assert 0, "we don't support loading from HF Hub yet"
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading LIHA from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = HandsOnVLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items() if 'mm_projector' in k}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        print(f"loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        print(f"loading model from {model_path} and kwargs: {kwargs}")
        kwargs['traj_decoder_name'] = 'CVAE'
        print("using traj_decoder_name: ", kwargs['traj_decoder_name'], "Please check if it is correct")
        model = HandsOnVLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        print(f"loading vision tower model from pretrained {vision_tower.vision_tower_name}")
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # time tokens and embeddings

    num_time_tokens = getattr(model.config, "num_time_tokens", 0)
    if num_time_tokens > 0:
        time_tokens = [TIME_TOKEN_TEMPLATE.format(t=x) for x in range(num_time_tokens)]
        num_new_tokens = tokenizer.add_tokens(time_tokens)

        if model_base is None:
            assert num_new_tokens == 0, "time tokens should already be in the tokenizer for full finetune model"

        num_added_tokens = tokenizer.add_tokens(HAND_TOKEN_TEMPLATE)
        # update the segmentation token id
        hand_traj_token_idx = tokenizer(HAND_TOKEN_TEMPLATE, add_special_tokens=False).input_ids[0]

        if num_new_tokens > 0:
            warnings.warn("looking for weights in mm_projector.bin")
            assert num_new_tokens == num_time_tokens
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                assert 'model.embed_tokens.weight' in weights and 'lm_head.weight' in weights

                dtype = input_embeddings.dtype
                device = input_embeddings.device

                tokenizer_time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
                time_token_ids = getattr(model.config, 'time_token_ids', tokenizer_time_token_ids)
                input_embeddings[tokenizer_time_token_ids] = weights['model.embed_tokens.weight'][time_token_ids].to(dtype).to(device)
                output_embeddings[tokenizer_time_token_ids] = weights['lm_head.weight'][time_token_ids].to(dtype).to(device)
            elif os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                weights = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                assert 'base.model.model.embed_tokens.weight' in weights and 'base.model.model.lm_head.weight' in weights

                dtype = input_embeddings.dtype
                device = input_embeddings.device

                tokenizer_time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
                time_token_ids = getattr(model.config, 'time_token_ids', tokenizer_time_token_ids)
                input_embeddings[tokenizer_time_token_ids] = weights['model.embed_tokens.weight'][time_token_ids].to(dtype).to(device)
                output_embeddings[tokenizer_time_token_ids] = weights['lm_head.weight'][time_token_ids].to(dtype).to(device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    load_sharded_checkpoint(model, model_path, strict=True)
    return tokenizer, model, image_processor, context_len
