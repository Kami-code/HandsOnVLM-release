import torch

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from lita.model.lita_arch import LitaMetaForCausalLM
from handsonvlm.constants import HAND_TOKEN_TEMPLATE


class HandsOnVLMMetaForCausalLM(LitaMetaForCausalLM):

    def visual_to_tokens(self, images):
        # todo: check the need for this
        input_type = getattr(self.config, 'input_type', 'image')
        if input_type == 'image':
            return self.images_to_tokens(images)
        elif input_type == 'video':
            visual_tokens = self.videos_to_tokens(images)
            return visual_tokens

    def initialize_hand_tokenizer(self, tokenizer):
        # add [HAND_TRAJ] token to represent the segmentation token
        num_added_tokens = tokenizer.add_tokens(HAND_TOKEN_TEMPLATE)
        # update the segmentation token id
        hand_traj_token_idx = tokenizer(HAND_TOKEN_TEMPLATE, add_special_tokens=False).input_ids[0]
        self.resize_token_embeddings(len(tokenizer))
        self.config.hand_token_id = hand_traj_token_idx

    def initialize_pixel_tokenizer(self, tokenizer, n_bins):
        bin_tokens = [f"<bin_{i}>" for i in range(1, n_bins + 1)]
        num_added_tokens = tokenizer.add_tokens(bin_tokens)
        # Resize token embeddings
        self.resize_token_embeddings(len(tokenizer))


    def initialize_constants(self, fuse_input_mode, video_arch, lambda_obj,
                             lambda_obj_kl, lambda_traj, lambda_traj_kl, lambda_last_hand,
                             hoi_lambda,
                             ):
        self.config.fuse_input_mode = fuse_input_mode
        self.config.video_compress_mode = video_arch
        self.config.lambda_obj = lambda_obj
        self.config.lambda_obj_kl = lambda_obj_kl
        self.config.lambda_traj = lambda_traj
        self.config.lambda_traj_kl = lambda_traj_kl
        self.config.lambda_last_hand = lambda_last_hand
        self.config.hoi_lambda = hoi_lambda

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # this function is borrowed from llava_arch.py
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

