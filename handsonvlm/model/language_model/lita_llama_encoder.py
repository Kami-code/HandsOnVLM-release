from typing import List, Optional, Tuple, Union

import torch

from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from hoi_forecast.model.visual_to_tokens import VisualToTokenHelper
from handsonvlm.model.handsonvlm_arch import HandsOnVLMMetaForCausalLM
from handsonvlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaLlamaModel


class LitaLlamaForCausalLM_encoder(LlamaForCausalLM, HandsOnVLMMetaForCausalLM):
    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.token_dim = self.config.hidden_size
        self.is_inference = kwargs.get('is_inference', False)
        self.added_modules = []
        self.extra_kwargs = {}

        # Initialize weights and apply final processing
        # self.post_init()

    def added_modules_requires_grad_(self, requires_grad):
        for module in self.added_modules:
            for p in module.parameters():
                p.requires_grad = requires_grad

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            images: Optional[torch.FloatTensor] = None,
            feat: Optional[torch.FloatTensor] = None,
            bbox_feat: Optional[torch.FloatTensor] = None,
            valid_mask: Optional[torch.Tensor] = None,
            future_hands: Optional[torch.FloatTensor] = None,
            contact_point: Optional[torch.FloatTensor] = None,
            future_valid: Optional[torch.Tensor] = None,
            label_valid: Optional[torch.Tensor] = None,

            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        raise NotImplementedError

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, **kwargs):
        # B, T = input_ids.shape
        # assert input_ids.shape == torch.Size([B, T]), input_ids.shape
        # assert attention_mask.shape == torch.Size([B, T]), attention_mask.shape
        #
        # if labels is not None:
        #     assert labels.shape == torch.Size([B, T]), labels.shape
        # assert images.shape == torch.Size([B, 10, 3, 224, 224]), images.shape

        vision_tower = self.get_vision_tower()  # get the vision backbone
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        visual_to_token_helper = VisualToTokenHelper(images_raw_encode_fn=self.get_model().get_vision_tower(),
                                                     images_mm_projector_fn=self.get_model().mm_projector,
                                                     fuse_input_mode=self.config.fuse_input_mode,
                                                     video_compress_mode=self.config.video_compress_mode,
                                                     mm_hidden_size=self.config.mm_hidden_size,
                                                     token_dim=self.token_dim,
                                                     )

        visual_tokens, visual_token_attn_mask = visual_to_token_helper.pipeline(images=images, **kwargs)
        visual_token_num = visual_tokens.shape[1]
        assert visual_tokens.shape == torch.Size([B, visual_token_num, self.token_dim]), visual_tokens.shape

        # new_input_ids = []
        new_input_embeds = []
        new_labels = None if labels is None else []
        new_attention_mask = None if attention_mask is None else []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            assert cur_input_ids.shape == torch.Size([T]), cur_input_ids.shape
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    # new_input_ids.append(input_ids[batch_idx])
                    new_labels.append(labels[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                cur_image_idx += 1
                continue
            # we do have image tokens in the current single sample, torch where return a tuple
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            num_image_tokens_this_batch = image_token_indices.shape[0]

            assert image_token_indices.shape == torch.Size([num_image_tokens_this_batch]), image_token_indices.shape
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == torch.Size([T]), cur_labels.shape
            if attention_mask is not None:
                cur_attention_mask = attention_mask[batch_idx]
                cur_new_attention_mask = []
            while image_token_indices.numel() > 0:
                cur_image_features = visual_tokens[cur_image_idx]
                cur_image_features_token_num = cur_image_features.shape[0]
                assert cur_image_features.shape == torch.Size([cur_image_features_token_num, self.token_dim])
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    """
                    In this case, there is <im_start> and <im_end> tokens in the input_ids, we embed them separately
                    """
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    # todo: why detach?
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))

                    if labels is not None:
                        # set the labels for the image tokens to IGNORE_INDEX
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features_token_num,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # cur_image_features.shape[0] is the number of tokens in the image
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                    if attention_mask is not None:
                        cur_new_attention_mask.append(cur_attention_mask[:image_token_start])
                        cur_new_attention_mask.append(visual_token_attn_mask[cur_image_idx])
                        cur_new_attention_mask.append(cur_attention_mask[image_token_start:image_token_start + 1])
                        cur_attention_mask = cur_attention_mask[image_token_start + 2:]

                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features_token_num,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        self.last_visual_token_index = image_token_start + cur_image_features_token_num - 1

                        cur_labels = cur_labels[image_token_start + 1:]
                    if attention_mask is not None:
                        cur_new_attention_mask.append(cur_attention_mask[:image_token_start])
                        cur_new_attention_mask.append(visual_token_attn_mask[cur_image_idx])
                        cur_attention_mask = cur_attention_mask[image_token_start + 1:]

                cur_image_idx += 1

                # update cur_input_ids and image_token_indices, find the next image token
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            # no more image tokens in the current input_ids, we add the rest of the tokens
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                if attention_mask is not None:
                    cur_new_attention_mask.append(cur_attention_mask)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
            if attention_mask is not None:
                cur_new_attention_mask = torch.cat(cur_new_attention_mask, dim=0)
                new_attention_mask.append(cur_new_attention_mask)

            # add the cur_new_input_embeds to new_input_embeds as the result of the current batch

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # deal with the case of each sentence has different number of tokens, do the alignment
            # which means padding the input_id with zero and labels with IGNORE_INDEX
            max_len = max(x.shape[0] for x in new_input_embeds)  # find the max length of the input_ids

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            T_modified = new_input_embeds.shape[1]
            assert new_input_embeds.shape == torch.Size([B, T_modified, self.token_dim])

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
                assert new_labels.shape == torch.Size([B, T_modified]), new_labels.shape
            if attention_mask is not None:
                new_attention_mask_align = []
                _new_attention_mask = new_attention_mask
                for cur_new_attention_mask in new_attention_mask:
                    cur_new_attention_mask = torch.cat((cur_new_attention_mask, torch.full((max_len - cur_new_attention_mask.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_attention_mask_align.append(cur_new_attention_mask)
                new_attention_mask = torch.stack(new_attention_mask_align, dim=0)
                assert new_attention_mask.shape == torch.Size([B, T_modified]), new_attention_mask.shape

            # if attention_mask is not None:
            #     new_attention_mask = []
            #     for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
            #         assert cur_attention_mask.shape == torch.Size([T]), cur_attention_mask.shape
            #         assert cur_new_labels.shape == torch.Size([T]), cur_new_labels.shape
            #         assert cur_new_labels_align.shape == torch.Size([T_modified]), cur_new_labels_align.shape
            #
            #
            #         new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
            #         new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
            #         cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
            #         new_attention_mask.append(cur_new_attention_mask)
            #     attention_mask = torch.stack(new_attention_mask, dim=0)
            #     assert attention_mask.shape == torch.Size([B, T_modified]), attention_mask.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            if attention_mask is not None:
                new_attention_mask = torch.stack(new_attention_mask, dim=0)

            # if attention_mask is not None:
            #     new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            #     attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            #     assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, new_attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # print("input_ids in prepare for generation:", input_ids, input_ids.shape)

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                'is_evaluate': True
            }
        )
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs
