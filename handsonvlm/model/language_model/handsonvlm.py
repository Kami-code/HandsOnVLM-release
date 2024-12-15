import warnings
from typing import List, Optional, Tuple, Union

import wandb
import deepspeed
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation import validate_stopping_criteria
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


from handsonvlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from handsonvlm.model.language_model.lita_llama_hoi import LitaLlamaForCausalLM_hoi
from handsonvlm.model.language_model.traj_decoder import *
from hoi_forecast.model.visual_to_tokens import VisualToTokenHelper


class HandsOnVLMConfig(LlamaConfig):
    model_type = "handsonvlm"


class HandsOnVLMForCausalLM(LitaLlamaForCausalLM_hoi):
    config_class = HandsOnVLMConfig

    def __init__(self, config, **kwargs):
        super(LitaLlamaForCausalLM_hoi, self).__init__(config)

        #
        # vision_tower_cfg = argparse.Namespace()
        # vision_tower_cfg.mm_vision_select_layer = -2
        # self.vision_tower: CLIPVisionTower = CLIPVisionTower("openai/clip-vit-large-patch14", args=vision_tower_cfg)
        self.lm_head = nn.Linear(self.token_dim, config.vocab_size, bias=False)
        self.traj_decoder_name = kwargs.get('traj_decoder_name', 'MLP')
        self.hand_traj_decoder: TrajDecoder = None
        self.initialize_traj_decoder()
        self.hand_traj_positional_embedding = nn.Linear(2, self.token_dim // 2)

        max_len = 20
        self.time_embedding = nn.Parameter(torch.zeros(max_len, self.token_dim))

        self.added_modules.append(self.hand_traj_decoder)
        # Initialize weights and apply final processing
        self.post_init()

    def initialize_traj_decoder(self):
        if self.traj_decoder_name == 'MLP':
            self.hand_traj_decoder = MLPTrajDecoder(token_dim=self.token_dim // 2)
        elif self.traj_decoder_name == 'CVAE':
            self.hand_traj_decoder = CVAETrajDecoder(token_dim=self.token_dim // 2)
        else:
            raise NotImplementedError
        print(f"use {self.traj_decoder_name} as decoder!!")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            image: Optional[torch.FloatTensor] = None,
            future_hands: Optional[torch.FloatTensor] = None,
            contact_point: Optional[torch.FloatTensor] = None,
            future_valid: Optional[torch.Tensor] = None,
            gt_label_valid: Optional[torch.Tensor] = None,

            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.hand_traj_decoder is None:
            print("initialize traj decoder! in forward!")
            self.initialize_traj_decoder()

        self.B = image.shape[0]
        B = self.B
        if labels is not None:
            # assert labels.shape == torch.Size([B, T]), labels.shape
            assert future_hands.shape == torch.Size([B, 2, 5, 2]), future_hands.shape
            assert contact_point.shape == torch.Size([B, 2]), contact_point.shape
            assert future_valid.shape == torch.Size([B, 2, ]), future_valid.shape
            assert gt_label_valid.shape == torch.Size([B]), gt_label_valid.shape
            future_hands = future_hands[:, :, 1:, :]
        # assert attention_mask.shape == torch.Size([B, T]), f"attention_mask shape should be [{B}, {T}], but got {attention_mask.shape}"
        assert image.shape == torch.Size([B, 100, 3, 224, 224]), image.shape
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        T = input_ids.shape[1]
        # print("input_ids shape: ", input_ids.shape)
        # print("input_ids: ", input_ids)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image, future_hands=future_hands,
                                                                                                                      future_valid=future_valid, is_evaluate=kwargs.get('is_evaluate', False))
        T_modified = min(T + 356 - 1, 2048)
        # print("input_embeds shape: ", inputs_embeds.shape)
        # assert labels.shape == torch.Size([B, T_modified]), f"labels shape should be [{B}, {T_modified}], but got {labels.shape}"
        assert attention_mask.shape == torch.Size([B, T_modified]), f"attention_mask shape should be [{B}, {T_modified}], but got {input_ids.shape}"


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        assert hidden_states.shape == torch.Size([B, T_modified, self.token_dim]), hidden_states.shape
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_text = loss_fct(shift_logits, shift_labels)
            hand_traj_token_idx = 32100
            hand_traj_token_masks = (labels == hand_traj_token_idx)
            num_hand_tokens = hand_traj_token_masks.int().sum(-1)
            assert num_hand_tokens.shape == torch.Size([B]), num_hand_tokens.shape
            assert hand_traj_token_masks.shape == torch.Size([B, T_modified]), hand_traj_token_masks.shape

            assert hidden_states.shape == torch.Size([B, T_modified, self.token_dim]), hidden_states.shape

            # todo: following is a working implementation
            # shifted_hand_traj_token_mask = hand_traj_token_masks[:, 1:]
            # assert shifted_hand_traj_token_mask.shape == torch.Size([B, T_modified - 1]), shifted_hand_traj_token_mask.shape
            # flase_padding = torch.zeros((B, 1), device=shifted_hand_traj_token_mask.device).to(torch.bool)
            # shifted_hand_traj_token_mask = torch.cat([shifted_hand_traj_token_mask, flase_padding], dim=-1)
            # assert shifted_hand_traj_token_mask.shape == torch.Size([B, T_modified]), shifted_hand_traj_token_mask.shape
            # pred_hand_embeddings = hidden_states[shifted_hand_traj_token_mask].reshape(B, 4, self.token_dim // 2, 2)
            # pred_hand_embeddings = pred_hand_embeddings.permute(0, 3, 1, 2)
            # assert pred_hand_embeddings.shape == torch.Size([B, 2, 4, self.token_dim // 2]), pred_hand_embeddings.shape

            pred_hand_embeddings = []
            for i in range(B):
                # per-batch calculation
                shifted_hand_traj_token_mask_i = hand_traj_token_masks[i][1:]
                assert shifted_hand_traj_token_mask_i.shape == torch.Size([T_modified - 1]), shifted_hand_traj_token_mask_i.shape

                shifted_hand_traj_token_mask_i = torch.cat([shifted_hand_traj_token_mask_i, torch.tensor([False], device=shifted_hand_traj_token_mask_i.device)], dim=0)
                assert shifted_hand_traj_token_mask_i.shape == torch.Size([T_modified]), shifted_hand_traj_token_mask_i.shape
                hand_token_pred_num = shifted_hand_traj_token_mask_i.sum()
                cur_embeddings = hidden_states[i]
                assert cur_embeddings.shape == torch.Size([T_modified, self.token_dim])
                if hand_token_pred_num == 0:
                    # todo: just a placeholder here, do not join the loss calculation because of gt hand valid label
                    cur_pred_hand_embeddings = torch.zeros(2, 4, self.token_dim // 2).to(cur_embeddings.device)
                    future_valid[i] = torch.zeros(2,).to(future_valid.device).to(torch.bool)
                else:
                    cur_pred_hand_embeddings = cur_embeddings[shifted_hand_traj_token_mask_i].reshape(4, self.token_dim // 2, 2)
                    cur_pred_hand_embeddings = cur_pred_hand_embeddings.permute(2, 0, 1)
                assert cur_pred_hand_embeddings.shape == torch.Size([2, 4, self.token_dim // 2]), cur_pred_hand_embeddings.shape

                pred_hand_embeddings.append(cur_pred_hand_embeddings)

            pred_hand_embeddings = torch.stack(pred_hand_embeddings, dim=0)
            assert pred_hand_embeddings.shape == torch.Size([B, 2, 4, self.token_dim // 2]), pred_hand_embeddings.shape
            assert future_hands.shape == torch.Size([B, 2, 4, 2]), future_hands.shape
            assert future_valid.shape == torch.Size([B, 2]), future_valid.shape

            loss: dict = self.hand_traj_decoder(pred_hand_embeddings=pred_hand_embeddings, future_hands=future_hands, future_valid=future_valid,
                                                lambda_traj=self.config.lambda_traj, lambda_traj_kl=self.config.lambda_traj_kl)
            loss['text loss'] = loss_text

            deepspeed.comm.barrier()
            if wandb.run is not None:  # only log to the main process
                for k, v in loss.items():
                    wandb.log({f"train/{k}": v.cpu().item()}, commit=False)

            loss = self.config.hoi_lambda * loss['total_loss'] + loss['text loss']
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, **kwargs):
        vision_tower = self.get_vision_tower()  # get the vision backbone
        T = input_ids.shape[1]

        visual_to_token_helper = VisualToTokenHelper(images_raw_encode=self.get_model().get_vision_tower(),
                                                     images_mm_projector=self.get_model().mm_projector,
                                                     fuse_input_mode=self.config.fuse_input_mode,
                                                     video_compress_mode=self.config.video_compress_mode,
                                                     mm_hidden_size=self.config.mm_hidden_size,
                                                     token_dim=self.token_dim,
                                                     )

        visual_tokens, visual_token_attn_mask = visual_to_token_helper.pipeline(images=images, **kwargs)
        visual_token_num = visual_tokens.shape[1]
        assert visual_tokens.shape == torch.Size([self.B, visual_token_num, self.token_dim]), visual_tokens.shape

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
                    self.last_visual_token_index = image_token_start + cur_image_features_token_num
                    cur_new_input_embeds.append(cur_image_features)

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features_token_num,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
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


            def process_traj_positional_embedding(gt_hand):
                channels = self.token_dim // 4
                num_hands = gt_hand.shape[1]
                assert gt_hand.shape == torch.Size([2, num_hands, 2]), gt_hand.shape

                gt_hand_flat = gt_hand.reshape(-1, 2)
                gt_hand_x, gt_hand_y = gt_hand_flat[:, 0], gt_hand_flat[:, 1]
                assert gt_hand_x.shape == gt_hand_y.shape == torch.Size([2 * num_hands]), gt_hand_x.shape

                device = gt_hand.device
                dtype = gt_hand.dtype
                inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2, device=device, dtype=dtype) / channels))

                gt_hand_x_enc = gt_hand_x.unsqueeze(-1) * inv_freq
                gt_hand_y_enc = gt_hand_y.unsqueeze(-1) * inv_freq
                assert gt_hand_x_enc.shape == gt_hand_y_enc.shape == torch.Size([2 * num_hands, channels // 2]), gt_hand_x_enc.shape


                gt_pos_enc = torch.cat([torch.sin(gt_hand_x_enc), torch.cos(gt_hand_y_enc),
                                        torch.sin(gt_hand_x_enc), torch.cos(gt_hand_y_enc)], dim=-1)

                assert gt_pos_enc.shape == torch.Size([2 * num_hands, channels * 2]), gt_pos_enc.shape
                hand_traj_embeds = gt_pos_enc.reshape(2, num_hands, self.token_dim // 2)
                assert hand_traj_embeds.shape == torch.Size([2, num_hands, self.token_dim // 2]), hand_traj_embeds.shape
                hand_traj_embeds = hand_traj_embeds.permute(1, 2, 0)
                assert hand_traj_embeds.shape == torch.Size([num_hands, self.token_dim // 2, 2]), hand_traj_embeds.shape
                hand_traj_embeds = hand_traj_embeds.reshape(num_hands, self.token_dim // 2 * 2)
                assert hand_traj_embeds.shape == torch.Size([num_hands, self.token_dim]), hand_traj_embeds.shape
                return hand_traj_embeds

            # no more image tokens in the current input_ids, we add the rest of the tokens
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    text_embeds = self.get_model().embed_tokens(cur_input_ids)
                    T_cur_ids = cur_input_ids.shape[0]
                    is_evaluate = kwargs.get('is_evaluate', False)

                    hand_traj_token_id = 32100
                    hand_traj_mask = (cur_input_ids == hand_traj_token_id)
                    assert hand_traj_mask.shape == torch.Size([T_cur_ids]), hand_traj_mask.shape
                    hand_token_cnt = hand_traj_mask.sum()


                    if not is_evaluate:
                        # for training
                        gt_hand: torch.Tensor = kwargs.get('future_hands')[batch_idx]
                        gt_hand = gt_hand.detach().requires_grad_(False).clone().detach()
                        assert gt_hand.shape == torch.Size([2, 4, 2]), gt_hand.shape
                        future_valid = kwargs.get('future_valid')[batch_idx]
                        future_valid = future_valid.detach().requires_grad_(False).clone()
                        assert future_valid.shape == torch.Size([2]), future_valid.shape

                        hand_traj_embeds = process_traj_positional_embedding(gt_hand)
                        assert hand_traj_embeds.shape == torch.Size([4, self.token_dim]), hand_traj_embeds.shape
                        hand_traj_index = torch.where(hand_traj_mask)[0]

                        for _ in range(4 - hand_token_cnt):
                            hand_traj_index = torch.cat((hand_traj_index, torch.tensor([0], device=hand_traj_index.device)), dim=0)
                        hand_traj_index = hand_traj_index.unsqueeze(1).expand(-1, self.token_dim)
                        assert hand_traj_index.shape == torch.Size([4, self.token_dim]), hand_traj_index.shape
                        hand_traj_embeds = hand_traj_embeds * (hand_token_cnt / 4)
                        zero_tensor = torch.zeros_like(text_embeds)
                        hand_traj_embeds = zero_tensor.scatter(0, hand_traj_index, hand_traj_embeds)
                        text_embeds += hand_traj_embeds

                        # todo: following is a working implementation
                        # assert hand_traj_index.shape == torch.Size([hand_token_cnt]), hand_traj_index.shape
                        # text_embeds[hand_traj_index] += hand_traj_embeds
                    elif is_evaluate and kwargs.get('future_hands', None) is not None:
                        gt_hand: torch.Tensor = kwargs.get('future_hands')[batch_idx]
                        gt_hand = gt_hand.detach().requires_grad_(False).clone().detach()
                        gt_hand_num = gt_hand.shape[1]
                        # todo: this maybe a issue when the input_ids is clipped
                        assert gt_hand_num == hand_token_cnt, f"gt_hand_num: {gt_hand_num}, hand_token_cnt: {hand_token_cnt}"
                        assert gt_hand.shape == torch.Size([2, gt_hand_num, 2]), gt_hand.shape
                        future_valid = torch.ones(2).to(gt_hand.device).to(torch.bool)

                        hand_traj_embeds = process_traj_positional_embedding(gt_hand)
                        assert hand_traj_embeds.shape == torch.Size([gt_hand_num, self.token_dim]), hand_traj_embeds.shape
                        hand_traj_index = torch.where(hand_traj_mask)[0]
                        assert hand_traj_index.shape == torch.Size([gt_hand_num]), hand_traj_index.shape
                        hand_traj_index = hand_traj_index.unsqueeze(1).expand(-1, self.token_dim)
                        zero_tensor = torch.zeros_like(text_embeds)
                        hand_traj_embeds = zero_tensor.scatter(0, hand_traj_index, hand_traj_embeds)
                        text_embeds += hand_traj_embeds
                    cur_new_input_embeds.append(text_embeds)
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
            assert new_input_embeds.shape == torch.Size([self.B, T_modified, self.token_dim])

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
                assert new_labels.shape == torch.Size([self.B, T_modified]), new_labels.shape
            if attention_mask is not None:
                new_attention_mask_align = []
                _new_attention_mask = new_attention_mask
                for cur_new_attention_mask in new_attention_mask:
                    cur_new_attention_mask = torch.cat((cur_new_attention_mask, torch.full((max_len - cur_new_attention_mask.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_attention_mask_align.append(cur_new_attention_mask)
                new_attention_mask = torch.stack(new_attention_mask_align, dim=0)
                assert new_attention_mask.shape == torch.Size([self.B, T_modified]), new_attention_mask.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            if attention_mask is not None:
                new_attention_mask = torch.stack(new_attention_mask, dim=0)
        return None, new_attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}


        B = input_ids.shape[0]
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image": kwargs.get("image", None),
                'is_evaluate': True
            }
        )
        return model_inputs



    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        pred_hands = []
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"future_hands": torch.stack(pred_hands, dim=2) if len(pred_hands) > 0 else None})

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step

            # input_ids.shape is (1, T)
            # next_tokens[:, None].shape is (1, 1)
            # print("next_tokens", next_tokens)

            if next_tokens.detach().cpu().item() == 32100:
                # print("we find a <hand> token!!")
                batch_size = 1
                output_hidden_states_ = outputs.hidden_states[-1]
                T_modifed = output_hidden_states_.shape[1]
                assert output_hidden_states_.shape == torch.Size([batch_size, T_modifed, self.token_dim])
                pred_hand_embedding = output_hidden_states_[:, -1, :]
                assert pred_hand_embedding.shape == torch.Size([batch_size, self.token_dim])
                pred_hand_embedding = pred_hand_embedding.reshape(batch_size, self.token_dim // 2, 2)
                pred_hand_embedding = pred_hand_embedding.permute(0, 2, 1).unsqueeze(2)
                assert pred_hand_embedding.shape == torch.Size([batch_size, 2, 1, self.token_dim // 2]), pred_hand_embedding.shape
                pred_hand = self.hand_traj_decoder.inference(pred_hand_embeddings=pred_hand_embedding).squeeze(2)
                assert pred_hand.shape == torch.Size([batch_size, 2, 2]), pred_hand.shape
                pred_hands.append(pred_hand)


            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    pred_hands=torch.stack(pred_hands, dim=2) if len(pred_hands) > 0 else torch.empty(1, 2, 0, 2).to(input_ids.device)
                )
        else:
            return input_ids