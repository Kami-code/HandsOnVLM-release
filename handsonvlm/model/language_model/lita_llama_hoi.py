from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import wandb
import deepspeed
from transformers import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from hoi_forecast.architecture.affordance_decoder import AffordanceCVAE
from hoi_forecast.architecture.net_utils import get_subsequent_mask, get_pad_mask, traj_affordance_dist
from hoi_forecast.architecture.traj_decoder import TrajCVAE
from hoi_forecast.architecture.object_transformer import ObjectTransformerDecoder, ObjectTransformerEncoder
from handsonvlm.model.language_model.lita_llama_hoi_encoder import LitaLlamaForCausalLM_hoi_encoder
from hoi_forecast.model.visual_to_tokens import VisualToTokenHelper


class LitaConfig_hoi(LlamaConfig):
    model_type = "lita-hoi"


class LitaLlamaForCausalLM_hoi(LitaLlamaForCausalLM_hoi_encoder):
    config_class = LitaConfig_hoi

    def __init__(self, config, **kwargs):
        super(LitaLlamaForCausalLM_hoi, self).__init__(config, **kwargs)
        self.lm_head = nn.Linear(self.token_dim, config.vocab_size, bias=False)
        trg_in_features = 2
        dec_depth = 4
        num_heads = 8
        mlp_ratio = 4
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        dropout = 0.1
        decoder_time_embed_type = "sin"
        num_frames_output = 4
        hidden_dim = 512
        latent_dim = 256

        if self.is_inference:
            print("in inference mode!")
            self.oct_decoder = ObjectTransformerDecoder(in_features=trg_in_features, token_dim=self.token_dim,
                                                        depth=dec_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                                        attn_drop_rate=attn_drop_rate,
                                                        drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                                                        time_embed_type=decoder_time_embed_type, num_frames=num_frames_output)
        else:
            print("in training mode!")
            with deepspeed.zero.Init():
                self.oct_decoder = ObjectTransformerDecoder(in_features=trg_in_features, token_dim=self.token_dim,
                                                            depth=dec_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                                            attn_drop_rate=attn_drop_rate,
                                                            drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                                                            time_embed_type=decoder_time_embed_type, num_frames=num_frames_output)

        self.hand_traj_decoder = TrajCVAE(in_dim=2, hidden_dim=hidden_dim,
                                          latent_dim=latent_dim, token_dim=self.token_dim,
                                          coord_dim=self.coord_dim, condition_contact=False)

        self.affordance_decoder = AffordanceCVAE(in_dim=2, hidden_dim=hidden_dim,
                                                 latent_dim=latent_dim, token_dim=self.token_dim, condition_traj=True)
        self.last_obs_rhand_embedding_predictor = nn.Linear(self.token_dim, self.token_dim)
        self.last_obs_lhand_embedding_predictor = nn.Linear(self.token_dim, self.token_dim)

        self.added_modules.append(self.hand_traj_decoder)
        self.added_modules.append(self.affordance_decoder)
        self.added_modules.append(self.oct_decoder)
        self.added_modules.append(self.last_obs_rhand_embedding_predictor)
        self.added_modules.append(self.last_obs_lhand_embedding_predictor)
        # Initialize weights and apply final processing
        self.post_init()

    def _get_last_hidden_state(self, hidden_states, src_mask):
        last_frame_src_mask = src_mask[:, -1, :]
        assert last_frame_src_mask.shape == torch.Size([self.B, 5]), last_frame_src_mask.shape
        if self.config.fuse_input_mode.startswith("hoi-hand"):
            last_hidden_state = hidden_states[:, -3:, :]
            last_frame_src_mask = last_frame_src_mask[:, :3]
            last_hidden_state_mask = get_pad_mask(last_frame_src_mask, pad_idx=0)
        elif self.config.fuse_input_mode.startswith("hoi-global"):
            last_hidden_state = hidden_states[:, -1:, :]
            last_frame_src_mask = last_frame_src_mask[:, :1]
            last_hidden_state_mask = get_pad_mask(last_frame_src_mask, pad_idx=0)
        elif self.config.fuse_input_mode.startswith("hoi"):
            last_hidden_state = hidden_states[:, -5:, :]
            last_hidden_state_mask = get_pad_mask(last_frame_src_mask, pad_idx=0)
        else:
            raise NotImplementedError
        return last_hidden_state, last_hidden_state_mask

    def get_last_hidden_state(self, inputs_embeds, attention_mask, use_cache, output_attentions, output_hidden_states, return_dict, feat, valid_mask):
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        if not valid_mask.shape[1] == feat.shape[1]:
            # if valid does not contain global feature mask, then add all 1 mask for global feature
            src_mask = torch.cat(
                (torch.ones_like(valid_mask[:, 0:1, :], dtype=valid_mask.dtype, device=valid_mask.device),
                 valid_mask), dim=1).transpose(1, 2)
        else:
            src_mask = valid_mask.transpose(1, 2)
        assert src_mask.shape == torch.Size([self.B, self.T_observed, 5]), src_mask.shape

        last_hidden_state, last_hidden_state_mask = self._get_last_hidden_state(hidden_states, src_mask)
        return last_hidden_state, last_hidden_state_mask, outputs


    def forward_decoder_and_get_loss(self, future_hands, future_valid, contact_point, last_hidden_state, last_hidden_state_mask):
        gt_rhand, gt_lhand = future_hands[:, 0, :, :], future_hands[:, 1, :, :]
        assert gt_rhand.shape == gt_lhand.shape == torch.Size([self.B, self.T_pred, 2]), gt_rhand.shape
        gt_rhand_input = gt_rhand[:, :-1, :]
        gt_lhand_input = gt_lhand[:, :-1, :]
        assert gt_rhand_input.shape == gt_lhand_input.shape == torch.Size([self.B, self.T_pred - 1, 2]), gt_rhand_input.shape
        hand_embedding_mask = torch.ones_like(gt_rhand_input[:, :, 0])
        hand_embedding_mask = get_subsequent_mask(hand_embedding_mask)
        assert hand_embedding_mask.shape == torch.Size([1, self.T_pred - 1, self.T_pred - 1]), hand_embedding_mask.shape

        gt_rhand_embedding = self.oct_decoder(gt_rhand_input, last_hidden_state,
                                              last_hidden_state_mask=last_hidden_state_mask,
                                              hand_embedding_mask=hand_embedding_mask)
        gt_lhand_embedding = self.oct_decoder(gt_lhand_input, last_hidden_state,
                                              last_hidden_state_mask=last_hidden_state_mask,
                                              hand_embedding_mask=hand_embedding_mask)

        gt_hand_embedding = torch.cat((gt_rhand_embedding, gt_lhand_embedding), dim=1).reshape(-1, self.token_dim)
        assert gt_hand_embedding.shape == torch.Size([self.B * 2 * (self.T_pred - 1), self.token_dim]), gt_hand_embedding.shape

        gt_target_hand = future_hands[:, :, 1:, :].reshape(-1, 2)
        assert gt_target_hand.shape == torch.Size([self.B * 2 * (self.T_pred - 1), 2]), gt_target_hand.shape

        pred_hand, traj_loss, traj_kl_loss = self.hand_traj_decoder(gt_hand_embedding, gt_target_hand, future_valid, contact_point=None)
        assert pred_hand.shape == torch.Size([self.B * 2 * (self.T_pred - 1), 2]), pred_hand.shape
        assert traj_loss.shape == traj_kl_loss.shape == torch.Size([self.B]), traj_loss.shape

        last_frame_global_token = last_hidden_state[:, 0, :]
        assert last_frame_global_token.shape == torch.Size([self.B, self.token_dim]), last_frame_global_token.shape

        pred_last_obs_rhand_embedding = self.last_obs_rhand_embedding_predictor(last_frame_global_token)
        pred_last_obs_lhand_embedding = self.last_obs_lhand_embedding_predictor(last_frame_global_token)
        mse_loss = nn.MSELoss(reduce=False)
        gt_last_obs_rhand_embedding = gt_rhand_embedding[:, 0, :]
        gt_last_obs_lhand_embedding = gt_lhand_embedding[:, 0, :]
        rhand_loss = mse_loss(pred_last_obs_rhand_embedding, gt_last_obs_rhand_embedding)
        lhand_loss = mse_loss(pred_last_obs_lhand_embedding, gt_last_obs_lhand_embedding)
        last_hand_loss = torch.stack([rhand_loss, lhand_loss], dim=1)
        last_hand_loss = last_hand_loss.mean(dim=2)
        assert last_hand_loss.shape == torch.Size([self.B, 2]), last_hand_loss.shape

        r_pred_contact, r_obj_loss, r_obj_kl_loss = self.affordance_decoder(last_frame_global_token, contact_point, gt_rhand)
        l_pred_contact, l_obj_loss, l_obj_kl_loss = self.affordance_decoder(last_frame_global_token, contact_point, gt_lhand)
        assert r_pred_contact.shape == l_pred_contact.shape == torch.Size([self.B, 2]), r_pred_contact.shape
        assert r_obj_loss.shape == l_obj_loss.shape == r_obj_kl_loss.shape == l_obj_kl_loss.shape == torch.Size([self.B]), r_obj_loss.shape


        obj_loss = torch.stack([r_obj_loss, l_obj_loss], dim=1)
        assert obj_loss.shape == torch.Size([self.B, 2]), obj_loss.shape
        obj_kl_loss = torch.stack([r_obj_kl_loss, l_obj_kl_loss], dim=1)
        obj_loss[~(future_valid > 0)] = 1e9

        selected_obj_loss, selected_idx = obj_loss.min(dim=1)
        selected_valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_obj_kl_loss = torch.gather(obj_kl_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_last_hand_loss = torch.gather(last_hand_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        assert selected_obj_loss.shape == selected_idx.shape == selected_valid.shape == selected_obj_loss.shape == selected_obj_kl_loss.shape == torch.Size([self.B]), selected_valid.shape

        obj_loss = selected_obj_loss * selected_valid
        obj_kl_loss = selected_obj_kl_loss * selected_valid
        last_hand_loss = selected_last_hand_loss * selected_valid

        obj_loss = self.config.lambda_obj * obj_loss.sum()
        obj_kl_loss = self.config.lambda_obj_kl * obj_kl_loss.sum()
        traj_loss = self.config.lambda_traj * traj_loss.sum()
        traj_kl_loss = self.config.lambda_traj_kl * traj_kl_loss.sum()
        last_hand_loss = self.config.lambda_last_hand * last_hand_loss.sum()

        loss = {}
        loss['obj_loss'] = obj_loss
        loss['obj_kl_loss'] = obj_kl_loss
        loss['traj_loss'] = traj_loss
        loss['traj_kl_loss'] = traj_kl_loss
        loss['last_hand_loss'] = last_hand_loss
        loss['hoi_forecast_loss'] = loss['traj_loss'] + loss['traj_kl_loss'] + loss['obj_loss'] + loss['obj_kl_loss'] + loss['last_hand_loss']

        if wandb.run is not None:  # only log to the main process
            wandb.log({"train/traj_loss": loss['traj_loss'].cpu().item(),
                       "train/traj_kl_loss": loss['traj_kl_loss'].cpu().item(),
                       "train/obj_loss": loss['obj_loss'].cpu().item(),
                       'train/obj_kl_loss': loss['obj_kl_loss'].cpu().item(),
                       'train/total_loss': loss['hoi_forecast_loss'].cpu().item(),
                       'train/last_hand_loss': loss['last_hand_loss'].cpu().item()},
                      commit=False)
        loss = self.config.hoi_lambda * loss['hoi_forecast_loss']
        return loss


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            image: Optional[torch.FloatTensor] = None,
            feat: Optional[torch.FloatTensor] = None,
            bbox_feat: Optional[torch.FloatTensor] = None,
            valid_mask: Optional[torch.Tensor] = None,
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
        """
        images: (B, 10, 3, 224, 224), we samplevideo_hand_to_tokens T frames from each video
        hands: (B, T, max_num_hands, 1280, 16, 12), we use HaMeR to extract the hand features
        input_ids: (B, L) where L is the length of the input sequence, each element is the token id in the vocabulary
        attention_mask: (B, L) where L is the length of the input sequence, each element is 0 or 1 indicating whether the corresponding token is a padding token
        past_key_values: a list of tuples, each tuple contains the past key and value tensors for each layer
        labels: (B, L) where L is the length of the input sequence, each element is the token id in the vocabulary


        # feat: (B, 5, T_obs, 1024), global, hand & obj, T_obs=10
        # bbox_feat: (B, 4, T_obs, 4), hand & objT_obs
        # valid_mask: (B, 4, T_obs), hand & obj / (B, 5, T_obs), hand & obj & global
        # future_hands: (B, 2, T_future, 2) right & left, T=5 (contain last observation frame)
        # contact_points: (B, 2)
        # future_valid: (B, 2), right & left traj valid
        label_valid: (B), whether the label is valid
        """
        self.T = input_ids.shape[1]
        B, self.T_observed = feat.shape[0], feat.shape[2]
        self.B = B

        assert attention_mask.shape == torch.Size([B, self.T]), "attention_mask shape should be [B, T], but got {}".format(attention_mask.shape)
        if labels is not None:
            self.T_pred = 5
            assert labels.shape == torch.Size([B, self.T]), labels.shape
            assert feat.shape == torch.Size([B, 5, self.T_observed, 1024]), feat.shape
            assert bbox_feat.shape == torch.Size([B, 4, self.T_observed, 4]), bbox_feat.shape
            assert valid_mask.shape == torch.Size([B, 5, self.T_observed]), valid_mask.shape
            assert future_hands.shape == torch.Size([B, 2, 5, 2]), future_hands.shape
            assert contact_point.shape == torch.Size([B, 2]), contact_point.shape
            assert future_valid.shape == torch.Size([B, 2, ]), future_valid.shape
            assert gt_label_valid.shape == torch.Size([B]), gt_label_valid.shape

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # insert the image tokens into the input_ids
        kwargs = {
            "feat": feat,
            "bbox_feat": bbox_feat,
            "valid_mask": valid_mask,
            'extra_kwargs': self.extra_kwargs,
        }

        _, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image, **kwargs)
        last_hidden_state, last_hidden_state_mask, outputs = self.get_last_hidden_state(inputs_embeds, attention_mask, use_cache, output_attentions, output_hidden_states, return_dict, feat, valid_mask)

        loss = None
        if labels is not None:
            loss = self.forward_decoder_and_get_loss(future_hands, future_valid, contact_point, last_hidden_state, last_hidden_state_mask)

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def inference(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            image: Optional[torch.FloatTensor] = None,
            feat: Optional[torch.FloatTensor] = None,
            bbox_feat: Optional[torch.FloatTensor] = None,
            valid_mask: Optional[torch.Tensor] = None,
            gt_hands: Optional[torch.FloatTensor] = None,
            gt_contact_point: Optional[torch.FloatTensor] = None,
            gt_hand_valid: Optional[torch.Tensor] = None,
            gt_label_valid: Optional[torch.Tensor] = None,

            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        images: (B, 10, 3, 224, 224), we samplevideo_hand_to_tokens T frames from each video
        hands: (B, T, max_num_hands, 1280, 16, 12), we use HaMeR to extract the hand features
        input_ids: (B, L) where L is the length of the input sequence, each element is the token id in the vocabulary
        attention_mask: (B, L) where L is the length of the input sequence, each element is 0 or 1 indicating whether the corresponding token is a padding token
        past_key_values: a list of tuples, each tuple contains the past key and value tensors for each layer
        labels: (B, L) where L is the length of the input sequence, each element is the token id in the vocabulary


        # feat: (B, 5, T_obs, 1024), global, hand & obj, T_obs=10
        # bbox_feat: (B, 4, T_obs, 4), hand & objT_obs
        # valid_mask: (B, 4, T_obs), hand & obj / (B, 5, T_obs), hand & obj & global
        # future_hands: (B, 2, T_future, 2) right & left, T=5 (contain last observation frame)
        # contact_points: (B, 2)
        # future_valid: (B, 2), right & left traj valid
        label_valid: (B), whether the label is valid
        """
        self.B = feat.shape[0]
        self.T_observed = feat.shape[2]
        assert feat.shape == torch.Size([self.B, 5, self.T_observed, 1024]), feat.shape
        assert bbox_feat.shape == torch.Size([self.B, 4, self.T_observed, 4]), bbox_feat.shape
        assert valid_mask.shape == torch.Size([self.B, 5, self.T_observed]), valid_mask.shape

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # insert the image tokens into the input_ids
        kwargs = {
            "feat": feat,
            "bbox_feat": bbox_feat,
            "valid_mask": valid_mask,
            'extra_kwargs': self.extra_kwargs,
        }

        _, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image, **kwargs)
        last_hidden_state, last_hidden_state_mask, outputs = self.get_last_hidden_state(inputs_embeds, attention_mask, use_cache, output_attentions, output_hidden_states, return_dict, feat, valid_mask)

        pred_len = 4

        last_frame_global_token = last_hidden_state[:, 0, :]
        pred_last_obs_rhand_embedding = self.last_obs_rhand_embedding_predictor(last_frame_global_token)
        pred_last_obs_lhand_embedding = self.last_obs_lhand_embedding_predictor(last_frame_global_token)
        assert pred_last_obs_rhand_embedding.shape == pred_last_obs_lhand_embedding.shape == torch.Size([self.B, self.token_dim]), f"{pred_last_obs_rhand_embedding.shape}, {torch.Size([self.B, self.token_dim])}"

        pred_rhand_first = self.hand_traj_decoder.inference(pred_last_obs_rhand_embedding, contact_point=None)
        pred_lhand_first = self.hand_traj_decoder.inference(pred_last_obs_lhand_embedding, contact_point=None)

        pred_rhand_by_now = pred_rhand_first.unsqueeze(1)
        pred_lhand_by_now = pred_lhand_first.unsqueeze(1)
        assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([self.B, 1, 2]), pred_rhand_by_now.shape
        for i in range(pred_len):
            assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([self.B, i + 1, 2]), pred_rhand_by_now.shape
            hand_embedding_mask = torch.ones_like(pred_rhand_by_now[:, :, 0])
            assert hand_embedding_mask.shape == torch.Size([self.B, i + 1]), hand_embedding_mask.shape
            hand_embedding_mask = get_subsequent_mask(hand_embedding_mask)
            assert hand_embedding_mask.shape == torch.Size([1, i + 1, i + 1]), hand_embedding_mask.shape
            pred_rhand_embedding = self.oct_decoder(pred_rhand_by_now, last_hidden_state, last_hidden_state_mask=last_hidden_state_mask, hand_embedding_mask=hand_embedding_mask)
            pred_lhand_embedding = self.oct_decoder(pred_lhand_by_now, last_hidden_state, last_hidden_state_mask=last_hidden_state_mask, hand_embedding_mask=hand_embedding_mask)
            assert pred_rhand_embedding.shape == pred_lhand_embedding.shape == torch.Size([self.B, i + 1, self.token_dim]), pred_rhand_embedding.shape
            pred_rhand_embedding = pred_rhand_embedding.reshape(self.B * (i + 1), self.token_dim)
            pred_lhand_embedding = pred_lhand_embedding.reshape(self.B * (i + 1), self.token_dim)
            assert pred_rhand_embedding.shape == pred_lhand_embedding.shape == torch.Size([self.B * (i + 1), self.token_dim]), pred_rhand_embedding.shape
            pred_rhand = self.hand_traj_decoder.inference(pred_rhand_embedding, contact_point=None)
            pred_lhand = self.hand_traj_decoder.inference(pred_lhand_embedding, contact_point=None)
            pred_rhand = pred_rhand.reshape(self.B, (i + 1), 2)
            pred_lhand = pred_lhand.reshape(self.B, (i + 1), 2)
            assert pred_rhand.shape == pred_lhand.shape == torch.Size([self.B, i + 1, 2]), pred_rhand.shape
            cur_pred_rhand = pred_rhand[:, -1:, :]
            cur_pred_lhand = pred_lhand[:, -1:, :]
            pred_rhand_by_now = torch.cat((pred_rhand_by_now, cur_pred_rhand), dim=1)
            pred_lhand_by_now = torch.cat((pred_lhand_by_now, cur_pred_lhand), dim=1)
            assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([self.B, i + 2, 2]), pred_rhand_by_now.shape

        pred_hand = torch.stack((pred_rhand_by_now[:, 1:, :], pred_lhand_by_now[:, 1:, :]), dim=1)

        r_pred_contact = self.affordance_decoder.inference(last_hidden_state[:, 0, :], pred_rhand_by_now)
        l_pred_contact = self.affordance_decoder.inference(last_hidden_state[:, 0, :], pred_lhand_by_now)
        pred_contact = torch.stack([r_pred_contact, l_pred_contact], dim=1)

        gt_hand_traj_valid = gt_hand_valid.any(dim=-1)
        assert gt_hand_traj_valid.shape == torch.Size([self.B, 2]), gt_hand_traj_valid.shape

        if gt_hand_traj_valid is not None and torch.all(gt_hand_traj_valid.sum(dim=1) >= 1):
            r_pred_contact_dist = traj_affordance_dist(pred_hand.reshape(-1, 2), r_pred_contact, gt_hand_traj_valid)
            l_pred_contact_dist = traj_affordance_dist(pred_hand.reshape(-1, 2), l_pred_contact, gt_hand_traj_valid)
            pred_contact_dist = torch.stack((r_pred_contact_dist, l_pred_contact_dist), dim=1)
            _, selected_idx = pred_contact_dist.min(dim=1)
            selected_idx = selected_idx.unsqueeze(dim=1).unsqueeze(dim=2).expand(pred_contact.shape[0], 1,
                                                                                 pred_contact.shape[2])
            pred_contact = torch.gather(pred_contact, dim=1, index=selected_idx).squeeze(dim=1)

        return pred_hand, pred_contact

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, **kwargs):
        assert self.config.video_compress_mode in ['none'], f"only support none mode, current is {self.config.video_compress_mode}"
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
        assert visual_token_attn_mask.shape == torch.Size([self.B, visual_token_num]), visual_token_attn_mask.shape
        return None, visual_token_attn_mask, None, visual_tokens, None
