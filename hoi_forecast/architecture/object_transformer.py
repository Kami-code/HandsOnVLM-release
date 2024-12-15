import torch
import torch.nn as nn
from einops import rearrange

from hoi_forecast.architecture.embedding import PositionalEncoding, Encoder_PositionalEmbedding, Decoder_PositionalEmbedding
from hoi_forecast.architecture.layer import EncoderBlock, DecoderBlock
from hoi_forecast.architecture.net_utils import trunc_normal_, get_pad_mask, get_subsequent_mask, traj_affordance_dist
from hoi_forecast.architecture.traj_decoder import TrajCVAE
from hoi_forecast.architecture.affordance_decoder import AffordanceCVAE


class ObjectTransformerEncoder(nn.Module):
    def __init__(self, num_patches=5, token_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 dropout=0., time_embed_type=None, num_frames=None):
        super().__init__()
        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        self.num_patches = num_patches  # (hand, object global feature patches, default: 5)
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.token_dim = token_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, token_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Encoder_PositionalEmbedding(token_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(token_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            dim=token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(token_dim)
        trunc_normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, token, valid_mask=None):
        B, T, N, _ = token.shape
        assert token.shape == torch.Size([B, T, N, self.token_dim]), f"{token.shape}, {torch.Size([B, T, N, self.token_dim])}"
        assert valid_mask.shape == torch.Size([B, T, N]), valid_mask.shape

        token = rearrange(token, 'b t n m -> (b t) n m', b=B, t=T, n=N)
        assert token.shape == torch.Size([B * T, N, self.token_dim]), token.shape

        assert self.pos_embed.shape == torch.Size([1, N, self.token_dim]), self.pos_embed.shape
        token = token + self.pos_embed
        token = self.pos_drop(token)
        assert token.shape == torch.Size([B * T, N, self.token_dim]), token.shape

        token = rearrange(token, '(b t) n m -> (b n) t m', b=B, t=T)
        token = self.time_embed(token)
        token = self.time_drop(token)
        token = rearrange(token, '(b n) t m -> b (n t) m', b=B, t=T)
        assert token.shape == torch.Size([B, N * T, self.token_dim]), token.shape

        valid_mask = valid_mask.transpose(1, 2)
        for blk in self.encoder_blocks:
            token = blk(token, B, T, N, mask=valid_mask)

        assert token.shape == torch.Size([B, N * T, self.token_dim]), token.shape
        token = rearrange(token, 'b (n t) m -> b t n m', b=B, t=T, n=N)
        token = self.norm(token)
        assert token.shape == torch.Size([B, T, N, self.token_dim]), token.shape
        return token


class ObjectTransformerDecoder(nn.Module):
    def __init__(self, in_features, token_dim, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 time_embed_type=None, num_frames=None):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.token_dim = token_dim
        self.in_features = in_features

        self.target_embedding = nn.Linear(self.in_features, self.token_dim)

        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Decoder_PositionalEmbedding(self.token_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(self.token_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.decoder_blocks = nn.ModuleList([DecoderBlock(
            dim=self.token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(self.token_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, single_hand, last_hidden_state, last_hidden_state_mask=None, hand_embedding_mask=None):
        B, T_pred = single_hand.shape[0], single_hand.shape[1] + 1
        last_frame_tokens = last_hidden_state.shape[1]

        assert single_hand.shape == torch.Size([B, T_pred - 1, self.in_features]), single_hand.shape
        assert last_hidden_state.shape == torch.Size([B, last_frame_tokens, self.token_dim]), last_hidden_state.shape
        assert last_hidden_state_mask.shape == torch.Size([B, 1, last_frame_tokens]), last_hidden_state_mask.shape
        assert hand_embedding_mask.shape == torch.Size([1, T_pred - 1, T_pred - 1]), hand_embedding_mask.shape

        hand_embedding = self.time_drop(self.time_embed(self.target_embedding(single_hand)))
        assert hand_embedding.shape == torch.Size([B, T_pred - 1, self.token_dim]), hand_embedding.shape

        for block in self.decoder_blocks:
            hand_embedding = block(hand_embedding, last_hidden_state, last_hidden_state_mask=last_hidden_state_mask, hand_embedding_mask=hand_embedding_mask)
        hand_embedding = self.norm(hand_embedding)

        assert hand_embedding.shape == torch.Size([B, T_pred - 1, self.token_dim]), hand_embedding.shape
        return hand_embedding


class ObjectTransformer(nn.Module):
    def __init__(self, src_in_features, trg_in_features, num_patches,
                 hand_head, obj_head,
                 token_dim=512, coord_dim=64, num_heads=8, enc_depth=6, dec_depth=4,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 encoder_time_embed_type='sin', decoder_time_embed_type='sin',
                 num_frames_input=None, num_frames_output=None):
        super().__init__()

        self.src_in_features = src_in_features
        self.token_dim = token_dim
        self.coord_dim = coord_dim
        self.downproject = nn.Linear(self.src_in_features, self.token_dim)
        self.num_patches = num_patches

        self.bbox_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim // 2, self.coord_dim),
            nn.ELU()
        )
        self.feat_fusion = nn.Sequential(
            nn.Linear(self.token_dim + self.coord_dim, self.token_dim),
            nn.ELU(inplace=True))

        self.oct_encoder = ObjectTransformerEncoder(num_patches=num_patches,
                                                    token_dim=self.token_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                                                    time_embed_type=encoder_time_embed_type, num_frames=num_frames_input)

        self.oct_decoder = ObjectTransformerDecoder(in_features=trg_in_features, token_dim=self.token_dim,
                                                    depth=dec_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                                    attn_drop_rate=attn_drop_rate,
                                                    drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                                                    time_embed_type=decoder_time_embed_type, num_frames=num_frames_output)



        self.last_obs_rhand_embedding_predictor = nn.Linear(self.token_dim, self.token_dim)
        self.last_obs_lhand_embedding_predictor = nn.Linear(self.token_dim, self.token_dim)

        self.hand_head: TrajCVAE = hand_head
        self.object_head: AffordanceCVAE = obj_head
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_input(self, feat, bbox_feat, src_mask):
        """
        feat: (B, 5, T, src_in_features), global, hand & obj, T=10
        bbox_feat: (B, 4, T, 4), hand & obj

        return:
        feat: (B, T, 5, embed_dim), hand object with bbox feature concat with global image feature
        """
        B, T = feat.shape[0], feat.shape[2]
        assert feat.shape == torch.Size([B, 5, T, self.src_in_features]), feat.shape
        assert src_mask.shape == torch.Size([B, T, 5]), src_mask.shape
        feat = self.downproject(feat)
        assert feat.shape == torch.Size([B, 5, T, self.token_dim]), feat.shape

        assert bbox_feat.shape == torch.Size([B, 4, T, 4]), bbox_feat.shape
        bbox_feat = bbox_feat.view(-1, 4)
        assert bbox_feat.shape == torch.Size([B * 4 * T, 4]), bbox_feat.shape
        bbox_feat = self.bbox_to_feature(bbox_feat)
        assert bbox_feat.shape == torch.Size([B * 4 * T, self.coord_dim]), bbox_feat.shape
        bbox_feat = bbox_feat.view(B, -1, T, self.coord_dim)
        assert bbox_feat.shape == torch.Size([B, 4, T, self.coord_dim]), bbox_feat.shape

        ho_feat = feat[:, 1:, :, :]
        global_feat = feat[:, 0:1, :, :]
        assert ho_feat.shape == torch.Size([B, 4, T, self.token_dim]), ho_feat.shape
        assert global_feat.shape == torch.Size([B, 1, T, self.token_dim]), global_feat.shape

        feat = torch.cat((ho_feat, bbox_feat), dim=-1)
        assert feat.shape == torch.Size([B, 4, T, self.token_dim + self.coord_dim]), feat.shape
        feat = feat.view(-1, self.token_dim + self.coord_dim)
        assert feat.shape == torch.Size([B * 4 * T, self.token_dim + self.coord_dim]), feat.shape

        feat = self.feat_fusion(feat)
        assert feat.shape == torch.Size([B * 4 * T, self.token_dim]), feat.shape
        feat = feat.view(B, -1, T, self.token_dim)
        assert feat.shape == torch.Size([B, 4, T, self.token_dim]), feat.shape

        feat = torch.cat((global_feat, feat), dim=1)
        assert feat.shape == torch.Size([B, 5, T, self.token_dim]), feat.shape
        feat = feat.transpose(1, 2)
        assert feat.shape == torch.Size([B, T, 5, self.token_dim]), feat.shape
        return feat, src_mask

    def forward(self,
                feat,
                bbox_feat,
                valid_mask,
                future_hands,
                contact_point,
                future_valid,
                **kwargs):
        """
            feat: (B, 5, T_obs, src_in_features), global, hand & obj, T=10
            bbox_feat: (B, 4, T_obs, 4), hand & obj
            valid_mask: (B, 4, T_obs), hand & obj / (B, 5, T), hand & obj & global
            future_hands: (B, 2, T_future, 2) right & left, T=5 (contain last observation frame)
            contact_points: (B, 2)
            future_valid: (B, 2), right & left traj valid
            return: traj_loss: (B), obj_loss (B)
        """
        B, T_observed = feat.shape[0], feat.shape[2]
        T_pred = future_hands.shape[2]
        assert feat.shape == torch.Size([B, 5, T_observed, 1024]), feat.shape
        assert bbox_feat.shape == torch.Size([B, 4, T_observed, 4]), bbox_feat.shape
        assert valid_mask.shape == torch.Size([B, 5, T_observed]), valid_mask.shape
        assert future_hands.shape == torch.Size([B, 2, T_pred, 2]), future_hands.shape
        assert contact_point.shape == torch.Size([B, 2]), contact_point.shape
        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape

        gt_rhand, gt_lhand = future_hands[:, 0, :, :], future_hands[:, 1, :, :]
        assert gt_rhand.shape == gt_lhand.shape == torch.Size([B, T_pred, 2]), gt_rhand.shape

        if not valid_mask.shape[1] == feat.shape[1]:
            # if valid does not contain global feature mask, then add all 1 mask for global feature
            src_mask = torch.cat(
                (torch.ones_like(valid_mask[:, 0:1, :], dtype=valid_mask.dtype, device=valid_mask.device),
                 valid_mask), dim=1).transpose(1, 2)
        else:
            src_mask = valid_mask.transpose(1, 2)  # todo: why transpose here?
        assert src_mask.shape == torch.Size([B, T_observed, 5]), src_mask.shape
        feat, src_mask = self.encoder_input(feat, bbox_feat, src_mask)
        assert feat.shape == torch.Size([B, T_observed, self.num_patches, self.token_dim]), feat.shape
        hidden_states = self.oct_encoder(token=feat, valid_mask=src_mask)
        assert hidden_states.shape == torch.Size([B, T_observed, self.num_patches, self.token_dim])
        last_hidden_state = hidden_states[:, -1, :, :]  # the feature of the last observation frame
        assert last_hidden_state.shape == torch.Size([B, self.num_patches, self.token_dim]), last_hidden_state.shape
        memory_mask = get_pad_mask(src_mask[:, -1, :], pad_idx=0)
        assert memory_mask.shape == torch.Size([B, 1, self.num_patches]), memory_mask.shape

        gt_rhand_input = gt_rhand[:, :-1, :]
        gt_lhand_input = gt_lhand[:, :-1, :]
        assert gt_rhand_input.shape == gt_lhand_input.shape == torch.Size([B, T_pred - 1, 2]), gt_rhand_input.shape
        trg_mask = torch.ones_like(gt_rhand_input[:, :, 0])
        trg_mask = get_subsequent_mask(trg_mask)
        assert trg_mask.shape == torch.Size([1, T_pred - 1, T_pred - 1]), trg_mask.shape
        # this mask is used to mask the future information

        # send GT future hand trajectory to the decoder is because of teacher forcing
        gt_rhand_embedding = self.oct_decoder(gt_rhand_input, last_hidden_state,
                                              last_hidden_state_mask=memory_mask,
                                              hand_embedding_mask=trg_mask)
        gt_lhand_embedding = self.oct_decoder(gt_lhand_input, last_hidden_state,
                                              last_hidden_state_mask=memory_mask,
                                              hand_embedding_mask=trg_mask)
        assert gt_rhand_embedding.shape == gt_lhand_embedding.shape == torch.Size([B, T_pred - 1, self.token_dim]), gt_rhand_embedding.shape

        gt_hand_embedding = torch.cat((gt_rhand_embedding, gt_lhand_embedding), dim=1).reshape(-1, self.token_dim)
        assert gt_hand_embedding.shape == torch.Size([B * 2 * (T_pred - 1), self.token_dim]), gt_hand_embedding.shape

        gt_target_hand = future_hands[:, :, 1:, :].reshape(-1, 2)
        assert gt_target_hand.shape == torch.Size([B * 2 * (T_pred - 1), 2]), gt_target_hand.shape

        pred_hand, traj_loss, traj_kl_loss = self.hand_head(gt_hand_embedding, gt_target_hand, future_valid, contact_point=None)
        assert pred_hand.shape == torch.Size([B * 2 * (T_pred - 1), 2]), pred_hand.shape
        assert traj_loss.shape == traj_kl_loss.shape == torch.Size([B]), traj_loss.shape

        last_frame_global_token = last_hidden_state[:, 0, :]
        assert last_frame_global_token.shape == torch.Size([B, self.token_dim]), last_frame_global_token.shape

        pred_last_obs_rhand_embedding = self.last_obs_rhand_embedding_predictor(last_frame_global_token)
        pred_last_obs_lhand_embedding = self.last_obs_lhand_embedding_predictor(last_frame_global_token)
        mse_loss = nn.MSELoss(reduce=False)
        gt_last_obs_rhand_embedding = gt_rhand_embedding[:, 0, :]
        gt_last_obs_lhand_embedding = gt_lhand_embedding[:, 0, :]
        rhand_loss = mse_loss(pred_last_obs_rhand_embedding, gt_last_obs_rhand_embedding)
        lhand_loss = mse_loss(pred_last_obs_lhand_embedding, gt_last_obs_lhand_embedding)
        last_hand_loss = torch.stack([rhand_loss, lhand_loss], dim=1)
        last_hand_loss = last_hand_loss.mean(dim=2)
        assert last_hand_loss.shape == torch.Size([B, 2]), last_hand_loss.shape

        r_pred_contact, r_obj_loss, r_obj_kl_loss = self.object_head(last_frame_global_token, contact_point, gt_rhand)
        l_pred_contact, l_obj_loss, l_obj_kl_loss = self.object_head(last_frame_global_token, contact_point, gt_lhand)
        assert r_pred_contact.shape == l_pred_contact.shape == torch.Size([B, 2]), r_pred_contact.shape
        assert r_obj_loss.shape == l_obj_loss.shape == r_obj_kl_loss.shape == l_obj_kl_loss.shape == torch.Size([B]), r_obj_loss.shape

        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape

        obj_loss = torch.stack([r_obj_loss, l_obj_loss], dim=1)
        obj_kl_loss = torch.stack([r_obj_kl_loss, l_obj_kl_loss], dim=1)
        assert obj_loss.shape == obj_kl_loss.shape == torch.Size([B, 2]), obj_loss.shape
        obj_loss[~(future_valid.sum() > 0)] = 1e9

        selected_obj_loss, selected_idx = obj_loss.min(dim=1)
        selected_valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_obj_kl_loss = torch.gather(obj_kl_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        assert selected_obj_loss.shape == selected_obj_kl_loss.shape == selected_idx.shape == torch.Size([B]), selected_obj_loss.shape
        obj_loss = selected_obj_loss * selected_valid
        obj_kl_loss = selected_obj_kl_loss * selected_valid

        selected_last_hand_loss = torch.gather(last_hand_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        last_hand_loss = selected_last_hand_loss * selected_valid
        assert obj_loss.shape == obj_kl_loss.shape == torch.Size([B]), obj_loss.shape
        return traj_loss, traj_kl_loss, obj_loss, obj_kl_loss, last_hand_loss

    def inference(self,
                  feat,
                  bbox_feat,
                  valid_mask,
                  future_valid=None,
                  pred_len=4,
                  **kwargs):
        B, T_observed = feat.shape[0], feat.shape[2]
        assert feat.shape == torch.Size([B, 5, T_observed, 1024]), feat.shape
        assert bbox_feat.shape == torch.Size([B, 4, T_observed, 4]), bbox_feat.shape
        # if gt_hand_valid is not None:
        #     assert gt_hand_valid.shape == torch.Size([B, 2]), gt_hand_valid.shape
        assert valid_mask.shape == torch.Size([B, 5, T_observed]), valid_mask.shape

        if not valid_mask.shape[1] == feat.shape[1]:
            src_mask = torch.cat(
                (torch.ones_like(valid_mask[:, 0:1, :], dtype=valid_mask.dtype, device=valid_mask.device),
                 valid_mask), dim=1).transpose(1, 2)
        else:
            src_mask = valid_mask.transpose(1, 2)
        assert src_mask.shape == torch.Size([B, T_observed, 5]), src_mask.shape

        feat, src_mask = self.encoder_input(feat, bbox_feat, src_mask)
        # feat = self.downproject(feat)
        # feat = feat.transpose(1, 2)
        # assert feat.shape == torch.Size([B, T_observed, 1, self.token_dim]), f"feat = {feat.shape}, embed_dim = {self.token_dim}"

        x = self.oct_encoder(feat, valid_mask=src_mask)
        # assert x.shape == torch.Size([B, T_observed, 1, self.token_dim]), x.shape

        last_hidden_state = x[:, -1, :, :]
        assert last_hidden_state.shape == torch.Size([B, self.num_patches, self.token_dim]), last_hidden_state.shape
        memory_mask = get_pad_mask(src_mask[:, -1, :], pad_idx=0)
        assert memory_mask.shape == torch.Size([B, 1, self.num_patches]), memory_mask.shape

        last_frame_global_token = last_hidden_state[:, 0, :]
        pred_last_obs_rhand_embedding = self.last_obs_rhand_embedding_predictor(last_frame_global_token)
        pred_last_obs_lhand_embedding = self.last_obs_lhand_embedding_predictor(last_frame_global_token)
        assert pred_last_obs_rhand_embedding.shape == pred_last_obs_lhand_embedding.shape == torch.Size([B, self.token_dim]), pred_last_obs_rhand_embedding.shape

        pred_rhand_first = self.hand_head.inference(pred_last_obs_rhand_embedding, contact_point=None)
        pred_lhand_first = self.hand_head.inference(pred_last_obs_lhand_embedding, contact_point=None)

        pred_rhand_by_now = pred_rhand_first.unsqueeze(1)
        pred_lhand_by_now = pred_lhand_first.unsqueeze(1)
        assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([B, 1, 2]), pred_rhand_by_now.shape
        for i in range(pred_len):
            assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([B, i + 1, 2]), pred_rhand_by_now.shape
            trg_mask = torch.ones_like(pred_rhand_by_now[:, :, 0])
            assert trg_mask.shape == torch.Size([B, i + 1]), trg_mask.shape
            trg_mask = get_subsequent_mask(trg_mask)
            assert trg_mask.shape == torch.Size([1, i + 1, i + 1]), trg_mask.shape
            pred_rhand_embedding = self.oct_decoder(pred_rhand_by_now, last_hidden_state, last_hidden_state_mask=memory_mask, hand_embedding_mask=trg_mask)
            pred_lhand_embedding = self.oct_decoder(pred_lhand_by_now, last_hidden_state, last_hidden_state_mask=memory_mask, hand_embedding_mask=trg_mask)
            assert pred_rhand_embedding.shape == pred_lhand_embedding.shape == torch.Size([B, i + 1, self.token_dim]), pred_rhand_embedding.shape
            pred_rhand_embedding = pred_rhand_embedding.reshape(B * (i + 1), self.token_dim)
            pred_lhand_embedding = pred_lhand_embedding.reshape(B * (i + 1), self.token_dim)
            assert pred_rhand_embedding.shape == pred_lhand_embedding.shape == torch.Size([B * (i + 1), self.token_dim]), pred_rhand_embedding.shape
            pred_rhand = self.hand_head.inference(pred_rhand_embedding, contact_point=None)
            pred_lhand = self.hand_head.inference(pred_lhand_embedding, contact_point=None)
            pred_rhand = pred_rhand.reshape(B, (i + 1), 2)
            pred_lhand = pred_lhand.reshape(B, (i + 1), 2)
            assert pred_rhand.shape == pred_lhand.shape == torch.Size([B, i + 1, 2]), pred_rhand.shape
            cur_pred_rhand = pred_rhand[:, -1:, :]
            cur_pred_lhand = pred_lhand[:, -1:, :]
            pred_rhand_by_now = torch.cat((pred_rhand_by_now, cur_pred_rhand), dim=1)
            pred_lhand_by_now = torch.cat((pred_lhand_by_now, cur_pred_lhand), dim=1)
            assert pred_rhand_by_now.shape == pred_lhand_by_now.shape == torch.Size([B, i + 2, 2]), pred_rhand_by_now.shape

        pred_hand = torch.stack((pred_rhand_by_now[:, 1:, :], pred_lhand_by_now[:, 1:, :]), dim=1)
        r_pred_contact = self.object_head.inference(last_hidden_state[:, 0, :], pred_rhand_by_now)
        l_pred_contact = self.object_head.inference(last_hidden_state[:, 0, :], pred_lhand_by_now)
        pred_contact = torch.stack([r_pred_contact, l_pred_contact], dim=1)

        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape

        if future_valid is not None and torch.all(future_valid.sum(dim=1) >= 1):
            r_pred_contact_dist = traj_affordance_dist(pred_hand.reshape(-1, 2), r_pred_contact, future_valid)
            l_pred_contact_dist = traj_affordance_dist(pred_hand.reshape(-1, 2), l_pred_contact, future_valid)
            pred_contact_dist = torch.stack((r_pred_contact_dist, l_pred_contact_dist), dim=1)
            _, selected_idx = pred_contact_dist.min(dim=1)
            selected_idx = selected_idx.unsqueeze(dim=1).unsqueeze(dim=2).expand(pred_contact.shape[0], 1,
                                                                                 pred_contact.shape[2])
            pred_contact = torch.gather(pred_contact, dim=1, index=selected_idx).squeeze(dim=1)

        return pred_hand, pred_contact


class ObjectTransformer_global(ObjectTransformer):
    def __init__(self, src_in_features, trg_in_features, num_patches,
                 hand_head, obj_head,
                 token_dim=512, coord_dim=64, num_heads=8, enc_depth=6, dec_depth=4,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 encoder_time_embed_type='sin', decoder_time_embed_type='sin',
                 num_frames_input=None, num_frames_output=None):

        super().__init__(src_in_features, trg_in_features, num_patches,
                         hand_head, obj_head,
                         token_dim, coord_dim, num_heads, enc_depth, dec_depth,
                         mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                         drop_path_rate, norm_layer, dropout,
                         encoder_time_embed_type, decoder_time_embed_type,
                         num_frames_input, num_frames_output)

    def encoder_input(self,
                      feat,
                      bbox_feat,
                      src_mask):
        B, T = feat.shape[0], feat.shape[2]
        assert feat.shape == torch.Size([B, 5, T, self.src_in_features]), feat.shape
        assert src_mask.shape == torch.Size([B, T, 5]), src_mask.shape
        feat = self.downproject(feat)
        assert feat.shape == torch.Size([B, 5, T, self.token_dim]), feat.shape

        global_feat = feat[:, 0:1, :, :].transpose(1, 2)
        assert global_feat.shape == torch.Size([B, T, 1, self.token_dim]), global_feat.shape
        src_mask = src_mask[:, :, :1]
        assert src_mask.shape == torch.Size([B, T, 1]), src_mask.shape
        return global_feat, src_mask

