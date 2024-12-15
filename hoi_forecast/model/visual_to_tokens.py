import torch
import torch.nn as nn
import einops
import numpy as np


class VisualToTokenHelper:
    def __init__(self, images_raw_encode, images_mm_projector, fuse_input_mode, video_compress_mode,
                 mm_hidden_size, token_dim):
        self.images_raw_encode = images_raw_encode
        self.images_mm_projector = images_mm_projector
        self.fuse_input_mode = fuse_input_mode
        self.video_compress_mode = video_compress_mode
        self.mm_hidden_size = mm_hidden_size
        self.token_dim = token_dim
        self.b = None
        self.t = None
        self.c = None
        self.h = None
        self.w = None
        self.input_feat_dim = None

    def pipeline(self, **kwargs):
        if kwargs.get('feat') is not None:
            self.b, _, self.t, _ = kwargs['feat'].shape
        if kwargs.get('images') is not None:
            self.b, self.t, self.c, self.h, self.w = kwargs['images'].shape
        input_tokens, attention_mask = self.fuse_input(**kwargs)
        if attention_mask is not None:
            pass
        else:
            attention_mask = torch.ones(input_tokens.shape[:-1], dtype=torch.bool, device=input_tokens.device)
        output_tokens, attention_mask = self.compress_tokens(tokens=input_tokens, attention_mask=attention_mask)
        visual_token_num = output_tokens.shape[1]
        assert output_tokens.shape == torch.Size([self.b, visual_token_num, self.token_dim]), f"output_tokens.shape = {output_tokens.shape}, expected shape is {torch.Size([self.b, visual_token_num, self.token_dim])}"
        assert attention_mask.shape == torch.Size([self.b, visual_token_num]), f"attention_mask.shape = {attention_mask.shape}, expected shape is {torch.Size([self.b, visual_token_num])}"
        return output_tokens, attention_mask

    def fuse_input(self, **kwargs):
        images = kwargs['images']

        def hoi_global():
            # this is borrowed from: https://github.com/stevenlsw/hoi-forecast/blob/fbc85a17cda21c29974994abcdb055e943f98dcc/networks/transformer.py#L158
            feat = kwargs['feat']
            bbox_feat = kwargs['bbox_feat']
            valid_mask = kwargs['valid_mask']
            assert feat.shape == torch.Size([self.b, 5, self.t, self.mm_hidden_size]), feat.shape
            assert bbox_feat.shape == torch.Size([self.b, 4, self.t, 4]), bbox_feat.shape
            assert valid_mask.shape == torch.Size([self.b, 5, self.t]), valid_mask.shape
            downproject = kwargs['extra_kwargs']['downproject']
            feat = downproject(feat)

            tokens = global_feat = feat[:, 0:1, :, :].transpose(1, 2)
            global_valid = valid_mask[:, 0:1, :]
            assert tokens.shape == torch.Size([self.b, self.t, 1, self.mm_hidden_size]), global_feat.shape
            assert global_valid.shape == torch.Size([self.b, 1, self.t]), global_valid.shape

            # todo: here is important, we apply mm projector to pre-extracted tokens
            tokens = self.images_mm_projector(tokens)
            assert tokens.shape == torch.Size([self.b, self.t, 1, self.token_dim]), f"tokens.shape = {tokens.shape}, we want it to be {torch.Size([self.b, self.t, 3, self.token_dim])}"
            attention_mask = global_valid.transpose(1, 2).to(torch.bool)
            assert attention_mask.shape == torch.Size([self.b, self.t, 1]), attention_mask.shape
            return tokens, attention_mask

        def clip():
            feat = kwargs['feat']
            bbox_feat = kwargs['bbox_feat']
            valid_mask = kwargs['valid_mask']
            assert feat.shape == torch.Size([self.b, self.t, 256, self.mm_hidden_size]), feat.shape
            assert bbox_feat.shape == torch.Size([self.b, 4, self.t, 4]), bbox_feat.shape
            assert valid_mask.shape == torch.Size([self.b, 5, self.t]), valid_mask.shape
            downproject = kwargs['extra_kwargs']['downproject']
            feat = downproject(feat)

            tokens = global_feat = feat[:, 0:1, :, :].transpose(1, 2)
            global_valid = valid_mask[:, 0:1, :]
            assert tokens.shape == torch.Size([self.b, self.t, 1, self.mm_hidden_size]), global_feat.shape
            assert global_valid.shape == torch.Size([self.b, 1, self.t]), global_valid.shape

            # todo: here is important, we apply mm projector to pre-extracted tokens
            tokens = self.images_mm_projector(tokens)
            assert tokens.shape == torch.Size([self.b, self.t, 1, self.token_dim]), f"tokens.shape = {tokens.shape}, we want it to be {torch.Size([self.b, self.t, 3, self.token_dim])}"
            attention_mask = global_valid.transpose(1, 2).to(torch.bool)
            assert attention_mask.shape == torch.Size([self.b, self.t, 1]), attention_mask.shape
            return tokens, attention_mask

        def hoi_hand():
            # this is borrowed from: https://github.com/stevenlsw/hoi-forecast/blob/fbc85a17cda21c29974994abcdb055e943f98dcc/networks/transformer.py#L158
            feat = kwargs['feat']
            bbox_feat = kwargs['bbox_feat']
            valid_mask = kwargs['valid_mask']
            assert feat.shape == torch.Size([self.b, 5, self.t, self.mm_hidden_size]), feat.shape
            assert bbox_feat.shape == torch.Size([self.b, 4, self.t, 4]), bbox_feat.shape
            assert valid_mask.shape == torch.Size([self.b, 5, self.t]), valid_mask.shape
            downproject = kwargs['extra_kwargs']['downproject']
            bbox_to_feature = kwargs['extra_kwargs']['bbox_to_feature']
            feat_fusion = kwargs['extra_kwargs']['feat_fusion']
            feat = downproject(feat)
            bbox_feat = bbox_feat[:, :2, :, :]
            bbox_feat = bbox_feat.reshape(-1, 4)
            assert bbox_feat.shape == torch.Size([self.b * 2 * self.t, 4]), bbox_feat.shape
            bbox_feat = bbox_to_feature(bbox_feat)
            assert bbox_feat.shape == torch.Size([self.b * 2 * self.t, 64]), bbox_feat.shape
            bbox_feat = bbox_feat.reshape(self.b, 2, self.t, 64)
            assert bbox_feat.shape == torch.Size([self.b, 2, self.t, 64]), bbox_feat.shape

            global_feat = feat[:, 0:1, :, :]
            hand_feat = feat[:, 1:3, :, :]
            global_hand_valid = valid_mask[:, 0:3, :]
            assert hand_feat.shape == torch.Size([self.b, 2, self.t, self.mm_hidden_size]), hand_feat.shape
            assert global_feat.shape == torch.Size([self.b, 1, self.t, self.mm_hidden_size]), global_feat.shape
            assert global_hand_valid.shape == torch.Size([self.b, 3, self.t]), global_hand_valid.shape

            hand_bbox_feat = torch.cat((hand_feat, bbox_feat), dim=-1)
            assert hand_bbox_feat.shape == torch.Size([self.b, 2, 10, self.mm_hidden_size + 64]), hand_bbox_feat.shape
            hand_bbox_feat = hand_bbox_feat.reshape(-1, self.mm_hidden_size + 64)
            assert hand_bbox_feat.shape == torch.Size([self.b * 2 * self.t, self.mm_hidden_size + 64]), hand_bbox_feat.shape
            hand_bbox_feat = feat_fusion(hand_bbox_feat)
            hand_bbox_feat = hand_bbox_feat.reshape(self.b, -1, self.t, self.mm_hidden_size)
            assert hand_bbox_feat.shape == torch.Size([self.b, 2, self.t, self.mm_hidden_size]), hand_bbox_feat.shape

            tokens = torch.cat((global_feat, hand_feat), dim=1)
            assert tokens.shape == torch.Size([self.b, 3, self.t, self.mm_hidden_size]), tokens.shape
            tokens = tokens.transpose(1, 2)
            assert tokens.shape == torch.Size([self.b, self.t, 3, self.mm_hidden_size]), tokens.shape
            # todo: here is important, we apply mm projector to pre-extracted tokens
            tokens = self.images_mm_projector(tokens)
            assert tokens.shape == torch.Size([self.b, self.t, 3, self.token_dim]), f"tokens.shape = {tokens.shape}, we want it to be {torch.Size([self.b, self.t, 3, self.token_dim])}"
            global_hand_valid = global_hand_valid.transpose(1, 2)
            attention_mask = global_hand_valid.to(torch.bool)
            assert attention_mask.shape == torch.Size([self.b, self.t, 3]), attention_mask.shape
            return tokens, attention_mask

        def hoi():
            # this is borrowed from: https://github.com/stevenlsw/hoi-forecast/blob/fbc85a17cda21c29974994abcdb055e943f98dcc/networks/transformer.py#L158
            feat = kwargs['feat']
            bbox_feat = kwargs['bbox_feat']
            valid_mask = kwargs['valid_mask']
            assert feat.shape == torch.Size([self.b, 5, self.t, 1024]), feat.shape
            assert bbox_feat.shape == torch.Size([self.b, 4, self.t, 4]), bbox_feat.shape
            assert valid_mask.shape == torch.Size([self.b, 5, self.t]), valid_mask.shape
            downproject = kwargs['extra_kwargs']['downproject']
            bbox_to_feature = kwargs['extra_kwargs']['bbox_to_feature']
            feat_fusion = kwargs['extra_kwargs']['feat_fusion']
            feat = downproject(feat)
            bbox_feat = bbox_feat.reshape(-1, 4)
            assert bbox_feat.shape == torch.Size([self.b * 4 * self.t, 4]), bbox_feat.shape
            bbox_feat = bbox_to_feature(bbox_feat)
            assert bbox_feat.shape == torch.Size([self.b * 4 * self.t, 64]), bbox_feat.shape
            bbox_feat = bbox_feat.reshape(self.b, 4, self.t, 64)
            assert bbox_feat.shape == torch.Size([self.b, 4, self.t, 64]), bbox_feat.shape

            global_feat = feat[:, 0:1, :, :]
            hand_obj_feat = feat[:, 1:5, :, :]
            global_hand_valid = valid_mask
            assert hand_obj_feat.shape == torch.Size([self.b, 4, self.t, self.mm_hidden_size]), hand_obj_feat.shape
            assert global_feat.shape == torch.Size([self.b, 1, self.t, self.mm_hidden_size]), global_feat.shape
            assert global_hand_valid.shape == torch.Size([self.b, 5, self.t]), global_hand_valid.shape

            hand_obj_bbox_feat = torch.cat((hand_obj_feat, bbox_feat), dim=-1)
            assert hand_obj_bbox_feat.shape == torch.Size([self.b, 4, 10, self.mm_hidden_size + 64]), hand_obj_bbox_feat.shape
            hand_obj_bbox_feat = hand_obj_bbox_feat.reshape(-1, self.mm_hidden_size + 64)
            assert hand_obj_bbox_feat.shape == torch.Size([self.b * 4 * self.t, self.mm_hidden_size + 64]), hand_obj_bbox_feat.shape
            hand_obj_bbox_feat = feat_fusion(hand_obj_bbox_feat)
            hand_obj_bbox_feat = hand_obj_bbox_feat.reshape(self.b, -1, self.t, self.mm_hidden_size)
            assert hand_obj_bbox_feat.shape == torch.Size([self.b, 4, self.t, self.mm_hidden_size]), hand_obj_bbox_feat.shape

            tokens = torch.cat((global_feat, hand_obj_bbox_feat), dim=1)
            assert tokens.shape == torch.Size([self.b, 5, self.t, self.mm_hidden_size]), tokens.shape
            tokens = tokens.transpose(1, 2)
            assert tokens.shape == torch.Size([self.b, self.t, 5, self.mm_hidden_size]), tokens.shape
            tokens = self.images_mm_projector(tokens)
            # todo: here is important, we apply mm projector to pre-extracted tokens
            global_hand_valid = global_hand_valid.transpose(1, 2)
            attention_mask = global_hand_valid.to(torch.bool)
            assert attention_mask.shape == torch.Size([self.b, self.t, 5]), attention_mask.shape
            return tokens, attention_mask

        if self.fuse_input_mode == 'origin':
            tokens = self.encode_images(images)
            assert tokens.shape == torch.Size([self.b, self.t, 256, self.token_dim])
            attention_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
            return tokens, attention_mask
        elif self.fuse_input_mode == 'origin-random':
            tokens = torch.rand(self.b, self.t, 256, self.token_dim, device=images.device, dtype=images.dtype)
            assert tokens.shape == torch.Size([self.b, self.t, 256, self.token_dim])
            attention_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-global':
            tokens, attention_mask = hoi_global()
            return tokens, attention_mask
        elif self.fuse_input_mode == 'clip':
            tokens, attention_mask = clip()
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-global-random':
            tokens, attention_mask = hoi_global()
            tokens = torch.rand_like(tokens.clone())
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-hand':
            tokens, attention_mask = hoi_hand()
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-hand-random':
            tokens, attention_mask = hoi_hand()
            tokens = torch.rand_like(tokens.clone())
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-hand-random-zero-embed-one-attention':
            tokens, attention_mask = hoi_hand()
            tokens = torch.zeros_like(tokens.clone())
            attention_mask = torch.ones_like(attention_mask.clone())
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-hand-reverse':
            tokens, attention_mask = hoi_hand()
            attention_mask = ~attention_mask
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi':
            tokens, attention_mask = hoi()
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-reverse':
            tokens, attention_mask = hoi()
            attention_mask = ~attention_mask
            return tokens, attention_mask
        elif self.fuse_input_mode == 'hoi-random':
            tokens, attention_mask = hoi()
            tokens = torch.rand_like(tokens)
            return tokens, attention_mask
        else:
            raise ValueError(f"Unknown fuse_input_mode: {self.fuse_input_mode}")


    def compress_tokens(self, tokens, attention_mask):
        _, _, token_num_per_frame, _ = tokens.shape
        if self.video_compress_mode == 'none':
            tokens = einops.rearrange(tokens, 'b t s d -> b (t s) d')
            assert tokens.shape == torch.Size([self.b, self.t * token_num_per_frame, self.token_dim]), tokens.shape
            attention_mask = einops.rearrange(attention_mask, 'b t s -> b (t s)')
            assert attention_mask.shape == torch.Size([self.b, self.t * token_num_per_frame]), f"current attention_mask shape is {attention_mask.shape}, expected shape is {torch.Size([self.b, self.t * token_num_per_frame])}"
            return tokens, attention_mask
        elif self.video_compress_mode == 'temporal':
            tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
            assert tokens.shape == torch.Size([self.b, self.t, self.token_dim])
            return tokens
        elif self.video_compress_mode == 'spatial':
            tokens = einops.reduce(tokens, 'b t s d -> b s d', 'mean')
            assert tokens.shape == torch.Size([self.b, token_num_per_frame, self.token_dim])
            return tokens
        elif self.video_compress_mode == 'temporal_spatial':
            t_tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
            s_tokens = einops.reduce(tokens, 'b t s d -> b s d', 'mean')
            tokens = torch.cat([t_tokens, s_tokens], dim=1)
            assert tokens.shape == torch.Size([self.b, self.t + token_num_per_frame, self.token_dim])
            return tokens
        elif self.video_compress_mode == 'temporal_spatial_pool' or self.video_compress_mode == 'spatial_pool':
            pool_size = 2
            selected_frames = np.round(np.linspace(0, tokens.shape[1] - 1, pool_size * pool_size)).astype(int)
            s_tokens = tokens[:, selected_frames, ...]
            assert s_tokens.shape == torch.Size([self.b, pool_size * pool_size, 256, self.token_dim])
            s_tokens = einops.rearrange(s_tokens, 'b t (h w) d -> (b t) d h w', h=16, w=16)
            assert s_tokens.shape == torch.Size([self.b * pool_size * pool_size, self.token_dim, 16, 16])
            s_tokens = nn.functional.avg_pool2d(s_tokens, kernel_size=pool_size)
            assert s_tokens.shape == torch.Size([self.b * pool_size * pool_size, self.token_dim, 8, 8])
            s_tokens = einops.rearrange(s_tokens, '(b t) d h w -> b (t h w) d', b=self.b)
            assert s_tokens.shape == torch.Size([self.b, token_num_per_frame, self.token_dim])
            if self.video_compress_mode == 'temporal_spatial_pool':
                t_tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
                assert t_tokens.shape == torch.Size([self.b, self.t, self.token_dim])
                tokens = torch.cat([t_tokens, s_tokens], dim=1)
                assert tokens.shape == torch.Size([self.b, self.t + token_num_per_frame, self.token_dim])
            elif self.video_compress_mode == 'spatial_pool':
                tokens = s_tokens
                assert tokens.shape == torch.Size([self.b, token_num_per_frame, self.token_dim])
            attention_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
        return tokens, attention_mask

    def encode_images(self, images):
        assert images.shape == torch.Size([self.b, self.t, self.c, self.h, self.w]), images.shape
        images = einops.rearrange(images, 'b t c h w -> (b t) c h w')
        assert images.shape == torch.Size([self.b * self.t, self.c, self.h, self.w]), images.shape
        tokens = self.images_raw_encode(images)
        tokens = self.images_mm_projector(tokens)
        num_tokens_pre_frame = tokens.shape[1]
        assert tokens.shape == torch.Size([self.b * self.t, num_tokens_pre_frame, self.token_dim]), tokens.shape
        tokens = einops.rearrange(tokens, '(b t) s d -> b t s d', b=self.b)
        assert tokens.shape == torch.Size([self.b, self.t, num_tokens_pre_frame, self.token_dim]), tokens.shape
        return tokens