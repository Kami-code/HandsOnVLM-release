import torch
import torch.nn as nn
import einops

from hoi_forecast.architecture.affordance_decoder import VAE


class TrajCVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, token_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale

        self.token_dim = token_dim
        if coord_dim is None:
            coord_dim = hidden_dim // 2
        self.coord_dim = coord_dim

        if self.condition_contact:
            self.contact_to_feature = nn.Sequential(
                nn.Linear(2, self.coord_dim, bias=False),
                nn.ELU(inplace=True))
            self.contact_context_fusion = nn.Sequential(
                nn.Linear(self.token_dim + self.coord_dim, self.token_dim, bias=False),
                nn.ELU(inplace=True))
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=self.token_dim)

    def forward(self,
                hand_embedding,
                shifted_gt_hand,
                future_valid,
                contact_point=None):
        B = future_valid.shape[0]
        T_pred = int(hand_embedding.shape[0] / B / 2)

        assert hand_embedding.shape == torch.Size([B * 2 * T_pred, self.token_dim]), hand_embedding.shape
        assert shifted_gt_hand.shape == torch.Size([B * 2 * T_pred, 2]), shifted_gt_hand.shape
        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape

        if self.condition_contact:
            assert contact_point.shape == torch.Size([B, 2]), contact_point.shape
            contact_feat = self.contact_to_feature(contact_point)
            assert contact_point.shape == torch.Size([B, self.coord_dim]), contact_point.shape
            contact_feat = einops.repeat(contact_feat, 'm n -> m p q n', p=2, q=T_pred).reshape(-1, self.coord_dim)
            assert contact_point.shape == torch.Size([B * 2 * T_pred, self.coord_dim]), contact_feat.shape
            fusion_feat = torch.cat([hand_embedding, contact_feat], dim=1)
            assert fusion_feat.shape == torch.Size([B * 2 * T_pred, self.token_dim + self.coord_dim]), fusion_feat.shape
            condition_context = self.contact_context_fusion(fusion_feat)
        else:
            condition_context = hand_embedding
        assert condition_context.shape == torch.Size([B * 2 * T_pred, self.token_dim]), condition_context.shape

        condition_context = condition_context.to(torch.bfloat16)

        pred_hand, recon_loss, KLD = self.cvae(shifted_gt_hand, condition=condition_context)
        assert pred_hand.shape == torch.Size([B * 2 * T_pred, 2])
        assert recon_loss.shape == KLD.shape == torch.Size([B * 2 * T_pred, ]), (recon_loss.shape, KLD.shape)

        # we only care about the loss on the valid hand
        recon_loss = recon_loss.reshape(B, 2, T_pred)
        # Check if the sum is greater than zero
        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape
        KLD = KLD.sum(-1)
        KLD = (KLD * future_valid).sum(1)
        recon_loss = recon_loss.sum(-1)
        traj_loss = (recon_loss * future_valid).sum(1)
        return pred_hand, traj_loss, KLD

    def inference(self, hand_embedding, contact_point=None):
        B_pairs, embed_dim = hand_embedding.shape[0], hand_embedding.shape[1]

        if self.condition_contact:
            assert contact_point is not None
            B = contact_point.shape[0]
            T_pred = int(hand_embedding.shape[0] / B) + 1
            contact_feat = self.contact_to_feature(contact_point)
            contact_feat = einops.repeat(contact_feat, 'm n -> m p n', p=T_pred - 1)
            contact_feat = contact_feat.reshape(-1, self.coord_dim)
            fusion_feat = torch.cat([hand_embedding, contact_feat], dim=1)
            condition_context = self.contact_context_fusion(fusion_feat)
        else:
            condition_context = hand_embedding
        assert condition_context.shape == torch.Size([B_pairs, self.token_dim]), condition_context.shape

        z = self.z_scale * torch.randn([B_pairs, self.latent_dim], device=hand_embedding.device).to(hand_embedding.dtype)
        pred_hand = self.cvae.inference(z, c=condition_context)
        assert pred_hand.shape == torch.Size([B_pairs, 2])
        return pred_hand


class TrajMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, token_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.token_dim = token_dim

        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )


    def forward(self, hand_embedding, shifted_gt_hand, gt_hand_valid, contact_point=None):
        B = gt_hand_valid.shape[0]
        T_pred = int(hand_embedding.shape[0] / B / 2)

        assert hand_embedding.shape == torch.Size([B * 2 * T_pred, self.token_dim]), hand_embedding.shape
        assert shifted_gt_hand.shape == torch.Size([B * 2 * T_pred, 2]), shifted_gt_hand.shape
        assert gt_hand_valid.shape == torch.Size([B, 2, T_pred]), gt_hand_valid.shape


        condition_context = hand_embedding
        assert condition_context.shape == torch.Size([B * 2 * T_pred, self.token_dim]), condition_context.shape

        # condition_context = condition_context.to(torch.bfloat16)

        pred_hand = self.mlp(condition_context)
        assert pred_hand.shape == torch.Size([B * 2 * T_pred, 2])

        recon_loss = torch.nn.functional.mse_loss(pred_hand, shifted_gt_hand, reduction='none')
        assert recon_loss.shape == torch.Size([B * 2 * T_pred, 2]), recon_loss.shape

        recon_loss = recon_loss.sum(dim=-1)
        assert recon_loss.shape == torch.Size([B * 2 * T_pred]), recon_loss.shape

        # we only care about the loss on the valid hand
        recon_loss = recon_loss.reshape(B, 2, T_pred)
        # Check if the sum is greater than zero
        assert gt_hand_valid.shape == torch.Size([B, 2, T_pred]), gt_hand_valid.shape

        valid_sum = torch.sum(gt_hand_valid) + 1e-6
        recon_loss = torch.sum(recon_loss * gt_hand_valid) / valid_sum
        recon_loss = recon_loss.repeat(B)
        KLD = torch.zeros(B, device=hand_embedding.device)
        return pred_hand, recon_loss, KLD

    def inference(self, hand_embedding, contact_point=None):
        B_pairs, embed_dim = hand_embedding.shape[0], hand_embedding.shape[1]

        condition_context = hand_embedding
        assert condition_context.shape == torch.Size([B_pairs, self.token_dim]), condition_context.shape

        pred_hand = self.mlp(condition_context)
        assert pred_hand.shape == torch.Size([B_pairs, 2])
        return pred_hand