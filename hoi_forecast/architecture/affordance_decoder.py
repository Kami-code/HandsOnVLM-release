import torch
import torch.nn as nn

from hoi_forecast.architecture.decoder_modules import VAE


class AffordanceCVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, token_dim, coord_dim=None,
                 pred_len=4, condition_traj=True, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_traj = condition_traj
        self.z_scale = z_scale
        self.T_pred = pred_len
        self.token_dim = token_dim
        if coord_dim is None:
            coord_dim = hidden_dim // 2
        self.coord_dim = coord_dim

        if self.condition_traj:

            self.traj_to_feature = nn.Sequential(
                nn.Linear(2 * (self.T_pred + 1), self.coord_dim * (self.T_pred + 1), bias=False),
                nn.ELU(inplace=True))
            self.traj_context_fusion = nn.Sequential(
                nn.Linear(self.token_dim + self.coord_dim * (self.T_pred + 1), self.token_dim, bias=False),
                nn.ELU(inplace=True))
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=self.token_dim)

    def forward(self, last_frame_global_token, contact_point, hand_traj=None):
        B = last_frame_global_token.shape[0]
        assert last_frame_global_token.shape == torch.Size([B, self.token_dim]), last_frame_global_token.shape
        assert contact_point.shape == torch.Size([B, 2]), contact_point.shape

        if self.condition_traj:
            assert hand_traj is not None
            T = hand_traj.shape[1]
            assert hand_traj.shape == torch.Size([B, T, 2])
            hand_traj = hand_traj.reshape(B, -1)
            traj_feat = self.traj_to_feature(hand_traj)
            fusion_feat = torch.cat([last_frame_global_token, traj_feat], dim=1)
            condition_context = self.traj_context_fusion(fusion_feat)
        else:
            condition_context = last_frame_global_token
        assert condition_context.shape == torch.Size([B, self.token_dim]), condition_context.shape

        pred_contact, recon_loss, KLD = self.cvae(contact_point, condition=condition_context)
        assert pred_contact.shape == torch.Size([B, 2]), pred_contact.shape
        assert recon_loss.shape == KLD.shape == torch.Size([B, ]), (recon_loss.shape, KLD.shape)
        return pred_contact, recon_loss, KLD

    def inference(self, last_frame_global_token, hand_traj=None):
        B = last_frame_global_token.shape[0]
        assert last_frame_global_token.shape == torch.Size([B, self.token_dim]), last_frame_global_token.shape

        if self.condition_traj:
            assert hand_traj is not None
            batch_size = last_frame_global_token.shape[0]
            hand_traj = hand_traj.reshape(batch_size, -1)
            traj_feat = self.traj_to_feature(hand_traj)
            fusion_feat = torch.cat([last_frame_global_token, traj_feat], dim=1)
            condition_context = self.traj_context_fusion(fusion_feat)
        else:
            condition_context = last_frame_global_token

        z = self.z_scale * torch.randn([condition_context.shape[0], self.latent_dim], device=condition_context.device).to(condition_context.dtype)
        pred_contact = self.cvae.inference(z, c=condition_context)
        return pred_contact