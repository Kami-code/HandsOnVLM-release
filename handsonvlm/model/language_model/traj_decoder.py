import torch
import torch.nn as nn

from hoi_forecast.architecture.traj_decoder import TrajCVAE, TrajMLP


class TrajDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hand_traj_decoder = None

    def forward(self, **kwargs):
        pred_hand_embeddings = kwargs['pred_hand_embeddings']
        future_hands = kwargs['future_hands']
        future_valid = kwargs['future_valid']

        B = pred_hand_embeddings.shape[0]
        assert pred_hand_embeddings.shape == torch.Size([B, 2, 4, self.token_dim]), pred_hand_embeddings.shape
        assert future_hands.shape == torch.Size([B, 2, 4, 2]), future_hands.shape
        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape
        pred_hand_embeddings = pred_hand_embeddings.reshape(-1, self.token_dim)
        future_hands = future_hands.reshape(B * 2 * 4, 2)
        pred_hand, traj_loss, traj_kl_loss = self.hand_traj_decoder(pred_hand_embeddings, future_hands, future_valid, contact_point=None)
        assert pred_hand.shape == torch.Size([B * 2 * 4, 2]), pred_hand.shape
        pred_hand = pred_hand.reshape(B, 2, 4, 2)

        loss = dict()
        lambda_traj = kwargs['lambda_traj']
        lambda_traj_kl = kwargs['lambda_traj_kl']

        traj_loss = lambda_traj * traj_loss.sum()
        traj_kl_loss = lambda_traj_kl * traj_kl_loss.sum()

        loss['traj_loss'] = traj_loss
        loss['traj_kl_loss'] = traj_kl_loss
        loss['total_loss'] = loss['traj_loss'] + loss['traj_kl_loss']
        return loss

    def inference(self, **kwargs):
        pred_hand_embeddings = kwargs['pred_hand_embeddings']
        B = pred_hand_embeddings.shape[0]
        T_pred = pred_hand_embeddings.shape[2]
        assert pred_hand_embeddings.shape == torch.Size([B, 2, T_pred, self.token_dim]), pred_hand_embeddings.shape
        pred_hand_embeddings = pred_hand_embeddings.reshape(-1, self.token_dim)
        pred_hands = self.hand_traj_decoder.inference(pred_hand_embeddings)
        pred_hands = pred_hands.reshape(B, 2, T_pred, 2)
        return pred_hands


class MLPTrajDecoder(TrajDecoder):
    def __init__(self, token_dim):
        super(MLPTrajDecoder, self).__init__()
        self.token_dim = token_dim
        hidden_dim = 512
        latent_dim = 256
        self.hand_traj_decoder = TrajMLP(in_dim=self.token_dim, hidden_dim=hidden_dim,
                                         latent_dim=latent_dim, token_dim=self.token_dim)


class CVAETrajDecoder(TrajDecoder):
    def __init__(self, token_dim):
        super(CVAETrajDecoder, self).__init__()
        self.token_dim = token_dim
        hidden_dim = 512
        latent_dim = 256
        coord_dim = 64
        self.hand_traj_decoder = TrajCVAE(in_dim=2, hidden_dim=hidden_dim,
                                          latent_dim=latent_dim, token_dim=self.token_dim,
                                          coord_dim=coord_dim, condition_contact=False)
