import torch
import torch.nn as nn

from hoi_forecast.architecture.affordance_decoder import AffordanceCVAE
from hoi_forecast.architecture.object_transformer import ObjectTransformer, ObjectTransformer_global, TrajCVAE


class HoiForecastModel(nn.Module):
    def __init__(self, object_transformer, lambda_obj=None, lambda_traj=None, lambda_obj_kl=None, lambda_traj_kl=None, lambda_last_hand=None):
        super(HoiForecastModel, self).__init__()
        self.object_transformer: ObjectTransformer = object_transformer
        self.lambda_obj = lambda_obj
        self.lambda_obj_kl = lambda_obj_kl
        self.lambda_traj = lambda_traj
        self.lambda_traj_kl = lambda_traj_kl
        self.lambda_last_hand = lambda_last_hand

    def forward(self,
                future_hands=None,
                contact_point=None,
                future_valid=None,
                num_samples=5,
                pred_len=4,
                **kwargs):
        if self.training:
            losses = {}
            total_loss = 0
            traj_loss, traj_kl_loss, obj_loss, obj_kl_loss, last_hand_loss = self.object_transformer(future_hands=future_hands,
                                                                                                     contact_point=contact_point,
                                                                                                     future_valid=future_valid,
                                                                                                     **kwargs)
            if self.lambda_traj is not None and traj_loss is not None:
                traj_loss = self.lambda_traj * traj_loss.sum()
                total_loss += traj_loss
                losses['traj_loss'] = traj_loss.detach().cpu()
            else:
                losses['traj_loss'] = 0.

            if self.lambda_traj_kl is not None and traj_kl_loss is not None:
                traj_kl_loss = self.lambda_traj_kl * traj_kl_loss.sum()
                total_loss += traj_kl_loss
                losses['traj_kl_loss'] = traj_kl_loss.detach().cpu()
            else:
                losses['traj_kl_loss'] = 0.

            if self.lambda_obj is not None and obj_loss is not None:
                obj_loss = self.lambda_obj * obj_loss.sum()
                total_loss += obj_loss
                losses['obj_loss'] = obj_loss.detach().cpu()
            else:
                losses['obj_loss'] = 0.

            if self.lambda_obj_kl is not None and obj_kl_loss is not None:
                obj_kl_loss = self.lambda_obj_kl * obj_kl_loss.sum()
                total_loss += obj_kl_loss
                losses['obj_kl_loss'] = obj_kl_loss.detach().cpu()
            else:
                losses['obj_kl_loss'] = 0.

            if self.lambda_last_hand is not None and last_hand_loss is not None:
                last_hand_loss = self.lambda_last_hand * last_hand_loss.sum()
                total_loss += last_hand_loss
                losses['last_hand_loss'] = last_hand_loss.detach().cpu()
            else:
                losses['last_hand_loss'] = 0.

            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0.
            return total_loss, losses
        else:
            future_hands_list = []
            contact_points_list = []
            for i in range(num_samples):
                future_hands, contact_point = self.object_transformer.module.inference(future_valid=future_valid,
                                                                                       pred_len=pred_len,
                                                                                       **kwargs)
                future_hands_list.append(future_hands)
                contact_points_list.append(contact_point)

            contact_points = torch.stack(contact_points_list, dim=0)
            contact_points = contact_points.transpose(0, 1)

            future_hands_list = torch.stack(future_hands_list, dim=0)
            future_hands_list = future_hands_list.transpose(0, 1)
            return future_hands_list, contact_points


token_dim = embed_dim = 512
src_in_features = 1024
trg_in_features = 2
num_patches = 5
coord_dim = 64
hidden_dim = 512
latent_dim = 256
num_heads = 8
enc_depth = 6
dec_depth = 4
encoder_time_embed_type = "sin"
decoder_time_embed_type = "sin"


def get_hoi_forecast_origin_model(lambda_obj,
                                  lambda_traj,
                                  lambda_obj_kl,
                                  lambda_traj_kl,
                                  lambda_last_hand,
                                  num_frames_input=10,
                                  num_frames_output=4,
                                  ):
    hand_head = TrajCVAE(in_dim=2, hidden_dim=hidden_dim,
                         latent_dim=latent_dim, token_dim=embed_dim,
                         coord_dim=coord_dim)
    obj_head = AffordanceCVAE(in_dim=2, hidden_dim=hidden_dim,
                              latent_dim=latent_dim, token_dim=token_dim)
    object_transformer: ObjectTransformer = ObjectTransformer(src_in_features=src_in_features,
                                                              trg_in_features=trg_in_features,
                                                              num_patches=num_patches,
                                                              hand_head=hand_head, obj_head=obj_head,
                                                              encoder_time_embed_type=encoder_time_embed_type,
                                                              decoder_time_embed_type=decoder_time_embed_type,
                                                              num_frames_input=num_frames_input,
                                                              num_frames_output=num_frames_output,
                                                              token_dim=token_dim, coord_dim=coord_dim,
                                                              num_heads=num_heads, enc_depth=enc_depth, dec_depth=dec_depth)
    object_transformer = torch.nn.DataParallel(object_transformer)

    model = HoiForecastModel(object_transformer,
                             lambda_obj=lambda_obj,
                             lambda_traj=lambda_traj,
                             lambda_obj_kl=lambda_obj_kl,
                             lambda_traj_kl=lambda_traj_kl,
                             lambda_last_hand=lambda_last_hand)
    return model


def get_hoi_forecast_global_model(lambda_obj,
                                  lambda_traj,
                                  lambda_obj_kl,
                                  lambda_traj_kl,
                                  lambda_last_hand,
                                  num_frames_input=10,
                                  num_frames_output=4):
    hand_head = TrajCVAE(in_dim=2, hidden_dim=hidden_dim,
                         latent_dim=latent_dim, token_dim=embed_dim,
                         coord_dim=coord_dim)
    obj_head = AffordanceCVAE(in_dim=2, hidden_dim=hidden_dim,
                              latent_dim=latent_dim, token_dim=token_dim)
    object_transformer: ObjectTransformer = ObjectTransformer_global(src_in_features=src_in_features,
                                                                     trg_in_features=trg_in_features,
                                                                     num_patches=1,
                                                                     hand_head=hand_head, obj_head=obj_head,
                                                                     encoder_time_embed_type=encoder_time_embed_type,
                                                                     decoder_time_embed_type=decoder_time_embed_type,
                                                                     num_frames_input=num_frames_input,
                                                                     num_frames_output=num_frames_output,
                                                                     token_dim=token_dim, coord_dim=coord_dim,
                                                                     num_heads=num_heads, enc_depth=enc_depth, dec_depth=dec_depth)
    object_transformer = torch.nn.DataParallel(object_transformer)

    model = HoiForecastModel(object_transformer, lambda_obj=lambda_obj, lambda_traj=lambda_traj,
                             lambda_obj_kl=lambda_obj_kl, lambda_traj_kl=lambda_traj_kl, lambda_last_hand=lambda_last_hand)
    return model