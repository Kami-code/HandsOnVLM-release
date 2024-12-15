import time

import numpy as np
import torch
import wandb

from hoi_forecast.model.epoch_utils import progress_bar as bar, AverageMeters
from hoi_forecast.evaluation.traj_eval import evaluate_traj_stochastic


def epoch_pass(loader, model, epoch, phase, optimizer=None, scheduler=None):
    time_meters = AverageMeters()
    print(f"{phase} epoch: {epoch + 1}")
    loss_meters = AverageMeters()
    model.train()
    device = torch.device('cuda')
    end = time.time()
    for batch_idx, sample in enumerate(loader):
        feat = sample['feat'].float().to(device)
        bbox_feat = sample['bbox_feat'].float().to(device)
        valid_mask = sample['valid_mask'].float().to(device)
        future_hands = sample['future_hands'].float().to(device)
        contact_point = sample['contact_point'].float().to(device)
        future_valid = sample['future_valid'].float().to(device)
        image = sample['image'].float().to(device)
        time_meters.add_loss_value("data_time", time.time() - end)
        B = feat.shape[0]
        T_observed, T_pred = 10, 5
        assert feat.shape == torch.Size([B, 5, T_observed, 1024]), feat.shape
        assert bbox_feat.shape == torch.Size([B, 4, T_observed, 4]), bbox_feat.shape
        assert valid_mask.shape == torch.Size([B, 5, T_observed]), valid_mask.shape
        assert future_valid.shape == torch.Size([B, 2]), future_valid.shape
        assert future_hands.shape == torch.Size([B, 2, T_pred, 2]), future_hands.shape
        assert contact_point.shape == torch.Size([B, 2]), contact_point.shape
        assert image.shape == torch.Size([B, T_observed, 3, 224, 224]), image.shape

        model_loss, model_losses = model(feat=feat, bbox_feat=bbox_feat, image=image,
                                         valid_mask=valid_mask, future_hands=future_hands,
                                         contact_point=contact_point, future_valid=future_valid)
        """
        model_losses is a dict with keys:
        - traj_loss
        - traj_kl_loss
        - obj_loss
        - obj_kl_loss
        - total_loss
        """

        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

        for key, val in model_losses.items():
            if val is not None:
                loss_meters.add_loss_value(key, val)

        time_meters.add_loss_value("batch_time", time.time() - end)

        suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                 "| Hand Traj Loss: {traj_loss:.3f} " \
                 "| Hand Traj KL Loss: {traj_kl_loss:.3f} " \
                 "| Object Affordance Loss: {obj_loss:.3f} " \
                 "| Object Affordance KL Loss: {obj_kl_loss:.3f} " \
                 "| Last Hand Embedding Loss: {last_hand_loss:.3f} " \
                 "| Total Loss: {total_loss:.3f} ".format(batch=batch_idx + 1, size=len(loader),
                                                          data=time_meters.average_meters["data_time"].val,
                                                          bt=time_meters.average_meters["batch_time"].avg,
                                                          traj_loss=loss_meters.average_meters["traj_loss"].avg,
                                                          traj_kl_loss=loss_meters.average_meters[
                                                              "traj_kl_loss"].avg,
                                                          obj_loss=loss_meters.average_meters["obj_loss"].avg,
                                                          obj_kl_loss=loss_meters.average_meters["obj_kl_loss"].avg,
                                                          last_hand_loss=loss_meters.average_meters["last_hand_loss"].avg,
                                                          total_loss=loss_meters.average_meters[
                                                              "total_loss"].avg)
        if wandb.run is not None:  # only log to the main process
            wandb.log({"train/traj_loss": loss_meters.average_meters["traj_loss"].avg,
                       "train/traj_kl_loss": loss_meters.average_meters["traj_kl_loss"].avg,
                       "train/obj_loss": loss_meters.average_meters["obj_loss"].avg,
                       "train/obj_kl_loss": loss_meters.average_meters["obj_kl_loss"].avg,
                       "train/last_hand_loss": loss_meters.average_meters["last_hand_loss"].avg,
                       "train/total_loss": loss_meters.average_meters["total_loss"].avg},
                      step=epoch)

        bar(suffix)
        end = time.time()
        if scheduler is not None:
            scheduler.step()

    return loss_meters




def epoch_evaluate(loader, model, epoch, phase, num_samples=5, pred_len=4, num_points=5, gaussian_sigma=3., gaussian_k_ratio=3.,
                   visualize=False):
    time_meters = AverageMeters()
    print(f"evaluate epoch {epoch}")
    preds_traj, gts_traj, valids_traj = [], [], []
    gts_affordance_dict, preds_affordance_dict = {}, {}
    model.eval()

    device = torch.device('cuda')
    end = time.time()

    visualize_cnt = 0
    for batch_idx, sample in enumerate(loader):
        feat = sample['feat'].float().to(device)
        b = feat.shape[0]
        assert feat.shape == torch.Size([b, 5, 10, 1024]), feat.shape

        bbox_feat = sample['bbox_feat'].float().to(device)
        assert bbox_feat.shape == torch.Size([b, 4, 10, 4]), bbox_feat.shape
        valid_mask = sample['valid_mask'].float().to(device)
        future_valid = sample['future_valid'].float().to(device)
        image = sample['image'].float().to(device)
        B = feat.shape[0]
        time_meters.add_loss_value("data_time", time.time() - end)
        with torch.no_grad():
            pred_future_hands, contact_points = model(feat=feat, bbox_feat=bbox_feat,
                                                      valid_mask=valid_mask,
                                                      image=image,
                                                      num_samples=num_samples,
                                                      future_valid=future_valid,
                                                      pred_len=pred_len)
            assert pred_future_hands.shape == (B, num_samples, 2, 4, 2)
            # assert contact_points.shape == (B, num_samples, 2, 2)

        # uids = sample['uid'].numpy()
        future_hands = sample['future_hands'][:, :, 1:, :].float().numpy()
        future_valid = sample['future_valid'].float().numpy()

        gts_traj.append(future_hands)
        valids_traj.append(future_valid)

        pred_future_hands = pred_future_hands.cpu().numpy()
        preds_traj.append(pred_future_hands)

        # if 'eval' in loader.dataset.split:
        #     contact_points = contact_points.cpu().numpy()
        #     for idx, uid in enumerate(uids):
        #         gts_affordance_dict[uid] = loader.dataset.eval_labels[uid]['norm_contacts']
        #         preds_affordance_dict[uid] = contact_points[idx]

        time_meters.add_loss_value("batch_time", time.time() - end)

        suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s" \
            .format(batch=batch_idx + 1, size=len(loader),
                    data=time_meters.average_meters["data_time"].val,
                    bt=time_meters.average_meters["batch_time"].avg)

        bar(suffix)
        end = time.time()

    val_info = {}
    if phase == "traj":
        gts_traj = np.concatenate(gts_traj)
        preds_traj = np.concatenate(preds_traj)
        valids_traj = np.concatenate(valids_traj)
        ade, fde, wde = evaluate_traj_stochastic(preds_traj, gts_traj, valids_traj)
        val_info.update({"traj_ade": ade, "traj_fde": fde, "traj_wde": wde})

    # if 'eval' in loader.dataset.split and phase == "affordance":
    #     affordance_metrics = evaluate_affordance(preds_affordance_dict,
    #                                              gts_affordance_dict,
    #                                              n_pts=num_points,
    #                                              gaussian_sigma=gaussian_sigma,
    #                                              gaussian_k_ratio=gaussian_k_ratio)
    #     val_info.update(affordance_metrics)

    return val_info