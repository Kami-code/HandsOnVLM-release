import torch
import numpy as np


# https://github.com/huang-xx/STGAT/blob/master/STGAT/utils.py#L129
def compute_ade(pred_traj, gt_traj, valid_traj=None, reduction=True):
    # pred_traj (B, num_obj, seq_len, 2)
    # gt_traj (B, num_obj, seq_len, 2)
    # valid: (B, num_obj)
    # return (B), (N), N is the number of valid traj in the batch, if reduction=True
    # return (B, num_obj), (B, num_obj) of ade and valid traj, if reduction=False

    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 1] >= 0) \
                & (gt_traj[:, :, :, 0] < 1) & (gt_traj[:, :, :, 1] < 1)  # (B, num_obj, seq_len)

    error = gt_traj - pred_traj  # (B, num_obj, seq_len, 2)
    error = error * valid_loc[:, :, :, None]  # (B, num_obj, seq_len, 2)

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])  # (B, num_obj)
        error = error ** 2
        ade = torch.sqrt(error.sum(dim=3)).mean(dim=2) * valid_traj  # (B, num_obj)
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=3)
        ade = error.mean(axis=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return ade, valid_traj


def compute_fde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    # pred_traj (B, num_obj, seq_len, 2)
    # gt_traj (B, num_obj, seq_len, 2)
    # valid: (B, num_obj)
    # return (B), (N), N is the number of valid traj in the batch, if reduction=True
    # return (B, num_obj), (B, num_obj) of ade and valid traj, if reduction=False
    pred_last = pred_traj[:, :, -1, :]  # (B, num_obj, 2)
    gt_last = gt_traj[:, :, -1, :]

    valid_loc = (gt_last[:, :, 0] >= 0) & (gt_last[:, :, 1] >= 0) \
                & (gt_last[:, :, 0] < 1) & (gt_last[:, :, 1] < 1)  # (B, num_obj)

    error = gt_last - pred_last  # (B, num_obj, 2)
    error = error * valid_loc[:, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])  # (B, num_obj)
        error = error ** 2
        fde = torch.sqrt(error.sum(dim=2)) * valid_traj  # (B, num_obj)
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=2)
        fde = error * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return fde, valid_traj


def evaluate_traj(preds, gts, valids, val_log=None):
    # preds: (len(test dataset), num_obj, seq_len, 2), normalized points
    # gts: (len(test dataset), num_obj, seq_len, 2), normalized points
    # valids: (len(test dataset), 2)

    len_dataset, num_obj = preds.shape[0], preds.shape[1]

    ade, num_valids = compute_fde(preds, gts, valids)
    fde, num_valids = compute_ade(preds, gts, valids)

    ade_info = 'ADE: %.3f (%d/%d)' % (ade, valids.sum(), len_dataset * num_obj)
    fde_info = "FDE: %.3f (%d/%d)" % (fde, valids.sum(), len_dataset * num_obj)

    if val_log is not None:
        with open(val_log, 'a') as f:
            f.write(ade_info + "\n")
            f.write(fde_info + "\n")
    print(ade_info)
    print(fde_info)
    return ade, fde



def compute_wde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    """
    Compute weighted distance error for trajectories

    Args:
        pred_traj: Predicted trajectories (B, N, T, 2) where B is batch size, N is number of hands (2),
                  T is number of timesteps (4), and 2 is for x,y coordinates
        gt_traj: Ground truth trajectories with same shape as pred_traj
        valid_traj: Validity mask for trajectories (B, N), defaults to ones
        reduction: Whether to reduce results to mean

    Returns:
        wde: Weighted distance error
        valid_traj: Number of valid trajectories
    """
    # Check valid locations (coordinates should be within [0,1])
    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 0] < 1) & \
                (gt_traj[:, :, :, 1] >= 0) & (gt_traj[:, :, :, 1] < 1)

    # Create weights for temporal steps
    weights = np.arange(1, 5) / 4  # [0.25, 0.5, 0.75, 1.0]

    if torch.is_tensor(pred_traj):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        weights = torch.from_numpy(weights).to(pred_traj.device)

        # Calculate error and apply spatial validity mask
        error = gt_traj - pred_traj
        error = error * valid_loc[..., None]

        # Calculate distances
        distances = torch.sqrt((error ** 2).sum(dim=-1))  # B,N,T

        # Apply temporal weights and validity mask
        weighted_distances = (distances * weights) * valid_loc

        # Average over time steps for each trajectory
        traj_errors = weighted_distances.sum(dim=-1) / valid_loc.sum(dim=-1).clamp(min=1)  # B,N

        # Apply trajectory validity mask
        wde = traj_errors * valid_traj

        if reduction:
            wde = wde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    else:  # numpy arrays
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]))

        # Calculate error and apply spatial validity mask
        error = gt_traj - pred_traj
        error = error * valid_loc[..., None]

        # Calculate distances
        distances = np.sqrt(np.sum(error ** 2, axis=-1))  # B,N,T

        # Apply temporal weights and validity mask
        weighted_distances = (distances * weights) * valid_loc

        # Average over time steps for each trajectory
        valid_sum = np.maximum(valid_loc.sum(axis=-1), 1)  # Avoid division by zero
        traj_errors = weighted_distances.sum(axis=-1) / valid_sum  # B,N

        # Apply trajectory validity mask
        wde = traj_errors * valid_traj

        if reduction:
            wde = wde.sum() / (valid_traj.sum() + 1e-6)  # Add small epsilon to avoid division by zero
            valid_traj = valid_traj.sum()

    return wde, valid_traj


def evaluate_traj_stochastic(preds, gts, valids, val_log=None):
    # stochastic model traj evaluation, multiple runs and takes the min
    # follow: https://github.com/HaozhiQi/RPIN/blob/master/rpin/evaluator_pred.py#L53
    # follow: https://github.com/abduallahmohamed/Social-STGCNN/issues/14#issuecomment-604882071
    # follow: https://github.com/agrimgupta92/sgan/blob/master/scripts/evaluate_model.py#L53

    # preds: (len(test dataset), num_samples, num_obj, seq_len, 2), normalized points
    # gts: (len(test dataset), num_obj, seq_len, 2), normalized points
    # valids: (len(test dataset), 2)
    # val_log: validation log file path

    len_dataset, num_samples, num_obj = preds.shape[0], preds.shape[1], preds.shape[2]
    ade_list, fde_list, wde_list = [], [], []
    for idx in range(num_samples):
        # ade, valids: (len(test datast), num_obj)
        ade, _ = compute_fde(preds[:, idx, :, :, :], gts, valids, reduction=False)
        ade_list.append(ade)
        # fde, valids: (len(test datast), num_obj)
        fde, _ = compute_ade(preds[:, idx, :, :, :], gts, valids, reduction=False)
        fde_list.append(fde)
        wde, _ =  compute_wde(preds[:, idx, :, :, :], gts, valids, reduction=False)
        wde_list.append(wde)

    if torch.is_tensor(preds):
        ade_list = torch.stack(ade_list, dim=0)
        fde_list = torch.stack(fde_list, dim=0)

        ade_err_min, _ = torch.min(ade_list, dim=0)
        ade_err_min = ade_err_min * valids
        fde_err_min, _ = torch.min(fde_list, dim=0)
        fde_err_min = fde_err_min * valids

        ade_err_mean = torch.mean(ade_list, dim=0)
        ade_err_mean = ade_err_mean * valids
        fde_err_mean = torch.mean(fde_list, dim=0)
        fde_err_mean = fde_err_mean * valids

        # numpy divided by 1/n while torch divided by 1 / n-1
        ade_err_std = torch.std(ade_list, dim=0) * np.sqrt((ade_list.shape[0] - 1.) / ade_list.shape[0])
        ade_err_std = ade_err_std * valids
        fde_err_std = torch.std(fde_list, dim=0) * np.sqrt((fde_list.shape[0] - 1.) / fde_list.shape[0])
        fde_err_std = fde_err_std * valids

    else:
        ade_list = np.array(ade_list, dtype=np.float32)  # (num_samples, len(test dataset), num_obj)
        fde_list = np.array(fde_list, dtype=np.float32)  # (num_samples, len(test dataset), num_obj)
        wde_list = np.array(wde_list, dtype=np.float32)  # (num_samples, len(test dataset), num_obj)

        ade_err_min = ade_list.min(axis=0) * valids
        fde_err_min = fde_list.min(axis=0) * valids

        ade_err_mean = ade_list.mean(axis=0) * valids
        fde_err_mean = fde_list.mean(axis=0) * valids

        ade_err_std = ade_list.std(axis=0) * valids
        fde_err_std = fde_list.std(axis=0) * valids

        assert ade_list.shape == (num_samples, len_dataset, 2), ade_list.shape
        assert ade_err_mean.shape == (len_dataset, num_obj), ade_err_mean.shape

        valid_ade, valid_fde = [], []
        for i in range(len_dataset):
            for j in range(num_obj):
                if valids[i, j] > 0:
                    valid_ade.append(ade_list[:, i, j])
                    valid_fde.append(fde_list[:, i, j])
        valid_ade = np.array(valid_ade)
        valid_fde = np.array(valid_fde)
        valid_wde = np.array(wde_list)
        ade_mean_chen = valid_ade.mean()
        fde_mean_chen = valid_fde.mean()
        wde_mean_chen = valid_wde.mean()
        ade_std_chen = valid_ade.std()
        fde_std_chen = valid_fde.std()
        wde_std_chen = valid_wde.std()
        print(f"origin, ade_mean = {ade_mean_chen}, ade_std: {ade_std_chen}")
        print(f"origin, fde_mean = {fde_mean_chen}, fde_std: {fde_std_chen}")
        print(f"origin, wde_mean = {wde_mean_chen}, wde_std: {wde_std_chen}")
    ade_mean = ade_err_mean.sum() / valids.sum()
    fde_mean = fde_err_mean.sum() / valids.sum()
    assert np.isclose(ade_mean, ade_mean_chen, atol=1e-3), (ade_mean, ade_mean_chen)
    assert np.isclose(fde_mean, fde_mean_chen, atol=1e-3), (fde_mean, fde_mean_chen)
    return ade_mean_chen, fde_mean_chen, wde_mean_chen