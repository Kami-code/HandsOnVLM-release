import numpy as np

from hoi_forecast.utils.const import anticipation_seconds, fps, epic_img_width, epic_img_height, future_hand_num


def sample_hand_trajectory(trajectory):
    assert trajectory.shape == (21, 2), trajectory.shape

    origin_fps = int((len(trajectory) - 1) / anticipation_seconds)
    gap = int(origin_fps // fps)
    stop_idx = len(trajectory)
    indices = [0] + list(range(gap, stop_idx, gap))
    hand_trajectory = []
    for idx in indices:
        x, y = trajectory[idx]
        x, y, = x / epic_img_width, y / epic_img_height
        hand_trajectory.append(np.array([x, y], dtype=np.float32))
    hand_trajectory = np.array(hand_trajectory, dtype=np.float32)
    assert hand_trajectory.shape == (future_hand_num, 2), hand_trajectory.shape
    return hand_trajectory, indices


def process_video_info(video_info):
    # video_info:  dict_keys(['frame_indices', 'homography',  'contact',  'hand_trajs', 'obj_trajs', 'affordance'])

    frames_idxs = video_info["frame_indices"]
    hand_trajs = video_info["hand_trajs"]

    obj_affordance = video_info['affordance']['select_points_homo']
    num_points = obj_affordance.shape[0]
    select_idx = np.random.choice(num_points, 1, replace=False)
    contact_point = obj_affordance[select_idx]
    cx = contact_point[0][0] / epic_img_width
    cy = contact_point[0][1] / epic_img_height
    contact_point = np.array([cx, cy], dtype=np.float32)

    valid_mask = np.zeros(2)
    if "RIGHT" in hand_trajs:
        future_right_hand, _ = sample_hand_trajectory(hand_trajs["RIGHT"]['traj'])
        valid_mask[0] = True
    else:
        future_right_hand = np.repeat(np.array([[0.75, 1.5]], dtype=np.float32), future_hand_num, axis=0)
    if "LEFT" in hand_trajs:
        future_left_hand, _ = sample_hand_trajectory(hand_trajs['LEFT']['traj'])
        valid_mask[1] = True
    else:
        future_left_hand = np.repeat(np.array([[0.25, 1.5]], dtype=np.float32), future_hand_num, axis=0)

    future_hands = np.stack((future_right_hand, future_left_hand), axis=0)
    assert future_hands.shape == (2, future_hand_num, 2), future_hands.shape
    future_valid = np.array(valid_mask, dtype=np.int64)

    last_frame_index = frames_idxs[0]
    return future_hands, contact_point, future_valid, last_frame_index


def process_eval_video_info(video_info):
    if "RIGHT" in video_info:
        future_right_hand = video_info["RIGHT"]
    else:
        future_right_hand = np.repeat(np.array([[0.75, 1.5]], dtype=np.float32), future_hand_num, axis=0)

    if "LEFT" in video_info:
        future_left_hand = video_info['LEFT']
    else:
        future_left_hand = np.repeat(np.array([[0.25, 1.5]], dtype=np.float32), future_hand_num, axis=0)

    gt_hands = np.stack((future_right_hand, future_left_hand), axis=0)
    gt_hand_valid = np.all((gt_hands >= 0.) & (gt_hands <= 1.), axis=-1)
    assert gt_hands.shape == (2, future_hand_num, 2), gt_hands.shape
    assert gt_hand_valid.shape == (2, future_hand_num), gt_hand_valid.shape
    return gt_hands, gt_hand_valid











