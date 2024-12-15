import os.path
import random
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from lmdbdict import lmdbdict

from hoi_forecast.dataset.action_sampler import ActionAnticipationSampler
from hoi_forecast.dataset.epic_utils import process_eval_video_info, process_video_info
from hoi_forecast.dataset.epic_action import EpicAction
from hoi_forecast.dataset.video_utils import load_video_frames
from hoi_forecast.utils.const import origin_fps, num_actions_prev, get_label_dir, observation_seconds, anticipation_seconds, future_hand_num, observation_frames_num, frame_template, EPIC_KITCHEN_DATASET_DIR, image_aspect_ratio, get_eval_label_path, get_lmdb_path, fps


class EpicVideo(object):
    def __init__(self, df_video, ori_fps, partition, t_ant=None):
        self.df = df_video
        self.ori_fps = ori_fps
        self.partition = partition
        self.t_ant = t_ant

        self.actions, self.actions_invalid = self.get_actions()
        self.duration = max([a.stop_time for a in self.actions])

    def get_actions(self):
        actions = []
        _actions_all = []
        actions_invalid = []
        for _, row in self.df.iterrows():
            action_args = {
                'uid': row.uid,
                'participant_id': row.participant_id,
                'video_id': row.video_id,
                'verb': row.verb if 'test' not in self.partition else None,
                'verb_class': row.verb_class if 'test' not in self.partition else None,
                'noun': row.noun if 'test' not in self.partition else None,
                'noun_class': row.noun_class if 'test' not in self.partition else None,
                'all_nouns': row.all_nouns if 'test' not in self.partition else None,
                'all_noun_classes': row.all_noun_classes if 'test' not in self.partition else None,
                'start_frame': row.start_frame,
                'stop_frame': row.stop_frame,
                'start_time': row.start_time,
                'stop_time': row.stop_time,
                'ori_fps': self.ori_fps,
                'partition': self.partition,
                'action': row.action if 'test' not in self.partition else None,
                'action_class': row.action_class if 'test' not in self.partition else None,
                'narration': row.narration if 'test' not in self.partition else None,
            }
            action: EpicAction = EpicAction(**action_args)
            action.set_previous_actions([aa for aa in _actions_all])
            assert self.t_ant is not None
            assert self.t_ant > 0.0
            if action.start_time - self.t_ant >= 0:
                actions += [action]
            else:
                actions_invalid += [action]
            _actions_all += [action]
        return actions, actions_invalid


class EpicDataset(Dataset):
    def __init__(self, df, split):
        super().__init__()
        self.split = split
        self.df = df
        self.videos = self._get_videos()
        self.actions, self.actions_invalid = self._get_actions()

    def _get_videos(self):
        video_ids = sorted(list(set(self.df['video_id'].values.tolist())))
        videos = []
        pbar = tqdm(desc=f'Loading {self.split} samples', total=len(self.df))
        for video_id in video_ids:
            video_args = {
                'df_video': self.df[self.df['video_id'] == video_id].copy(),
                'ori_fps': origin_fps,
                'partition': self.split,
                't_ant': anticipation_seconds
            }
            video = EpicVideo(**video_args)
            videos += [video]
            pbar.update(len(video.actions))
        pbar.close()
        return videos

    def _get_actions(self):
        actions = []
        actions_invalid = []
        for video in self.videos:
            actions += video.actions
            actions_invalid += video.actions_invalid
        return actions, actions_invalid

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        a = self.actions[idx]
        sample = {'uid': a.uid}

        if 'test' not in self.split:
            sample['verb_class'] = a.verb_class
            sample['noun_class'] = a.noun_class
            sample['action_class'] = a.action_class

        actions_prev = [-1] + [aa.action_class for aa in a.actions_prev]
        actions_prev = actions_prev[-num_actions_prev:]
        if len(actions_prev) < num_actions_prev:
            actions_prev = actions_prev[0:1] * (num_actions_prev - len(actions_prev)) + actions_prev
        actions_prev = np.array(actions_prev, dtype=np.int64)
        sample['action_class_prev'] = actions_prev
        return sample


class EpicHOIDataset(EpicDataset):
    def __init__(self, df, split, ek_version, image_processor=None, rephrase_rate=0., use_wrong_narration=False, use_percentage=1.0):
        EpicDataset.__init__(self, df=df, split=split)
        self.ek_version = ek_version
        self.rephrase_rate = rephrase_rate
        self.use_wrong_narration = use_wrong_narration
        self.discarded_labels, self.discarded_ids = self._get_discarded()

        # we won't load full image here if image_processor is None
        self.image_processor = image_processor
        self.sampler = ActionAnticipationSampler(observation_seconds=observation_seconds, anticipation_seconds=anticipation_seconds, fps=fps, origin_fps=origin_fps)
        # load labels
        self.label_dir = get_label_dir(ek_version)
        # preload all the labels in memory
        self.labels = {}

        if os.path.exists(os.path.join(self.label_dir, f"{ek_version}_{split}_labels.npy")):
            self.labels = np.load(os.path.join(self.label_dir, f"{ek_version}_{split}_labels.npy"), allow_pickle=True).item()
            print(f"Loaded labels from {self.label_dir}")
        else:
            print(f"Loading labels from {self.label_dir}, this may take a while...But accelerate the training process!!! Whoo!")
            for filename in tqdm(os.listdir(self.label_dir)):
                if filename.startswith("label_") and filename.endswith(".pkl"):
                    uid = int(filename[6:-4].strip())  # Extract UID from filename
                    label_path = os.path.join(self.label_dir, filename)
                    try:
                        with open(label_path, 'rb') as f:
                            video_info = pickle.load(f)
                        future_hands, contact_point, future_valid, _ = process_video_info(video_info)
                        assert future_hands.shape == (2, future_hand_num, 2), future_hands.shape
                        assert contact_point.shape == (2,), contact_point.shape
                        assert future_valid.shape == (2,), future_valid.shape
                        # print("gt_hand_valid = ", gt_hand_valid)
                        self.labels[uid] = {
                            'future_hands': future_hands,
                            'contact_point': contact_point,
                            'future_valid': future_valid
                        }
                    except Exception as e:
                        print(f"Error loading label {filename}: {e}")
            # dump the labels into a new npz file
            np.save(os.path.join(self.label_dir, f"{ek_version}_{split}_labels.npy"), self.labels)

        # re-filter the actions according to the labels
        filter_actions = []
        for action in self.actions:
            if action.uid in self.labels:
                filter_actions.append(action)
        # Sort the filter_actions list by uid
        filter_actions.sort(key=lambda action: action.uid)

        # update the determinstic actions based on use_percentage
        # If use_percentage is less than 1.0, get the front percentage of filter_actions
        if use_percentage < 1.0 and use_percentage > 0.0:
            num_actions_to_keep = int(len(filter_actions) * use_percentage)
            filter_actions = filter_actions[:num_actions_to_keep]
        elif use_percentage == 0:
            filter_actions = filter_actions[:1]
        print(f"We filtered out {len(filter_actions)} from {len(self.actions)} actions., {ek_version}, {split}")
        self.actions = filter_actions

        # load hoi features
        self.lmdb_path = get_lmdb_path(ek_version, split)
        self.env = lmdbdict(self.lmdb_path, 'r')

    def __len__(self):
        return len(self.actions)

    def _get_discarded(self):
        discarded_ids = []
        discarded_labels = []
        if 'train' not in self.split:
            label_type = ['verb', 'noun', 'action']
        else:
            label_type = 'action'
        if 'test' in self.split:
            challenge = True
        else:
            challenge = False

        for action in self.actions_invalid:
            discarded_ids.append(action.uid)
            if isinstance(label_type, list):
                if challenge:
                    discarded_labels.append(-1)
                else:
                    verb, noun, action_class = action.verb_class, action.noun_class, action.action_class
                    label = np.array([verb, noun, action_class], dtype=np.int64)
                    discarded_labels.append(label)
            else:
                if challenge:
                    discarded_labels.append(-1)
                else:
                    action_class = action.action_class
                    discarded_labels.append(action_class)
        return discarded_labels, discarded_ids

    def sample_different_action(self, action: EpicAction) -> EpicAction:
        verb = action.verb
        verb_class = action.verb_class
        noun = action.noun
        noun_class = action.noun_class
        while True:
            sampled_action: EpicAction = random.choice(self.actions)
            sampled_action_verb = sampled_action.verb
            sampled_action_verb_class = sampled_action.verb_class
            sampled_action_noun = sampled_action.noun
            sampled_action_noun_class = sampled_action.noun_class
            if sampled_action.narration is None or sampled_action.narration == "":
                continue
            if verb != sampled_action_verb and verb_class != sampled_action_verb_class and noun != sampled_action_noun and noun_class != sampled_action_noun_class:
                return sampled_action


    def load_image_paths(self, action: EpicAction):
        frame_aligned_observation_times, observation_frame_idxs = self.sampler(action)
        assert observation_frames_num <= len(observation_frame_idxs), \
            "num of observation exceed the limit of {}, set smaller t_observe, current is {}".format(len(observation_frame_idxs), observation_seconds)
        frames_names = [frame_template.format(i) for i in observation_frame_idxs]
        start_frame_idx = len(observation_frame_idxs) - observation_frames_num
        frames_names = frames_names[start_frame_idx:]
        image_paths = []
        for f_name in frames_names:
            # full_name: e.g. 'P24/rgb_frames/P24_05/frame_0000075700.jpg'
            full_name = os.path.join(action.participant_id, "rgb_frames", action.video_id, f_name)
            image_paths.append(full_name)
        return image_paths

    def load_hoi_features(self, action: EpicAction):
        frame_aligned_observation_times, observation_frame_idxs = self.sampler(action)
        assert observation_frames_num <= len(observation_frame_idxs), \
            "num of observation exceed the limit of {}, set smaller t_observe, current is {}".format(len(observation_frame_idxs), observation_seconds)
        frames_names = [frame_template.format(i) for i in observation_frame_idxs]
        start_frame_idx = len(observation_frame_idxs) - observation_frames_num
        frames_names = frames_names[start_frame_idx:]

        full_names = []
        image_abs_paths = []
        global_feats, global_masks = [], []
        rhand_feats, rhand_masks, rhand_bboxs = [], [], []
        lhand_feats, lhand_masks, lhand_bboxs = [], [], []
        robj_feats, robj_masks, robj_bboxs = [], [], []
        lobj_feats, lobj_masks, lobj_bboxs = [], [], []

        for f_name in frames_names:
            # full_name: e.g. 'P24/rgb_frames/P24_05/frame_0000075700.jpg'
            full_name = os.path.join(action.participant_id, "rgb_frames", action.video_id, f_name)
            image_abs_path = os.path.join(EPIC_KITCHEN_DATASET_DIR, full_name)
            image_abs_paths.append(image_abs_path)
            full_names.append(full_name)
            # f_dict: 'GLOBAL_FEAT',
            # 'HAND_RIGHT_FEAT', 'HAND_RIGHT_BBOX', 'OBJECT_RIGHT_FEAT', 'OBJECT_RIGHT_BBOX',
            # 'HAND_LEFT_FEAT', 'HAND_LEFT_BBOX', 'OBJECT_LEFT_FEAT', 'OBJECT_LEFT_BBOX']
            key_enc = full_name.strip().encode()
            if key_enc not in self.env:
                raise KeyError("invalid key {}, check lmdb file in {}".format(full_name.strip(), self.lmdb_path))


            f_dict = self.env[key_enc]
            """
            f_dict is a dictionary with the following keys
            - 'GLOBAL_FEAT': np.ndarray of shape (2048,)
            - 'HAND_RIGHT_FEAT': np.ndarray of shape (2048,)
            - 'HAND_RIGHT_BBOX': np.ndarray of shape (4,)
            - 'OBJECT_RIGHT_FEAT': np.ndarray of shape (2048,)
            - 'OBJECT_RIGHT_BBOX': np.ndarray of shape (4,)
            - 'HAND_LEFT_FEAT': np.ndarray of shape (2048,)
            - 'HAND_LEFT_BBOX': np.ndarray of shape (4,)
            - 'OBJECT_LEFT_FEAT': np.ndarray of shape (2048,)
            - 'OBJECT_LEFT_BBOX': np.ndarray of shape (4,)
            """

            global_feat = f_dict['GLOBAL_FEAT']
            global_masks.append(1)
            global_feats.append(global_feat)

            if 'HAND_RIGHT_FEAT' in f_dict:
                rhand_feat = f_dict['HAND_RIGHT_FEAT']
            else:
                rhand_feat = np.zeros_like(global_feat, dtype=np.float32)
            rhand_feats.append(rhand_feat)

            if 'HAND_LEFT_FEAT' in f_dict:
                lhand_feat = f_dict['HAND_LEFT_FEAT']
            else:
                lhand_feat = np.zeros_like(global_feat, dtype=np.float32)
            lhand_feats.append(lhand_feat)

            if 'OBJECT_RIGHT_FEAT' in f_dict:
                robj_feat = f_dict['OBJECT_RIGHT_FEAT']
            else:
                robj_feat = np.zeros_like(global_feat, dtype=np.float32)
            robj_feats.append(robj_feat)

            if 'OBJECT_LEFT_FEAT' in f_dict:
                lobj_feat = f_dict['OBJECT_LEFT_FEAT']
            else:
                lobj_feat = np.zeros_like(global_feat, dtype=np.float32)
            lobj_feats.append(lobj_feat)

            if 'HAND_RIGHT_BBOX' in f_dict:
                rhand_bbox = f_dict['HAND_RIGHT_BBOX']
                rhand_masks.append(1)
            else:
                cx, cy = (0.75, 1.5)
                sx, sy = (0.1, 0.1)
                rhand_bbox = np.array([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2])
                rhand_masks.append(0)
            rhand_bboxs.append(rhand_bbox)

            if 'HAND_LEFT_BBOX' in f_dict:
                lhand_bbox = f_dict['HAND_LEFT_BBOX']
                lhand_masks.append(1)
            else:
                cx, cy = (0.25, 1.5)
                sx, sy = (0.1, 0.1)
                lhand_bbox = np.array([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2])
                lhand_masks.append(0)
            lhand_bboxs.append(lhand_bbox)

            if 'OBJECT_RIGHT_BBOX' in f_dict:
                robj_bbox = f_dict['OBJECT_RIGHT_BBOX']
                robj_masks.append(1)
            else:
                robj_bbox = np.array([0.0, 0.0, 1.0, 1.0])
                robj_masks.append(0)
            robj_bboxs.append(robj_bbox)

            if 'OBJECT_LEFT_BBOX' in f_dict:
                lobj_bbox = f_dict['OBJECT_LEFT_BBOX']
                lobj_masks.append(1)
            else:
                lobj_bbox = np.array([0.0, 0.0, 1.0, 1.0])
                lobj_masks.append(0)
            lobj_bboxs.append(lobj_bbox)

        global_feats = np.stack(global_feats, axis=0)
        rhand_feats = np.stack(rhand_feats, axis=0)
        lhand_feats = np.stack(lhand_feats, axis=0)
        robj_feats = np.stack(robj_feats, axis=0)
        lobj_feats = np.stack(lobj_feats, axis=0)

        feats = np.stack((global_feats, rhand_feats, lhand_feats, robj_feats, lobj_feats), axis=0)

        rhand_bboxs = np.stack(rhand_bboxs, axis=0)
        lhand_bboxs = np.stack(lhand_bboxs, axis=0)
        robj_bboxs = np.stack(robj_bboxs, axis=0)
        lobj_bboxs = np.stack(lobj_bboxs, axis=0)

        bbox_feats = np.stack((rhand_bboxs, lhand_bboxs, robj_bboxs, lobj_bboxs), axis=0)

        global_masks = np.stack(global_masks, axis=0)
        rhand_masks = np.stack(rhand_masks, axis=0)
        lhand_masks = np.stack(lhand_masks, axis=0)
        robj_masks = np.stack(robj_masks, axis=0)
        lobj_masks = np.stack(lobj_masks, axis=0)

        valid_masks = np.stack((global_masks, rhand_masks, lhand_masks, robj_masks, lobj_masks), axis=0)
        assert feats.shape == (5, observation_frames_num, 1024), feats.shape
        assert bbox_feats.shape == (4, observation_frames_num, 4), bbox_feats.shape
        assert valid_masks.shape == (5, observation_frames_num), valid_masks.shape
        assert len(frame_aligned_observation_times) == observation_frames_num, len(frame_aligned_observation_times)
        assert len(observation_frame_idxs) == observation_frames_num, len(observation_frame_idxs)

        hoi_feature_dict = {"name": full_names, "feat": feats, "bbox_feat": bbox_feats, "valid_mask": valid_masks, 'times': frame_aligned_observation_times,
                            'start_time': action.start_time, 'frames_idxs': observation_frame_idxs,
                            "image_abs_paths": image_abs_paths}

        return hoi_feature_dict

    def __getitem__(self, idx):
        action: EpicAction = self.actions[idx]

        hoi_feature_dict = self.load_hoi_features(action)

        if self.image_processor is not None:
            image: torch.Tensor = load_video_frames(hoi_feature_dict['image_abs_paths'], self.image_processor, image_aspect_ratio)
            assert image.shape == (observation_frames_num, 3, 224, 224), image.shape
            hoi_feature_dict['image'] = image
        else:
            hoi_feature_dict['image'] = torch.zeros((observation_frames_num, 3, 224, 224), dtype=torch.float32)

        # update the hoi_feature_dict using correct action
        hoi_feature_dict['uid'] = action.uid
        hoi_feature_dict.update(self.labels[action.uid])
        if 'test' not in self.split:
            hoi_feature_dict['verb_class'] = action.verb_class
            hoi_feature_dict['noun_class'] = action.noun_class
            hoi_feature_dict['action_class'] = action.action_class
            hoi_feature_dict['label'] = np.array([action.verb_class, action.noun_class, action.action_class], dtype=np.int64)

        # handle narration, we support use_wrong_narration and rephrase_rate
        # only the returned ACTION and NARRATION will be affected by use_wrong_narration
        if self.use_wrong_narration:
            action = self.sample_different_action(action)
        narration = action.narration
        if narration is None:
            print("warning, this sample does not have narration! set to empty")
            narration = ""
        if random.random() < self.rephrase_rate:
            from handsonvlm.constants import rephrease_narration
            hoi_feature_dict['narration'] = rephrease_narration(narration)
        else:
            hoi_feature_dict['narration'] = narration


        if isinstance(narration, list):
            narration = narration[0]

        # assert isinstance(narration, str), narration
        return hoi_feature_dict, action


class EpicHOIDatasetEval(EpicHOIDataset):
    def __init__(self, df, split, ek_version, image_processor=None, rephrase_rate=0., use_wrong_narration=False):
        EpicDataset.__init__(self, df=df, split=split)
        self.ek_version = ek_version
        self.discarded_labels, self.discarded_ids = self._get_discarded()
        self.rephrase_rate = rephrase_rate
        self.use_wrong_narration = use_wrong_narration

        self.sampler = ActionAnticipationSampler(observation_seconds=observation_seconds, anticipation_seconds=anticipation_seconds, fps=fps, origin_fps=origin_fps)
        self.image_processor = image_processor
        # load labels
        eval_label_path = get_eval_label_path(ek_version)
        with open(eval_label_path, 'rb') as f:
            self.eval_labels = pickle.load(f)
        # load hoi features
        self.lmdb_path = get_lmdb_path(ek_version, split)
        self.env = lmdbdict(self.lmdb_path, 'r')

    def load_eval_labels(self, uid):
        video_info: dict = self.eval_labels[uid]
        gt_hands, gt_hand_valid = process_eval_video_info(video_info)
        assert gt_hands.shape == (2, future_hand_num, 2), gt_hands.shape
        assert gt_hand_valid.shape == (2, future_hand_num), gt_hand_valid.shape
        return gt_hands, gt_hand_valid

    def __getitem__(self, idx):
        action: EpicAction = self.actions[idx]

        hoi_feature_dict = self.load_hoi_features(action)
        if self.image_processor is not None:
            image = load_video_frames(hoi_feature_dict['image_abs_paths'], self.image_processor, image_aspect_ratio)
            assert image.shape == (observation_frames_num, 3, 224, 224), image.shape
            hoi_feature_dict['image'] = image
        else:
            hoi_feature_dict['image'] = torch.zeros((observation_frames_num, 3, 224, 224), dtype=torch.float32)
        hoi_feature_dict['uid'] = action.uid

        hoi_feature_dict['verb_class'] = action.verb_class
        hoi_feature_dict['noun_class'] = action.noun_class
        hoi_feature_dict['action_class'] = action.action_class
        hoi_feature_dict['label'] = np.array([action.verb_class, action.noun_class, action.action_class], dtype=np.int64)

        gt_hands, gt_hand_valid = self.load_eval_labels(action.uid)
        hoi_feature_dict['gt_hands'] = gt_hands
        hoi_feature_dict['gt_hand_valid'] = gt_hand_valid
        hoi_feature_dict['gt_label_valid'] = True

        # handle narration, we support use_wrong_narration and rephrase_rate
        # only the returned ACTION and NARRATION will be affected by use_wrong_narration
        if self.use_wrong_narration:
            action = self.sample_different_action(action)
        narration = action.narration
        if random.random() < self.rephrase_rate:
            hoi_feature_dict['narration'] = rephrease_narration(narration)
        else:
            hoi_feature_dict['narration'] = narration
        return hoi_feature_dict, action
