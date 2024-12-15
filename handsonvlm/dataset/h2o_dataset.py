import copy
import random

import torch
import numpy as np
from torch.utils.data import Dataset


from hoi_forecast.dataset_general.dataset_h2o import H2OHandTrajectoryDataset
from handsonvlm.constants import DEFAULT_IMAGE_TOKEN
from handsonvlm.dataset.base_dataset import preprocess
from handsonvlm.dataset.epic_dataset import preprocess_multimodal
from hoi_forecast.utils.const import anticipation_frames_num
from handsonvlm.constants import action_question_templates, action_answer_templates


class H2OConversationDataset(Dataset):
    def __init__(self, tokenizer, dataset: H2OHandTrajectoryDataset, deterministic=False):
        super(H2OConversationDataset, self).__init__()
        self.dataset: H2OHandTrajectoryDataset = dataset
        self.tokenizer = tokenizer
        self.deterministic = deterministic

    def get_sources(self, i) -> dict:
        hoi_feature_dict = self.dataset.__getitem__(i)

        hand_traj_str = ""
        for j in range(anticipation_frames_num):
            hand_traj_str += "<hand_traj>"

        narration = hoi_feature_dict["narration"]

        selected_question = random.choice(action_question_templates).format(narration)
        selected_answer = random.choice(action_answer_templates).format(narration, hand_traj_str)
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, i):
        if not self.deterministic:
            rng = np.random.RandomState()  # local rng independent of global
            i = rng.randint(0, self.__len__())  # random index
        hoi_feature_dict: dict = self.get_sources(i)
        # add <image> to first human
        # print("before conversation: ", hoi_feature_dict['conversations'])
        # print(" hoi_feature_dict['conversations'][0]['value'] = ", hoi_feature_dict['conversations'][0]['value'])
        hoi_feature_dict['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + hoi_feature_dict['conversations'][0]['value']

        # print("conversation: ", hoi_feature_dict['conversations'])

        # tokenize the sentences
        sources = copy.deepcopy(hoi_feature_dict)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), mm_use_im_start_end=False)
        data_dict = preprocess(sources, self.tokenizer, has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # update data_dict with following keys
        data_dict['feat'] = torch.tensor(hoi_feature_dict['feat'])
        assert data_dict['feat'].shape == torch.Size([5, 10, 1024]), data_dict['feat'].shape
        data_dict['bbox_feat'] = torch.tensor(hoi_feature_dict['bbox_feat'])
        assert data_dict['bbox_feat'].shape == torch.Size([4, 10, 4]), data_dict['bbox_feat'].shape
        data_dict['valid_mask'] = torch.tensor(hoi_feature_dict['valid_mask'])
        assert data_dict['valid_mask'].shape == torch.Size([5, 10]), data_dict['valid_mask'].shape

        data_dict['gt_hands'] = torch.tensor(hoi_feature_dict['gt_hands'])
        assert data_dict['gt_hands'].shape == torch.Size([2, 5, 2]), data_dict['gt_hands'].shape
        data_dict['gt_contact_point'] = torch.tensor(hoi_feature_dict['gt_contact_point'])
        data_dict['gt_hand_valid'] = torch.tensor(hoi_feature_dict['gt_hand_valid'])
        assert data_dict['gt_hand_valid'].shape == torch.Size([2, 5]), data_dict['gt_hand_valid'].shape
        data_dict['prompt'] = hoi_feature_dict['prompt']
        data_dict["image_abs_paths"] = hoi_feature_dict["image_abs_paths"]


        # special process for the image
        image = hoi_feature_dict['image']
        assert image.shape == torch.Size([10, 3, 224, 224]), image.shape
        image = image.unsqueeze(0).repeat(10, 1, 1, 1, 1)
        assert image.shape == torch.Size([10, 10, 3, 224, 224]), image.shape
        image = image.reshape(100, 3, 224, 224)
        data_dict['image'] = image
        return data_dict



