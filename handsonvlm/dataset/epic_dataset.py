import os
import json
import copy
import re
import random
from typing import Sequence

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from hoi_forecast.utils.const import anticipation_frames_num
from hoi_forecast.dataset.epic_structures import EpicHOIDataset
from handsonvlm.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, general_trajectory_answer_templates, action_question_templates, general_question_templates, action_answer_templates, action_prediction_templates, general_specific_question_templates
from handsonvlm.dataset.base_dataset import preprocess
from llava import conversation as conversation_lib


def preprocess_multimodal(sources: Sequence[str], mm_use_im_start_end=False) -> dict:
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources



def extract_questions(value):
    # 使用正则表达式匹配双引号内的内容
    pattern = r'"([^"]*)"'
    questions = re.findall(pattern, value)
    return questions


class EpicConversationDataset(Dataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset, deterministic=False):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"

        # load rephrase templates
        reasoning_val_path = "/ocean/projects/cis240031p/cbao/codes/lita/ek100_questions_val.json"
        self.reasoning_templates = {}
        with open(reasoning_val_path, "r") as file:
            rephrase_file = json.load(file)
            cnt = 0
            for key, value in rephrase_file.items():
                questions = extract_questions(value)
                self.reasoning_templates[key] = questions
                cnt += 1
            print(f"loaded reasoning templates, len = {len(self.reasoning_templates)}, total questions = {cnt}")
        self.deterministic = deterministic

    # def get_sources(self, i) -> dict:
    #     hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
    #     hand_traj_str = ""
    #     for j in range(anticipation_frames_num):
    #         hand_traj_str += "<hand_traj>"
    #     narration = hoi_feature_dict["narration"]
    #     selected_answer = random.choice(general_trajectory_answer_templates).format(hand_traj_str)
    #     image_abs_paths = hoi_feature_dict["image_abs_paths"]
    #     last_image_path = image_abs_paths[-1]
    #
    #     if self.epic_hoi_dataset.split == "validation" and last_image_path in self.reasoning_templates and self.epic_hoi_dataset.rephrase_rate == 1:
    #         # assert last_image_path in self.reasoning_templates, f"last_image_path = {last_image_path}"
    #         questions = self.reasoning_templates[last_image_path]
    #         if len(questions) == 0:
    #             print("this is strange that the question list are empty: ", last_image_path)
    #             selected_question = random.choice(action_question_templates).format(narration)
    #         else:
    #
    #             selected_question = random.choice(questions)
    #             print(f"rephrased question = {selected_question}, action narration = {narration}")
    #     else:
    #         print("using explicit narration!!")
    #         selected_question = random.choice(action_question_templates).format(narration)
    #     hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
    #                                          {"from": "gpt", "value": selected_answer}]
    #     hoi_feature_dict['prompt'] = selected_question
    #     return hoi_feature_dict

    def __len__(self):
        return self.epic_hoi_dataset.__len__()

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

        data_dict['future_hands'] = torch.tensor(hoi_feature_dict['future_hands'])
        assert data_dict['future_hands'].shape == torch.Size([2, 5, 2]), data_dict['future_hands'].shape
        data_dict['contact_point'] = torch.tensor(hoi_feature_dict['contact_point'])
        data_dict['future_valid'] = torch.tensor(hoi_feature_dict['future_valid'])
        assert data_dict['future_valid'].shape == torch.Size([2, ]), data_dict['future_valid'].shape
        data_dict['prompt'] = hoi_feature_dict['prompt']
        # print("data_dict['prompt'] = ", data_dict['prompt'])
        data_dict["image_abs_paths"] = hoi_feature_dict["image_abs_paths"]

        # special process for the image
        image = hoi_feature_dict['image']
        assert image.shape == torch.Size([10, 3, 224, 224]), image.shape
        image = image.unsqueeze(0).repeat(10, 1, 1, 1, 1)
        assert image.shape == torch.Size([10, 10, 3, 224, 224]), image.shape
        image = image.reshape(100, 3, 224, 224)
        data_dict['image'] = image
        return data_dict


    def get_sources(self, i) -> dict:
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
        hand_traj_str = ""
        for j in range(anticipation_frames_num):
            hand_traj_str += "<hand_traj>"
        narration = hoi_feature_dict["narration"]
        selected_answer = random.choice(general_trajectory_answer_templates).format(hand_traj_str)
        image_abs_paths = hoi_feature_dict["image_abs_paths"]
        last_image_path = image_abs_paths[-1]

        if self.epic_hoi_dataset.split == "validation" and last_image_path in self.reasoning_templates and self.epic_hoi_dataset.rephrase_rate == 1:
            # assert last_image_path in self.reasoning_templates, f"last_image_path = {last_image_path}"
            questions = self.reasoning_templates[last_image_path]
            if len(questions) == 0:
                print("this is strange that the question list are empty: ", last_image_path)
                selected_question = random.choice(action_question_templates).format(narration)
            else:

                selected_question = random.choice(questions)
                print(f"rephrased question = {selected_question}, action narration = {narration}")
        else:
            print("using explicit narration!!")
            selected_question = random.choice(action_question_templates).format(narration)

        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict

class EpicReasoningConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset, deterministic=False):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"

        # load rephrase templates
        if epic_hoi_dataset.split == "train":
            reasoning_train_path = "/ocean/projects/cis240031p/cbao/codes/lita/ek100_questions.json"
            print("using training reasoning dataset")
        else:
            reasoning_train_path = "/ocean/projects/cis240031p/cbao/codes/lita/ek100_questions_val.json"
            print("using validation reasoning dataset")
        self.reasoning_templates = {}
        with open(reasoning_train_path, "r") as file:
            rephrase_file = json.load(file)
            cnt = 0
            for key, value in rephrase_file.items():
                questions = extract_questions(value)
                self.reasoning_templates[key] = questions
                cnt += 1
            print(f"loaded reasoning templates, len = {len(self.reasoning_templates)}, total questions = {cnt}")
        self.deterministic = deterministic

        self.valid_index = []
        for index, action in tqdm(enumerate(self.epic_hoi_dataset.actions)):
            hoi_feature_dict = self.epic_hoi_dataset.load_hoi_features(action)
            image_abs_paths = hoi_feature_dict['image_abs_paths']
            last_image_path = image_abs_paths[-1]
            if last_image_path in self.reasoning_templates:
                questions = self.reasoning_templates[last_image_path]
                if len(questions) != 0:
                    self.valid_index.append(index)
        print(f"we filter {len(self.valid_index)} actions for reasoning conversation from {len(self.epic_hoi_dataset.actions)} actions")

    def __len__(self):
        return len(self.valid_index)

    def get_sources(self, i) -> dict:
        i = i % len(self.valid_index)
        valid_id = self.valid_index[i]
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(valid_id)
        hand_traj_str = ""
        for j in range(anticipation_frames_num):
            hand_traj_str += "<hand_traj>"
        narration = hoi_feature_dict["narration"]
        selected_answer = random.choice(general_trajectory_answer_templates).format(hand_traj_str)
        image_abs_paths = hoi_feature_dict["image_abs_paths"]
        last_image_path = image_abs_paths[-1]

        questions = self.reasoning_templates[last_image_path]

        print("rephrase using RBHP dataset!!")
        selected_question = random.choice(questions)
        # print(f"rephrased question = {selected_question}, action narration = {narration}")
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict

class EpicMultiturnConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"
        self.deterministic = False

    def get_sources(self, i) -> dict:
        hoi_feature_dict, action = self.epic_hoi_dataset.__getitem__(i)

        hand_traj_str = ""
        for j in range(anticipation_frames_num):
            hand_traj_str += "<hand_traj>"
        narration = hoi_feature_dict["narration"]

        action_prediction_answer_templates = []
        action_prediction_answer_templates.append("The predicted action in this video is {}.".format(narration))
        action_prediction_answer_templates.append("The expected action in this video is {}.".format(narration))
        action_prediction_answer_templates.append("The action that is going to happen in this video is {}. Because there are {} in the video.".format(narration, action.noun))
        convo = []
        mode = random.randint(0, 3)
        if mode == 0:
            # this means single turn conversation
            selected_question = random.choice(action_question_templates).format(narration)
            selected_answer = random.choice(action_answer_templates).format(narration, hand_traj_str)
            convo.append({"from": "human", "value": selected_question})
            convo.append({"from": "gpt", "value": selected_answer})
        elif mode == 1:
            # ask the trajectory in the first turn
            convo.append({"from": "human", "value": random.choice(general_question_templates)})
            convo.append({"from": "gpt", "value": random.choice(general_trajectory_answer_templates).format(hand_traj_str)})
        elif mode == 2:
            # two turns conversation
            # ask the possible action in the first turn, then ask it to produce the hand trajectory
            convo.append({"from": "human", "value": random.choice(action_prediction_templates)})
            convo.append({"from": "gpt", "value": random.choice(action_prediction_answer_templates)})
            # ask the hand trajectory in the second turn
            convo.append({"from": "human", "value": random.choice(general_specific_question_templates)})
            convo.append({"from": "gpt", "value": random.choice(general_trajectory_answer_templates).format(hand_traj_str)})
        elif mode == 3:
            # two turns conversation
            # ask the trajectory in the first turn, then ask it to predict the action
            convo.append({"from": "human", "value": random.choice(general_question_templates)})
            convo.append({"from": "gpt", "value": random.choice(general_trajectory_answer_templates).format(hand_traj_str)})
            # predict the action
            convo.append({"from": "human", "value": random.choice(action_prediction_templates)})
            convo.append({"from": "gpt", "value": random.choice(action_prediction_answer_templates)})
        hoi_feature_dict['conversations'] = convo

        hoi_feature_dict['prompt'] = ""
        return hoi_feature_dict


class EpicTextOnlyConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"


    def get_sources(self, i) -> dict:
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
        gt_hands = hoi_feature_dict['gt_hands'][:, 1:, :]
        assert gt_hands.shape == (2, 4, 2), gt_hands.shape
        # convert the gt_hands to hand_traj_str
        hand_traj_string_left = ""
        hand_traj_string_right = ""
        for j in range(4):
            gt_hand_right = gt_hands[0, j]
            gt_hand_left = gt_hands[1, j]
            hand_traj_string_right += "(" + str(gt_hand_right[0]) + "," + str(gt_hand_right[1]) + ")"
            hand_traj_string_left += "(" + str(gt_hand_left[0]) + "," + str(gt_hand_left[1]) + ")"
        hand_traj_str = "right hand: " + hand_traj_string_right + " left hand: " + hand_traj_string_left
        narration = hoi_feature_dict["narration"]

        selected_question = random.choice(action_question_templates).format(narration)
        selected_answer = random.choice(action_answer_templates).format(narration, hand_traj_str)
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict


class EpicPixelSeqConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"
        self.n_bins = 400

    def get_sources(self, i) -> dict:
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
        gt_hands = hoi_feature_dict['gt_hands'][:, 1:, :]
        assert gt_hands.shape == (2, 4, 2), gt_hands.shape
        # convert the gt_hands to hand_traj_str
        hand_traj_string_left = ""
        hand_traj_string_right = ""

        def float_2_bin(x):
            x_disc = int(np.floor(x * self.n_bins))
            x_disc = max(0, min(x_disc, self.n_bins - 1))
            bin_str = f"<bin_{x_disc}>"
            return bin_str

        for j in range(4):
            gt_hand_right = gt_hands[0, j]
            gt_hand_left = gt_hands[1, j]
            hand_traj_string_right += "(" + float_2_bin(gt_hand_right[0]) + "," + float_2_bin(gt_hand_right[1]) + ")"
            hand_traj_string_left += "(" + float_2_bin(gt_hand_left[0]) + "," + float_2_bin(gt_hand_left[1]) + ")"
        hand_traj_str = "right hand: " + hand_traj_string_right + " left hand: " + hand_traj_string_left
        narration = hoi_feature_dict["narration"]

        selected_question = random.choice(action_question_templates).format(narration)
        selected_answer = random.choice(action_answer_templates).format(narration, hand_traj_str)
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        # print("convo: ", hoi_feature_dict['conversations'])
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict


class EpicSeperateHandConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"

    def get_sources(self, i) -> dict:
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
        gt_hands = hoi_feature_dict['gt_hands'][:, 1:, :]
        assert gt_hands.shape == (2, 4, 2), gt_hands.shape
        # convert the gt_hands to hand_traj_str
        hand_traj_string_left = ""
        hand_traj_string_right = ""

        for j in range(4):
            hand_traj_string_right += "<hand_traj>"
            hand_traj_string_left += "<hand_traj>"
        hand_traj_str = "right hand: " + hand_traj_string_right + " left hand: " + hand_traj_string_left
        narration = hoi_feature_dict["narration"]

        selected_question = random.choice(action_question_templates).format(narration)
        selected_answer = random.choice(action_answer_templates).format(narration, hand_traj_str)
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        # print("convo: ", hoi_feature_dict['conversations'])
        hoi_feature_dict['prompt'] = selected_question
        return hoi_feature_dict


class EpicNarrationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"

    def get_sources(self, i) -> dict:
        hoi_feature_dict, _ = self.epic_hoi_dataset.__getitem__(i)
        narration = hoi_feature_dict["narration"]
        hoi_feature_dict['conversations'] = [{"from": "human", "value": narration},
                                             {"from": "gpt", "value": narration}]
        hoi_feature_dict['prompt'] = narration
        return hoi_feature_dict


class EpicActionPredictionConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is False, "The dataset should use correct narration"

    def get_sources(self, i) -> dict:
        hoi_feature_dict, action = self.epic_hoi_dataset.__getitem__(i)
        narration = hoi_feature_dict["narration"]
        action_prediction_answer_templates = []
        action_prediction_answer_templates.append("The predicted action in this video is {}.".format(narration))
        action_prediction_answer_templates.append("The expected action in this video is {}.".format(narration))
        action_prediction_answer_templates.append("The action that is going to happen in this video is {}. Because there are {} in the video.".format(narration, action.noun))
        selected_question = random.choice(action_prediction_templates)
        selected_answer = random.choice(action_prediction_answer_templates)
        hoi_feature_dict['conversations'] = [{"from": "human", "value": selected_question},
                                             {"from": "gpt", "value": selected_answer}]
        return hoi_feature_dict


class EpicNarrationRejectionConversationDataset(EpicConversationDataset):
    def __init__(self, tokenizer, epic_hoi_dataset: EpicHOIDataset):
        super(EpicConversationDataset, self).__init__()
        self.epic_hoi_dataset: EpicHOIDataset = epic_hoi_dataset
        self.tokenizer = tokenizer
        assert epic_hoi_dataset.use_wrong_narration is True, "The dataset should use wrong narration"

    def get_sources(self, i) -> dict:
        sample, wrong_action = self.epic_hoi_dataset.__getitem__(i)

        wrong_narration = sample['narration']
        selected_question = random.choice(action_question_templates).format(wrong_narration)
        reject_answer_templates = []
        reject_answer_templates.append("I'm sorry, it seems impossible to {} in this video.".format(wrong_narration))
        reject_answer_templates.append("I'm sorry, I can't provide the hand trajectory because there is no {} in this video.".format(wrong_action.noun))
        selected_answer = random.choice(reject_answer_templates)
        sample['conversations'] = [{"from": "human", "value": selected_question},
                                   {"from": "gpt", "value": selected_answer}]
        return sample
