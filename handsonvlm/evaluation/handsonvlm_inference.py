import copy

import numpy as np
import torch
from tqdm import tqdm

from handsonvlm.evaluation.utils import create_trajectory_video
from handsonvlm.handsonvlm_utils import load_video_frames, load_video
from handsonvlm.model.builder import load_pretrained_model
from handsonvlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from handsonvlm.dataset.epic_dataset import EpicConversationDataset, EpicReasoningConversationDataset

from hoi_forecast.dataset.dataset import get_epic_hoi_dataset_by_name
from hoi_forecast.dataset.epic_structures import EpicHOIDataset
from hoi_forecast.evaluation.traj_eval import evaluate_traj_stochastic

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def evaluate_traj(cur_split_dict):
    print("len(cur_split_dict) = ", len(cur_split_dict))
    preds_traj = []
    gts_traj = []
    valids_traj = []
    hand_pred_cnt = 0
    total_pred_cnt = 0
    for index, batch in enumerate(cur_split_dict.keys()):
        cur_batch_dict = cur_split_dict[batch]
        pred_hand_trajectory = cur_batch_dict['pred_hand_trajectory']
        pred_hand_is_valid = cur_batch_dict['pred_hand_is_valid']
        pred_trajectory_is_valid = cur_batch_dict['pred_trajectory_is_valid']

        total_pred_cnt += 1
        if not pred_trajectory_is_valid:
            continue
        hand_pred_cnt += 1

        future_hands = cur_batch_dict['future_hands']
        future_valid = cur_batch_dict['future_valid']

        if pred_hand_trajectory.shape == (1, 1, 2, 5, 2):
            pred_hand_trajectory = pred_hand_trajectory[:, :, :, 1:, :]
        if future_hands.shape == (1, 2, 5, 2):
            future_hands = future_hands[:, :, 1:, :]

        assert pred_hand_trajectory.shape == (1, 1, 2, 4, 2)
        assert future_hands.shape == (1, 2, 4, 2)
        assert future_valid.shape == (1, 2), future_valid.shape

        preds_traj.append(pred_hand_trajectory)
        gts_traj.append(future_hands)
        valids_traj.append(future_valid)

    gts_traj = np.concatenate(gts_traj)
    preds_traj = np.concatenate(preds_traj)
    valids_traj = np.concatenate(valids_traj)

    ade, fde, wde = evaluate_traj_stochastic(preds_traj, gts_traj, valids_traj)


class HandsOnVLMInference:
    def __init__(self, model_path, model_base, load_8bit, load_4bit, conv_mode):
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path=model_path,
                                                                                                   model_base=model_base,
                                                                                                   model_name=self.model_name,
                                                                                                   load_8bit=load_8bit,
                                                                                                   load_4bit=load_4bit)
        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        if conv_mode is not None and conv_mode != self.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(self.conv_mode, conv_mode, conv_mode))
            self.conv_mode = conv_mode
        self.temperature = 0.5
        self.top_p = 0.9
        self.num_beams = 1

    def init_conversation(self):
        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles


    def inference(self, sample):
        input_ids = sample['input_ids']
        image = sample['image'].half()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                image=image,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=100,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True)
            output_ids = outputs.sequences
            text = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # todo: try to support multi-turn conversation, wait to debug
            self.conv.append_message(self.conv.roles[1], text)
            pred_hand = outputs.pred_hands
            if pred_hand is None:
                pred_hand = torch.zeros(1, 1, 2, 4, 2).cuda().half()
                return pred_hand, False
            total_hand_tokens_number = pred_hand.shape[2]
            pred_hands = pred_hand.unsqueeze(0)
            assert pred_hands.shape == torch.Size([1, 1, 2, total_hand_tokens_number, 2]), pred_hands.shape
            if total_hand_tokens_number > 4:
                pred_hands = pred_hands[:, :, :, -4:, :]
            elif total_hand_tokens_number < 4:
                pred_hands = torch.cat([pred_hands, torch.zeros(1, 1, 2, 4 - total_hand_tokens_number, 2).cuda().half()], dim=3)
        return pred_hands, True


    def evaluate_epic_kitchen_traj(self, test_version, split, use_reason=True):
        if use_reason:
            epic_hoi_dataset: EpicHOIDataset = get_epic_hoi_dataset_by_name(test_version, split,
                                                                            image_processor=self.image_processor,
                                                                            rephrase_rate=1,
                                                                            use_wrong_narration=False,
                                                                            use_percentage=1)
            dataset = EpicReasoningConversationDataset(tokenizer=self.tokenizer, epic_hoi_dataset=epic_hoi_dataset, deterministic=True)
        else:
            epic_hoi_dataset: EpicHOIDataset = get_epic_hoi_dataset_by_name(test_version, split,
                                                                            image_processor=self.image_processor,
                                                                            rephrase_rate=0,
                                                                            use_wrong_narration=False,
                                                                            use_percentage=1)
            dataset = EpicConversationDataset(tokenizer=self.tokenizer, epic_hoi_dataset=epic_hoi_dataset, deterministic=True)
        val_info = {}
        for batch_idx in tqdm(range(len(dataset))):
            sample = dataset[batch_idx]
            sample['image'] = sample['image'].unsqueeze(0).half().cuda()
            sample["future_hands"] = sample["future_hands"].unsqueeze(0).cuda()
            sample['future_valid'] = sample['future_valid'].unsqueeze(0).cuda()

            # init the conversation
            self.init_conversation()
            image_abs_paths = sample["image_abs_paths"]
            prompt = copy.deepcopy(sample['prompt'])
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            # first message
            self.conv.append_message(self.conv.roles[0], prompt)
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()
            sample['input_ids'] = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            pred_hands, pred_trajectory_is_valid = self.inference(sample)
            # evaluate hand trajectory
            future_hands = sample['future_hands'][:, :, :, :].cpu().float().numpy()
            future_valid = sample['future_valid'].cpu().float().numpy()
            result = {}
            result['pred_hand_trajectory'] = pred_hands.cpu().numpy() if pred_trajectory_is_valid else None
            result['pred_hand_is_valid'] = None
            result['pred_trajectory_is_valid'] = pred_trajectory_is_valid
            result['future_hands'] = future_hands
            result['future_valid'] = future_valid
            result["image_abs_paths"] = image_abs_paths
            result['prompt'] = sample['prompt']
            result["answer"] = self.conv.messages[-1][1]
            val_info[batch_idx] = result
            evaluate_traj(val_info)
        return val_info

    def wait_for_user_input(self):
        while True:
            try:
                user_input = input(f"{self.roles[0]}: ")
                user_input = user_input
            except EOFError:
                user_input = ""
            if not user_input:
                print("exit...")
                return None
            return user_input

    def user_input_inference(self, path, output_video_path):
        self.init_conversation()
        user_input = self.wait_for_user_input()

        query_video_path = []

        sample = {}
        if path.endswith("png") or path.endswith("jpg"):
            # single image inference
            for i in range(10):
                query_video_path.append(path)
            image = load_video_frames(visual_path=query_video_path,
                                      processor=self.image_processor,
                                      image_aspect_ratio='square')
        elif path.endswith("mp4"):
            # video branch
            image = load_video(path, self.image_processor, num_frames=10)
        assert image.shape == torch.Size([10, 3, 224, 224]), image.shape
        image = image.unsqueeze(0).repeat(10, 1, 1, 1, 1).reshape(100, 3, 224, 224).unsqueeze(0).half().cuda()
        assert image.shape == torch.Size([1, 100, 3, 224, 224]), image.shape

        sample['image'] = image.cuda()

        prompt = DEFAULT_IMAGE_TOKEN + '\n' + user_input
        # first message
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)

        while True:
            print("current_conv:", self.conv.messages)
            prompt = self.conv.get_prompt()
            sample['input_ids'] = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            pred_hand_trajectory, _, pred_hand_valid, text = self.inference(sample)
            print("response: ", text)
            if pred_hand_valid:
                create_trajectory_video(
                    query_video_path,
                    pred_hand_trajectory,
                    output_video_path,
                )
                break
            user_input = self.wait_for_user_input()
            self.conv.append_message(self.conv.roles[0], user_input)

