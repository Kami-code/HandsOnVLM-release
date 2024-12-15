from typing import Dict
from dataclasses import dataclass


from torch.utils.data import Dataset
import transformers
from transformers import CLIPImageProcessor

from hoi_forecast.dataset.dataset import get_epic_hoi_dataset_by_name
from handsonvlm.dataset.epic_dataset import *
from handsonvlm.arguments import DataArguments
from handsonvlm.constants import IGNORE_INDEX
from lita.dataset.dvc_dataset import DVCDataset_activitynet, DVCDataset_howto100m, DVCDataset_youcook2, DVCDataset_vitt
from lita.dataset.event_loc_dataset import EventLocDataset_activitynet, EventLocDataset_youcook2, EventLocDataset_vitt
from lita.dataset.vidqa_dataset import VidQADataset_msrvttqa, VidQADataset_msvdqa, VidQADataset_nextqa, VidQADataset_videochat
from lita.dataset.instruct_dataset import LlavaDataset
from lita.dataset.temporal_reasoning_dataset import TemporalReasoningDataset_activitynet


class HybridDataset(Dataset):
    def __init__(self, data_path: str, data_args: DataArguments, tokenizer):
        super(HybridDataset, self).__init__()

        self.samples_per_epoch = data_args.samples_per_epoch
        self.tasks = data_args.tasks.split("||")  # TODO: change to tasks?
        task_sample_rate = data_args.task_sample_rate
        s = sum(task_sample_rate)
        self.task_sample_rate = [float(x) / s for x in task_sample_rate]
        assert len(self.task_sample_rate) == len(self.tasks)

        ds_dict = {
            'dvc': {
                'activitynet': DVCDataset_activitynet,
                'youcook2': DVCDataset_youcook2,
                'vitt': DVCDataset_vitt,
                'howto100m': DVCDataset_howto100m,
            },
            'event_loc': {
                'activitynet': EventLocDataset_activitynet,
                'youcook2': EventLocDataset_youcook2,
                'vitt': EventLocDataset_vitt
            },
            'imgqa': {
                'llava': LlavaDataset,
            },
            'vidqa': {
                'msrvttqa': VidQADataset_msrvttqa,
                'msvdqa': VidQADataset_msvdqa,
                'nextqa': VidQADataset_nextqa,
                'videochat': VidQADataset_videochat,
            },
            'temporal_reasoning': {
                'activitynet': TemporalReasoningDataset_activitynet,
            },
            'epic_kitchen': {
                'narration_conversation': EpicConversationDataset
                # reasoning dataset is not defined here
            }
        }

        self.all_datasets = []
        self.all_sample_rate = []

        for task in self.tasks:
            task_data = getattr(data_args, task + '_data', '')
            datasets = []
            sample_counts = []
            if task != 'epic_kitchen':
                for data in task_data.split('||'):
                    dataset = ds_dict[task][data](data_path, tokenizer, data_args)
                    datasets.append(dataset)
                    sample_counts.append(len(dataset))
            else:
                print("in epic_kitchen, the sample rate is {}".format(data_args.epic_kitchen_sample_rate))
                image_processor: CLIPImageProcessor = data_args.image_processor
                epic_hoi_dataset_correct_narration: EpicHOIDataset = get_epic_hoi_dataset_by_name(ek_version=data_args.ek_version,
                                                                                                  split='train',
                                                                                                  image_processor=image_processor,
                                                                                                  rephrase_rate=data_args.ek_conversation_rephrase_rate,
                                                                                                  use_wrong_narration=False,
                                                                                                  use_percentage=data_args.epic_kitchen_use_percentage)
                if "narration_conversation" in task_data:
                    epic_conv_dataset = EpicMultiturnConversationDataset(tokenizer, epic_hoi_dataset_correct_narration)
                    datasets.append(epic_conv_dataset)
                    sample_counts.append(len(epic_conv_dataset))
                if "reasoning_conversation" in task_data:
                    epic_reasoning_conv_dataset = EpicReasoningConversationDataset(tokenizer, epic_hoi_dataset_correct_narration)
                    datasets.append(epic_reasoning_conv_dataset)
                    sample_counts.append(len(epic_reasoning_conv_dataset))

            sample_rate = getattr(data_args, task + '_sample_rate', sample_counts)
            assert len(sample_rate) == len(datasets), f"sample rate = {sample_rate}, datasets = {datasets}"
            s = sum(sample_rate)
            sample_rate = [float(x) / s for x in sample_rate]
            self.all_sample_rate.append(sample_rate)
            self.all_datasets.append(datasets)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.RandomState()  # local rng independent of global
        task = rng.choice(list(range(len(self.all_datasets))), p=self.task_sample_rate)
        dataset = rng.choice(list(range(len(self.all_datasets[task]))), p=self.all_sample_rate[task])
        return self.all_datasets[task][dataset][0]  # random sample idx inside so just input 0


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert instances[0]['input_ids'].shape == instances[0]['labels'].shape, (instances[0]['input_ids'].shape, instances[0]['labels'].shape)
        assert 'input_ids' in instances[0], 'input_ids not in instances'
        assert 'labels' in instances[0], 'labels not in instances'
        for instance in instances:
            device = instance['input_ids'].device
            dtype = torch.bfloat16

            if not instance.__contains__('feat'):
                instance['feat'] = torch.zeros(5, 10, 1024).to(device).to(dtype)
            if not instance.__contains__('bbox_feat'):
                instance['bbox_feat'] = torch.zeros(4, 10, 4).to(device).to(dtype)
            if not instance.__contains__('valid_mask'):
                instance['valid_mask'] = torch.zeros(5, 10).to(device).to(torch.bool)
            if not instance.__contains__('future_hands'):
                instance['future_hands'] = torch.zeros(2, 5, 2).to(device).to(dtype)
            if not instance.__contains__('contact_point'):
                instance['contact_point'] = torch.zeros(2).to(device).to(dtype)
            if not instance.__contains__('future_valid'):
                instance['future_valid'] = torch.zeros(2, ).to(device).to(torch.bool)
            if not instance.__contains__('gt_label_valid'):
                instance['gt_label_valid'] = torch.tensor(0).to(device).to(torch.bool)

            if type(instance['gt_label_valid']) == bool:
                instance['gt_label_valid'] = torch.tensor(instance['gt_label_valid']).to(device)
            if not instance.__contains__('prompt'):
                instance['prompt'] = ""
            if instance['image'].shape == torch.Size([3, 224, 224]):
                instance['image'] = instance['image'].unsqueeze(0).repeat(100, 1, 1, 1)

            assert instance['image'].shape == torch.Size([100, 3, 224, 224]), instance['image'].shape
            assert instance['feat'].shape == torch.Size([5, 10, 1024]), instance['feat'].shape
            assert instance['bbox_feat'].shape == torch.Size([4, 10, 4]), instance['bbox_feat'].shape
            assert instance['valid_mask'].shape == torch.Size([5, 10]), instance['valid_mask'].shape
            assert instance['future_hands'].shape == torch.Size([2, 5, 2]), instance['future_hands'].shape
            assert instance['contact_point'].shape == torch.Size([2]), instance['contact_point'].shape
            assert instance['future_valid'].shape == torch.Size([2, ]), instance['future_valid'].shape
            assert instance['gt_label_valid'].shape == torch.Size([]), instance['gt_label_valid'].shape
            valid_keys = ['feat', 'bbox_feat', 'valid_mask', 'future_hands', 'contact_point', 'future_valid', 'gt_label_valid', 'image']

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        prompt = [np.array(instance['prompt']) for instance in instances]  # https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset

        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), prompt=prompt)
        for key in valid_keys:
            feat = [instance[key] for instance in instances]
            if all(x is not None and x.shape == feat[0].shape for x in feat):
                batch[key] = torch.stack(feat)
            else:
                batch[key] = feat
        return batch
