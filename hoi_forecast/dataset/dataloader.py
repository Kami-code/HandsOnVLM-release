from torch.utils.data import DataLoader, default_collate

from hoi_forecast.dataset.dataset import get_epic_hoi_dataset_by_name
from hoi_forecast.dataset.epic_structures import EpicHOIDataset


def action_removing_collate(batch):
    # this collate function is used to handle the case where the dataset returns a tuple (sample, action), action is not able to be batched.
    samples = []
    for item in batch:
        sample, action = item
        samples.append(sample)
    return default_collate(samples)


def get_epic_hoi_dataloader_by_name(ek_version, split, batch_size, num_workers, image_processor=None, rephrase_rate=0, use_wrong_narration=False, use_percentage=1.0):
    epic_hoi_dataset: EpicHOIDataset = get_epic_hoi_dataset_by_name(ek_version, split, image_processor=image_processor, rephrase_rate=rephrase_rate, use_wrong_narration=use_wrong_narration, use_percentage=use_percentage)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    epic_hoi_dataloader = DataLoader(epic_hoi_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, drop_last=False)
    epic_hoi_dataloader.collate_fn = action_removing_collate
    return epic_hoi_dataloader


def get_epic_conversation_dataloader_by_name(ek_version, split, batch_size, num_workers, tokenizer, image_processor=None, rephrase_rate=0, use_wrong_narration=False, use_percentage=1.0, shuffle=None):
    epic_hoi_dataset: EpicHOIDataset = get_epic_hoi_dataset_by_name(ek_version, split, image_processor=image_processor, rephrase_rate=rephrase_rate, use_wrong_narration=use_wrong_narration, use_percentage=use_percentage)
    if shuffle is None:
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

    from handsonvlm.dataset.epic_dataset import EpicConversationDataset
    epic_conversation_dataset = EpicConversationDataset(tokenizer=tokenizer, epic_hoi_dataset=epic_hoi_dataset)
    epic_conversation_dataloader = DataLoader(epic_conversation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, drop_last=False)
    return epic_conversation_dataloader