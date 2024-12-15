from hoi_forecast.dataset.annotation import get_ek55_annotation, get_ek100_annotation
from hoi_forecast.dataset.epic_structures import EpicHOIDataset, EpicHOIDatasetEval


def get_epic_hoi_dataset_by_name(ek_version, split, image_processor, rephrase_rate, use_wrong_narration, use_percentage) -> EpicHOIDataset:
    if ek_version == 'ek55':
        assert split in ['train', 'validation', 'eval', 'test_s1', 'test_s2']
        data_frame = get_ek55_annotation(split=split)
    elif ek_version == 'ek100':
        assert split in ['train', 'validation', 'eval', 'test']
        data_frame = get_ek100_annotation(split=split)
    else:
        raise ValueError(f'Error. EPIC-Kitchens Version "{ek_version}" not supported.')
    if split != 'eval':
        dataset = EpicHOIDataset(df=data_frame, split=split, ek_version=ek_version, image_processor=image_processor, rephrase_rate=rephrase_rate, use_wrong_narration=use_wrong_narration, use_percentage=use_percentage)
    else:
        dataset = EpicHOIDatasetEval(df=data_frame, split='eval', ek_version=ek_version, image_processor=image_processor, rephrase_rate=rephrase_rate, use_wrong_narration=use_wrong_narration)
    return dataset