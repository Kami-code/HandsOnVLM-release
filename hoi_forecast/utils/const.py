import os

const_dir_path = os.path.dirname(os.path.abspath(__file__))
HOI_PROJECT_PATH = os.path.dirname(const_dir_path)


EPIC_KITCHEN_DATASET_DIR = os.path.join(HOI_PROJECT_PATH, 'data/EPIC-KITCHENS')

ek55_annotation_dir = os.path.join(HOI_PROJECT_PATH, 'common/epic-kitchens-55-annotations')
ek100_annotation_dir = os.path.join(HOI_PROJECT_PATH, 'common/epic-kitchens-100-annotations')

ek55_rulstm_annotation_dir = os.path.join(HOI_PROJECT_PATH, "common/rulstm/RULSTM/data/ek55")
ek100_rulstm_annotation_dir = os.path.join(HOI_PROJECT_PATH, "common/rulstm/RULSTM/data/ek100")


def get_ru_lstm_annotation_dir(ek_version):
    if ek_version == 'ek55':
        return ek55_rulstm_annotation_dir
    elif ek_version == 'ek100':
        return ek100_rulstm_annotation_dir
    else:
        raise ValueError(f'Error. EPIC-Kitchens Version "{ek_version}" not supported.')


ek55_feature_dir = os.path.join(HOI_PROJECT_PATH, 'data/ek55/feats')
ek100_feature_dir = os.path.join(HOI_PROJECT_PATH, 'data/ek100/feats')

# generated data labels
ek55_label_dir = os.path.join(HOI_PROJECT_PATH, 'data/ek55')
ek100_label_dir = os.path.join(HOI_PROJECT_PATH, 'data/ek100')

ek55_label_path = os.path.join(HOI_PROJECT_PATH, "data/ek55/labels")
ek100_label_path = os.path.join(HOI_PROJECT_PATH, "data/ek100/labels")


def get_label_dir(ek_version):
    if ek_version == 'ek55':
        return ek55_label_path
    elif ek_version == 'ek100':
        return ek100_label_path
    else:
        raise ValueError(f'Error. EPIC-Kitchens Version "{ek_version}" not supported.')


# amazon-annotated eval labels
ek55_eval_label_path = os.path.join(HOI_PROJECT_PATH, 'data/ek55/ek55_eval_labels.pkl')
ek100_eval_label_path = os.path.join(HOI_PROJECT_PATH, 'data/ek100/ek100_eval_labels.pkl')


def get_eval_label_path(ek_version):
    if ek_version == 'ek55':
        return ek55_eval_label_path
    elif ek_version == 'ek100':
        return ek100_eval_label_path
    else:
        raise ValueError(f'Error. EPIC-Kitchens Version "{ek_version}" not supported.')


# pretrained backbone path
ek55_pretrained_backbone_path = os.path.join(HOI_PROJECT_PATH, 'common/rulstm/FEATEXT/models/ek55', 'TSN-rgb.pth.tar')
ek100_pretrained_backbone_path = os.path.join(HOI_PROJECT_PATH, 'common/rulstm/FEATEXT/models/ek100', 'TSN-rgb-ek100.pth.tar')
ek55_lmdb_path = os.path.join(HOI_PROJECT_PATH, "data/ek55/feats", "full_data_chenbao_processed.lmdb")
ek100_lmdb_path = os.path.join(HOI_PROJECT_PATH, "data/ek100/feats", "full_data_chenbao_processed.lmdb")

def get_lmdb_path(ek_version, mode):
    if ek_version == 'ek55':
        return ek55_lmdb_path
    elif ek_version == 'ek100':
        return ek100_lmdb_path
    else:
        raise ValueError(f'Error. EPIC-Kitchens Version "{ek_version}" not supported.')


observation_seconds = 2.5
fps = 4.0
origin_fps = 60.0
anticipation_seconds = 1.0
observation_frames_num = int(observation_seconds * fps)
anticipation_frames_num = int(anticipation_seconds * fps)
future_hand_num = anticipation_frames_num + 1
frame_template = 'frame_{:010d}.jpg'
num_actions_prev = 1
epic_img_shape = (456, 256)  # (width, height)
epic_img_width = epic_img_shape[0]
epic_img_height = epic_img_shape[1]
use_rulstm_splits = True
validation_ratio = 0.2
image_aspect_ratio = 'square'  # for clip processor
