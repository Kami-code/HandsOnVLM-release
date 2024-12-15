import json
import os
import random


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

TIME_TOKEN_TEMPLATE = "<t{t}>"
HAND_TOKEN_TEMPLATE = "<hand_traj>"


handsonvlm_utils_dir_path = os.path.dirname(os.path.abspath(__file__))
HANDSONVLM_PROJECT_PATH = os.path.dirname(handsonvlm_utils_dir_path)


# For Epic Conversation Dataset
general_question_templates = [
    "Can you provide the hand trajectory?",
    "What is the recommended hand movement?",
    "What is the future hand trajectory in this video?",
    "What is the predicted hand trajectory given current observations?"
]

action_question_templates = [
    "Where should my hand move to if I want to {}?",
    "Can you provide the hand trajectory for {}?",
    "What is the recommended hand movement for {}?",
]

general_specific_question_templates = [
    "What is the recommended hand trajectory for doing this?",
    "What is the predicted hand trajectory to do it?",
    "What is the future hand trajectory for doing it?",
    "Can you provide the hand trajectory for doing this action?"
]

action_prediction_templates = ["What kind of action do you think are going to happen in this video?",
                               "What is the predicted action in this video?",
                               "What is the expected action in this video?"]

action_answer_templates = [
    "Certainly! The hand trajectory for {} is as follows: {}.",
    "To {}, the recommended hand trajectory is: {}.",
]

general_trajectory_answer_templates = [
    "The hand trajectory for this action is as follows: {}.",
    "The possible following hand trajectory may be: {}.",
]

general_answer_templates = [
    "Sure! Here is the hand trajectory {}.",
    "Based on the video, the hand trajectory is as follows: {}.",
    "The predicted hand trajectory is as follows: {}."
]

ek_conversation_rephrase_dict_path = os.path.join(HANDSONVLM_PROJECT_PATH, "assets", "rephrase_ek100.json")
ek_conversation_rbhp_rephrase_dict_path = os.path.join(HANDSONVLM_PROJECT_PATH, "assets", "ek100_questions.json")
ek_conversation_rbhp_rephrase_dict_path_val = os.path.join(HANDSONVLM_PROJECT_PATH, "assets", "ek100_questions_val.json")

with open(ek_conversation_rephrase_dict_path, 'r') as file:
    ek_conversation_rephrase_dict = json.load(file)


def rephrease_narration(narration):
    if narration in ek_conversation_rephrase_dict:
        if isinstance(ek_conversation_rephrase_dict[narration], str):
            return random.choice([ek_conversation_rephrase_dict[ek_conversation_rephrase_dict[narration]]])
        elif isinstance(ek_conversation_rephrase_dict[narration], list):
            return random.choice(ek_conversation_rephrase_dict[narration])
        else:
            raise ValueError("rephrase_dict[narration] should be str or list")
    return narration