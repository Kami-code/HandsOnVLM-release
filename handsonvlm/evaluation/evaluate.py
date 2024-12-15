import argparse
import os

from handsonvlm.evaluation.handsonvlm_inference import HandsOnVLMInference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use_reason", action="store_true")
    parser.add_argument("--mode", type=str, default="general")
    args = parser.parse_args()

    checkpoint_abs_path = args.model_path

    handsonvlm_inference = HandsOnVLMInference(model_path=checkpoint_abs_path,
                                               model_base=None,
                                               load_8bit=args.load_8bit,
                                               load_4bit=args.load_4bit,
                                               conv_mode=args.conv_mode)

    traj_info = handsonvlm_inference.evaluate_epic_kitchen_traj(test_version='ek100',
                                                                split='validation',
                                                                use_reason=args.use_reason)