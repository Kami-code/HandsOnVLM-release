import os
import argparse

from handsonvlm.evaluation.handsonvlm_inference import HandsOnVLMInference
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--visual-path", type=str, required=True)
    parser.add_argument("--output-video-path", type=str, default='./output.mp4')
    args = parser.parse_args()

    handsonvlm_inference = HandsOnVLMInference(model_path=args.model_path,
                                               model_base=None,
                                               load_8bit=args.load_8bit,
                                               load_4bit=args.load_4bit,
                                               conv_mode=args.conv_mode)
    while True:
        handsonvlm_inference.user_input_inference(path=args.visual_path, output_video_path=args.output_video_path)
