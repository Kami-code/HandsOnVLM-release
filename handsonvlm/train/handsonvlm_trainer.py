import math
import time
import copy
from typing import Dict, List, Optional

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import logging

from hoi_forecast.evaluation.traj_eval import evaluate_traj_stochastic

logger = logging.get_logger(__name__)

from handsonvlm.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.train.llava_trainer import LLaVATrainer


class HandsOnVLMTrainer(LLaVATrainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        ade_list = []
        fde_list = []
        wde_list = []
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            print("step: ", step)
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            conv = conv_templates['llava_v0'].copy()

            prompt = str(inputs['prompt'][0])
            # print("prompt: ", prompt)
            prompt = copy.deepcopy(prompt)
            # todo: suppot multi-turn conversation here
            # first message
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # print("device: ", device, "inputs: ", inputs.keys(), "batch_size: ", batch_size, "image: ", inputs["image"].shape, "labels: ", inputs["labels"].shape)
            temperature = 0.5
            top_p = 0.9
            num_beams = 1
            outputs = model.generate(
                input_ids,
                image=inputs["image"].bfloat16(),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=30,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True)

            # Prediction step
            # loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            pred_hand = outputs.pred_hands
            # todo: handle multi-turn conversation here / handle different shape predicted here
            assert pred_hand.shape == torch.Size([batch_size, 2, 4, 2]), pred_hand.shape
            pred_hands = self.accelerator.gather(pred_hand)
            # print("pred_hands: ", pred_hands.shape)
            B_new = pred_hands.shape[0]
            # print("B_new: ", B_new)
            assert pred_hands.shape == torch.Size([B_new, 2, 4, 2]), pred_hands.shape
            pred_hands = pred_hands.unsqueeze(1)
            assert pred_hands.shape == torch.Size([B_new, 1, 2, 4, 2]), pred_hands.shape
            gt_hands = self.accelerator.gather(inputs["gt_hands"])[..., 1:, :]
            # print("gt_hands: ", gt_hands.shape)
            assert gt_hands.shape == torch.Size([B_new, 2, 4, 2]), gt_hands.shape
            gt_hand_valids = self.accelerator.gather(inputs["gt_hand_valid"])[..., 1:]
            # print("gt_hand_valids: ", gt_hand_valids.shape)
            assert gt_hand_valids.shape == torch.Size([B_new, 2, 4]), gt_hand_valids.shape
            pred_hand_trajectory_list = [pred_hands.cpu().float().numpy()]
            gt_hand_trajectory_list = [gt_hands.cpu().float().numpy()]
            gt_hand_is_valid_list = [gt_hand_valids.cpu().float().numpy()]

            ade, fde, wde = evaluate_traj_stochastic(pred_hand_trajectory_list, gt_hand_trajectory_list, gt_hand_is_valid_list)
            # print("ade: ", ade, "fde: ", fde, "wde: ", wde)
            ade_list.append(ade)
            fde_list.append(fde)
            wde_list.append(wde)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        # if losses_host is not None:
        #     losses = nested_numpify(losses_host)
        #     all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            preds = nested_numpify(preds_host)
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
        # if inputs_host is not None:
        #     inputs_decode = nested_numpify(inputs_host)
        #     all_inputs = (
        #         inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
        #     )
        # if labels_host is not None:
        #     labels = nested_numpify(labels_host)
        #     all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        metrics = {f"{metric_key_prefix}/traj_ade": np.mean(ade_list), f"{metric_key_prefix}/traj_fde": np.mean(fde_list), f"{metric_key_prefix}/traj_wde": np.mean(wde_list)}

        if wandb.run is not None:  # only log to the main process
            print(metrics)
            wandb.log(metrics, commit=False)
        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=None, metrics=metrics, num_samples=num_samples)