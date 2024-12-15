# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM, PretrainedConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.language_model.llava_llama import LlavaLlamaModel
from lita.model.lita_arch import LitaMetaForCausalLM


class LitaConfig(LlamaConfig):
    model_type = "lita"


class LitaLlamaForCausalLM(LlamaForCausalLM, LitaMetaForCausalLM):
    config_class = LitaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        input_ids: (B, T) where T is the length of the input sequence, each element is the token id in the vocabulary
        labels: (B, T) the same as input_ids but we set IGNORE_INDEX to human sentence
        attention_mask: (B, T) where 0 is the padding token and 1 is the real token
        images: (B, 100, 3, 224, 224)
        """
        B, T = input_ids.shape
        # assert labels.shape == torch.Size([B, T]), labels.shape
        # assert attention_mask.shape == torch.Size([B, T]), attention_mask.shape
        # assert images.shape == torch.Size([B, 100, 3, 224, 224]), images.shape
        """
        Prepare the return variables for the model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # T_modified = attention_mask.shape[1]

        # assert input_ids is None
        # assert attention_mask.shape == torch.Size([B, T_modified]), attention_mask.shape
        # d = inputs_embeds.shape[-1]
        # assert inputs_embeds.shape == torch.Size([B, T_modified, d]), inputs_embeds.shape
        # assert labels.shape == torch.Size([B, T_modified]), labels.shape

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # assert hidden_states.shape == torch.Size([B, T_modified, d]), hidden_states.shape
        logits = self.lm_head(hidden_states)
        # assert logits.shape == torch.Size([B, T_modified, 32100]), logits.shape

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("lita", LitaConfig)
AutoModelForCausalLM.register(LitaConfig, LitaLlamaForCausalLM)


if __name__ == "__main__":
    # create a LITA instance and pass the fake video tensor to it
    lita_instance = LitaLlamaForCausalLM(config=PretrainedConfig())