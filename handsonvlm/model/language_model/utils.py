from transformers import AutoConfig, AutoModelForCausalLM
from handsonvlm.model.language_model.handsonvlm import HandsOnVLMConfig, HandsOnVLMForCausalLM


AutoConfig.register("handsonvlm", HandsOnVLMConfig)
AutoModelForCausalLM.register(HandsOnVLMConfig, HandsOnVLMForCausalLM)