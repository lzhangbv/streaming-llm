"""
Expand LLaMA model into UNet with skip connection.  
"""

import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache


class ULlamaDecoderLayer(nn.Module):
    def __init__(self, decoder_layer, skip=False):
        super().__init__()
        self.hidden_size = decoder_layer.hidden_size
        self.self_attn = decoder_layer.self_attn
        self.mlp = decoder_layer.mlp
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

        if skip:
            self.skip_w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.skip_w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.skip_w1.weight.data.copy_(torch.eye(self.hidden_size))
            self.skip_w2.weight.data.fill_(0.)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        skip_states: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Skip Connection
        if skip_states is not None:
            hidden_states = self.skip_w1(hidden_states) + self.skip_w2(skip_states)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def ullama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    skips = []

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if idx > self.num_layers // 2: # 17-31
            # out-blocks
            skip = skips.pop()
        else:
            skip = None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                skip,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                skip_states=skip,
            )

        hidden_states = layer_outputs[0]

        if idx < (self.num_layers - 1) // 2: # 0-14 
            # in_blocks
            skips.append(hidden_states)

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def add_skip_connection(model, num_layers=32):
    layers = nn.ModuleList()
    for i, layer in enumerate(model.model.layers):
        if i <= num_layers // 2: 
            new_layer = ULlamaDecoderLayer(layer, skip=False)
        else:
            new_layer = ULlamaDecoderLayer(layer, skip=True)
        layers.append(new_layer)
    
    assert num_layers - 1 == i, "the number of layers is not correct"

    model.model.layers = layers
    model.model.num_layers = num_layers
    model.model.forward = types.MethodType(ullama_model_forward, model.model)


if __name__ == "__main__": 
    """
    The implementation is tested with transformers=4.38.1, which has adopted SDPA and Dynamic-Cache. 
    """
    # llama
    model_id = "lmsys/vicuna-7b-v1.3"
    num_layers = 32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()

    # add skip connection
    add_skip_connection(model, num_layers)
    #print(model)

    # inference device and dtype
    model.to(device='cuda', dtype=torch.float16)

    # test
    text = "What is deep learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

