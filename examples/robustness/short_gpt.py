import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import types

"""
ShortGPT: Layers in Large Language Models are More Redundant Than You Expect

For a residual structure of y = x + f(x), it assumes that y is close to x. 
"""

def delete_decoder_layer(model, start, number): 
    layers = torch.nn.ModuleList()
    layer_idx = 0
    for i, layer in enumerate(model.model.layers):
        if i >= start and i < start + number:
            print(f"delete decoder in layer {i}")
            continue
        
        layer.self_attn.layer_idx = layer_idx
        layers.append(layer)
        layer_idx += 1

    model.model.layers = layers


def decoder_layer_forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position=None, **kwargs):
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
        #cache_position=cache_position, # if transformers < 4.38, it did not need cache_position
        **kwargs,
    )
    if self.skip_attn:
        hidden_states = residual
    else:
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    if self.skip_mlp:
        hidden_states = residual
    else:
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def delete_attn_layer(model, start, number):
    for i, layer in enumerate(model.model.layers): 
        if i >= start and i < start + number:
            skip_attn = True
            print(f"skip attn in layer {i}")
        else:
            skip_attn = False

        layer.skip_attn = skip_attn
        layer.skip_mlp = False
        layer.forward = types.MethodType(decoder_layer_forward, layer)


def delete_mlp_layer(model, start, number):
    for i, layer in enumerate(model.model.layers): 
        if i >= start and i < start + number:
            skip_mlp = True
            print(f"skip mlp in layer {i}")
        else:
            skip_mlp = False
            
        layer.skip_attn = False
        layer.skip_mlp = skip_mlp
        layer.forward = types.MethodType(decoder_layer_forward, layer)


if __name__ == "__main__":
    model_id = "lmsys/vicuna-7b-v1.3"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()

    number = 4
    start = 31 - number # delete layers 27, 28, 29, 30 (exclude last layer)
    
    # three options
    delete_decoder_layer(model, start, number)
    #delete_attn_layer(model, start, number)
    #delete_mlp_layer(model, start, number)

    model.to("cuda")

    # test
    text = "What is deep learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
