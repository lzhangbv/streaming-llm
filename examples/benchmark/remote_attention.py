"""
Experimental: Remote Attention for Long-context LLM Inference
1. Local attention: model_device="cuda:0" and kv_cache_device="cuda:0"
2. Remote attention: model_device="cuda:0" and kv_cache_device="cuda:1"
3. HF-style pipeline parallelism: model_device="auto" and kv_cache_device="auto"
"""

import types
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast

def set_remote_attention(model, max_seq_length=2048, device="cuda:0", dtype=torch.float16):
    config = model.config
    n_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    # it only supports bs=1
    cache_shape = (1, n_heads, max_seq_length, head_dim)

    for name, module in model.named_modules(): 
        if "attn" in name and "attn." not in name:
            # replace llama attention foward
            module.forward = types.MethodType(llama_attention_forward, module)
            # prepare static kv cache
            kv_device =  module.o_proj.weight.data.device if device == "auto" else device
            module.k_cache = torch.zeros(cache_shape, dtype=dtype, device=kv_device)
            module.v_cache = torch.zeros(cache_shape, dtype=dtype, device=kv_device)
            module.max_position_embeddings = max_seq_length
    
        if "model" in name and "model." not in name:
            # replace llama model forward
            module.forward = types.MethodType(llama_model_forward, module)
            module.kv_length = 0
    
def llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,  
    output_attentions=False,  
    use_cache=False,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = self.max_position_embeddings # static shape
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # update remote kv cache
    if q_len > 1:
        # prefill
        self.k_cache[:, :, :q_len] = key_states
        self.v_cache[:, :, :q_len] = value_states
        # todo: compute prefill attention locally and overlap with kv cache transfer
        key_states = self.k_cache[:, :, :q_len]
        value_states = self.v_cache[:, :, :q_len]
    else:
        # decode
        pos_id = position_ids[0].item()
        self.k_cache[:, :, pos_id:pos_id+1] = key_states
        self.v_cache[:, :, pos_id:pos_id+1] = value_states
        key_states = self.k_cache[:, :, :pos_id+1]
        value_states = self.v_cache[:, :, :pos_id+1]

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # SDPA: avoid explicit attention mask
    if q_len > 1:
        # prefill
        attn_output = F.scaled_dot_product_attention(
            query_states.to(key_states.device), 
            key_states, 
            value_states, 
            attn_mask=None, 
            dropout_p=0.0,
            is_causal=True,
        )
    else:
        # decode
        attn_output = F.scaled_dot_product_attention(
            query_states.to(key_states.device), 
            key_states, 
            value_states, 
            attn_mask=None, 
            dropout_p=0.0,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = attn_output.to(hidden_states.device)

    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value

def llama_model_forward(
    self,
    input_ids=None, 
    attention_mask=None,  
    position_ids=None, 
    past_key_values=None, 
    inputs_embeds=None, 
    use_cache=None, 
    output_attentions=None, 
    output_hidden_states=None, 
    return_dict=None, 
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # position ids
    if position_ids is None:
        q_len = inputs_embeds.shape[1]
        if q_len > 1:
            # prefill
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device)
            self.kv_length = q_len
        else:
            # decode
            position_ids = torch.arange(self.kv_length, self.kv_length+q_len, dtype=torch.long, device=hidden_states.device)
            self.kv_length += q_len
        position_ids = position_ids.unsqueeze(0)

    # past key values and attention mask
    past_key_values = None
    attention_mask = None

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values, 
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    # get last token's hidden state to reduce lm_head computation and memory overheads
    hidden_states = hidden_states[:, -1, :].unsqueeze(1)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def test(args):
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load model
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        device_map=args.model_device,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # setup remote attention
    set_remote_attention(model, max_seq_length=args.max_seq_length, device=args.kv_cache_device)

    # prompt
    prompt = "What is deep learning?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    outputs = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--model_device", type=str, default="cuda:0")
    parser.add_argument("--kv_cache_device", type=str, default="cuda:1")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    print(args)
    test(args)
