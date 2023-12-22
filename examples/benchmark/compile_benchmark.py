"""
Accelerate LLM Inference with torch.compile
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast

import time
import math
import argparse
import types

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# prepare causal mask and past_key_values with setup_caches
causal_mask = None
past_key_values = None

def setup_caches(max_batch_size, config):
    num_layers = config.num_hidden_layers
    n_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    max_seq_length = config.max_position_embeddings
    dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16

    global causal_mask, past_key_values
    cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
    past_key_values = [
        (torch.zeros(cache_shape, dtype=dtype), torch.zeros(cache_shape, dtype=dtype)) for _ in range(num_layers)
    ]
    
    cond = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))
    causal_mask = torch.full((max_seq_length, max_seq_length), torch.finfo(dtype).min)
    causal_mask.masked_fill_(cond, 0)

def llama_attention_forward(
    self,
    hidden_states, 
    attention_mask,  
    position_ids, 
    past_key_value,  
    output_attentions,  
    use_cache
):
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
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

    kv_seq_len = self.max_position_embeddings # use static shape
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # update past key value: [bsz, n_heads, seq_length, head_dim]
        past_key_value[0][:, :, position_ids[0]] = key_states
        past_key_value[1][:, :, position_ids[0]] = value_states

        # use all KVs (future KVs will be masked out)
        key_states = past_key_value[0]
        value_states = past_key_value[1]

    past_key_value = (past_key_value[0], past_key_value[1]) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # todo: use F.scaled_dot_product_attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    # causal mask
    if causal_mask is not None: 
        attention_mask = causal_mask[None, None, position_ids[0]]
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_model_forward(
    self,
    input_ids, 
    attention_mask,  
    position_ids, 
    past_key_values,  
    inputs_embeds, 
    use_cache, 
    output_attentions, 
    output_hidden_states,
    return_dict,
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

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None


        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
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
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_llama_attention_forward(model):
    for name, module in model.named_modules():
        if "attn" in name and "attn." not in name:
            module.forward = types.MethodType(llama_attention_forward, module)
        if "model" in name and "model." not in name:
            module.forward = types.MethodType(llama_model_forward, module)

def prefill(model, input_ids, position_ids, past_key_values):
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # past_key_values = outputs.past_key_values
    idx_next = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    return idx_next

def decode_one_token(model, input_ids, position_ids, past_key_values):
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # past_key_values = outputs.past_key_values
    idx_next = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    return idx_next

@torch.no_grad()
def generate(model, input_ids, max_new_tokens):
    T = input_ids.shape[1]
    T_new = T + max_new_tokens

    device, dtype = input_ids.device, input_ids.dtype

    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty(T_new, dtype=dtype, device=device)
    seq[:T] = input_ids

    position_ids = torch.arange(0, T, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, T) #[1, T]

    next_token = prefill(model, input_ids, position_ids, past_key_values)
    seq[T] = next_token[0]

    position_ids = torch.arange(T, T+1, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, 1) #[1, 1]

    for i in range(1, max_new_tokens):
        next_token = decode_one_token(model, next_token, position_ids, past_key_values)
        seq[T+i] = next_token[0]
        position_ids += 1
    return seq

def main(args):
    print("Loading model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype="auto")
    model.eval()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt = "What is deep learning?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    print(f"Prompt length: {input_ids.shape[1]}")

    # replace attention
    replace_llama_attention_forward(model)

    # setup caches
    device, dtype = input_ids.device, input_ids.dtype
    with torch.device(device):
        setup_caches(max_batch_size=1, config=model.config)

    torch.manual_seed(1234)
    if args.compile:
        # torch.compile optimization
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
    
    tokens_per_sec = []
    start = -1

    for i in range(start, args.num_samples): 
        torch.cuda.synchronize()
        
        t0 = time.time()

        y = generate(
            model,
            input_ids,
            args.max_new_tokens,
        )

        if i < 0: 
            print(f"Compilation time: {time.time() - t0:.2f} seconds")
            continue

        torch.cuda.synchronize()
        t = time.time() - t0
        print(tokenizer.decode(y.tolist()))

        tokens_sec = args.max_new_tokens / t
        tokens_per_sec.append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(tokens_per_sec)).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_prefill", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()
    print(args)
    main(args)
