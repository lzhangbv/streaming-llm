"""
Accelerate LLM inference with torch.compile and support batching with right-side padding. 
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast

import time
import argparse
import types
import math

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def setup_llama_model(model, max_batch_size=1): 
    config = model.config
    n_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    max_seq_length = config.max_position_embeddings
    cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)

    for name, module in model.named_modules(): 
        if "attn" in name and "attn." not in name:
            # replace llama attention foward
            module.forward = types.MethodType(llama_attention_forward, module)
            # prepare past key value
            dtype = module.o_proj.weight.data.dtype
            device = module.o_proj.weight.data.device
            module.past_key_value = KVCache(cache_shape, dtype, device)

        if "model" in name and "model." not in name:
            # replace llama model forward
            module.forward = types.MethodType(llama_model_forward, module)
            # prepare causal mask
            dtype = module.embed_tokens.weight.data.dtype
            device = module.embed_tokens.weight.data.device
            print(f"token embed dtype: {dtype}, device: {device}")
            module.causal_mask = torch.tril(torch.ones((max_seq_length, max_seq_length), dtype=torch.bool, device=device))

class KVCache(nn.Module):
    def __init__(self, cache_shape, dtype, device):
        super().__init__()
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def prefill_update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

    def decode_update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        # tofix: it is not efficient to update each request's kv in loops
        for i in range(input_pos.shape[0]):
            pos = input_pos[i]
            k_out[i, :, pos] = k_val[i]
            v_out[i, :, pos] = v_val[i]
        return k_out, v_out

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

    if attention_mask is None:
        key_states, value_states = self.past_key_value.prefill_update(position_ids[0], key_states, value_states)
    else:
        key_states, value_states = self.past_key_value.decode_update(position_ids, key_states, value_states)
        
    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # SDPA
    if attention_mask is None:
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=True,
        )
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attention_mask, 
            dropout_p=0.0, 
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

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

    # prepare attention mask (q_len > 1 means prefill)
    q_len = position_ids.shape[-1]
    attention_mask = None if q_len > 1 else self.causal_mask[position_ids].unsqueeze(1)

    # past key values are not used here
    past_key_values = None

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
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def prefill(model, input_ids, position_ids):
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
    )
    return outputs

def decode_one_token(model, input_ids, position_ids):
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
    )
    return outputs

@torch.no_grad()
def generate(model, inputs, max_new_tokens):
    # prepare model inputs
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # prepare model outputs
    B, T = input_ids.shape[0], input_ids.shape[1]
    seq = input_ids.new_empty((B, T + max_new_tokens))
    seq[:, :T] = input_ids

    # prepare position ids
    position_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0) #[1, T]

    # prefill
    outputs = prefill(model, input_ids, position_ids)

    # next position ids
    position_ids = attention_mask.sum(dim=-1, keepdim=True).to(torch.long) #[B, 1]

    # prepare token ids 
    token_ids = torch.arange(0, B*T, T, dtype=torch.long, device=input_ids.device)
    token_ids = token_ids + position_ids.flatten() - 1

    # get next tokens
    logits = outputs.logits.view(-1, outputs.logits.shape[-1]) #[B*T, V]
    next_token = logits[token_ids].argmax(dim=-1).unsqueeze(1)
    seq[:, T] = next_token[:, -1]

    # decode
    for i in range(1, max_new_tokens): 
        outputs = decode_one_token(model, next_token, position_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        seq[:, T+i] = next_token[:, -1]
        position_ids += 1
    return seq

def main(args):
    # load model
    print("Loading model ...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(args.model_id)
    if args.max_position_embeddings is not None:
        config.max_position_embeddings = args.max_position_embeddings
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'right'

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        config=config,
        device_map="auto", 
        torch_dtype="auto",
        )
    model.eval()

    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    print('# of gpus: ', torch.cuda.device_count())
    #print('device map: ', model.hf_device_map)

    # prompts
    prompts = ["What is deep learning?", "Hello, what is your name?"]
    inputs = tokenizer(prompts, padding='longest', return_tensors="pt").to(model.device)
    print(f"Prompt length: {inputs.input_ids.shape[1]}")

    if args.max_batch_size is None:
        args.max_batch_size = len(prompts)
    assert len(prompts) == args.max_batch_size

    # setup model
    setup_llama_model(model, max_batch_size=args.max_batch_size)

    torch.manual_seed(1234)
    if args.compile:
        global decode_one_token, prefill
        fullgraph = not torch.cuda.device_count() > 1
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=fullgraph)

        if args.compile_prefill:
            prefill = torch.compile(prefill, dynamic=True, fullgraph=fullgraph)
    
    tokens_per_sec = []
    start = -1

    for i in range(start, args.num_samples): 
        torch.cuda.synchronize()
        t0 = time.time()

        y = generate(
            model,
            inputs, 
            args.max_new_tokens,
        )

        if i < 0: 
            print(f"Compilation time: {time.time() - t0:.2f} seconds")
            continue

        torch.cuda.synchronize()
        t = time.time() - t0
        print("\n")
        print(tokenizer.batch_decode(y, skip_special_tokens=True))

        tokens_sec = args.max_batch_size * args.max_new_tokens / t
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
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    args = parser.parse_args()
    print(args)
    main(args)
