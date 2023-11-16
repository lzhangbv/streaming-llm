import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

start_size = 1024
recent_size = 1024

def llama_kmeans_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

    # add cluster sizes
    value_states = torch.cat([value_states, torch.ones(bsz, self.num_key_value_heads, q_len, 1).to(value_states)], dim=-1) 

    # kv_seq_len for only cached tokens
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # abs_kv_seq_len for all tokens
    new_kv_seq_len = key_states.shape[-2]
    if past_key_value is None:
        self.abs_kv_seq_len = new_kv_seq_len
    else:
        self.abs_kv_seq_len += new_kv_seq_len
    
    # absolute position ids
    position_ids = torch.arange(self.abs_kv_seq_len - new_kv_seq_len, self.abs_kv_seq_len, dtype=torch.long, device=key_states.device)
    position_ids = position_ids.unsqueeze(0).view(-1, new_kv_seq_len)

    cos, sin = self.rotary_emb(value_states, seq_len=self.abs_kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states)

    count_states = value_states[:,:,:,-1].unsqueeze(-2).to(attn_weights)
    true_value_states = value_states[:,:,:,:-1]

    # if any empty cluster
    #count_states_sign = count_states > 0
    #attn_weights = attn_weights * count_states_sign

    # rescaling attention weights
    accum_attn_weights = attn_weights * count_states
    attn_weights = attn_weights/accum_attn_weights.sum(dim=-1, keepdim=True)

    attn_output = torch.matmul(attn_weights.to(true_value_states), true_value_states)
    attn_output = attn_output.to(hidden_states.dtype)

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

    # kv cache compression
    cache_size = start_size + recent_size
    bs, num_head, sq_len, _ = key_states.size()
    if sq_len >= cache_size: 
        past_key = key_states[:,:,:-recent_size,:]
        past_value = value_states[:,:,:-recent_size,:]
        recent_key = key_states[:,:,-recent_size:,:]
        recent_value = value_states[:,:,-recent_size:,:]
        
        # cluster past_kv into 1/gap size
        gap = 2
        key_kernel = past_key[:,:,0::gap, :]
        kernel_num = key_kernel.shape[2]

        for _ in range(5):
            y_norm = key_kernel.norm(dim=-1)
            y_norm = y_norm * y_norm
            dis =  y_norm.unsqueeze(2) - 2 * torch.matmul(past_key, key_kernel.transpose(2, 3))
            _, index = torch.min(dis, dim=-1) # [bs, head, seq], value in [0, kernel_num)

            index =  F.one_hot(index, kernel_num) # [bs, head, seq, past_seq/gap]
            index = index.transpose(-1, -2).to(past_key) # [bs, head, past_seq/gap, seq]
            key_kernel = torch.matmul(index, past_key)/(index.sum(dim=-1, keepdim=True)+0.001)

        past_key = key_kernel
        past_value = torch.matmul(index.to(past_value), past_value)

        key_states = torch.cat([past_key, recent_key], dim=2)
        value_states = torch.cat([past_value, recent_value], dim=2)

    value_states = value_states.to(torch.float32)
    past_key_value = (key_states, value_states) if use_cache else None    

    return attn_output, attn_weights, past_key_value


def enable_llama_kmeans_attention(model, start=1024, recent=1024):
    global start_size, recent_size
    start_size = start
    recent_size = recent
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_kmeans_attention(
                module, start, recent
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_kmeans_attention_forward, model._modules[name]
            )

