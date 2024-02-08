import math
from typing import Optional, Tuple
import types

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

TOPK = 2

def llama_topk_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # attention
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
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # topk: (bsz, num_heads, q_len, topk)
    attn_weights, selected_tokens = torch.topk(attn_weights, TOPK, dim=-1)
    attn_weights /= attn_weights.sum(dim=-1, keepdim=True)

    # select topk value states
    bsz_offsets = torch.arange(bsz, device=selected_tokens.device, dtype=selected_tokens.dtype)
    head_offsets = torch.arange(self.num_heads, device=selected_tokens.device, dtype=selected_tokens.dtype)
    offsets = bsz_offsets[:, None, None, None] * self.num_heads * kv_seq_len + head_offsets[None, :, None, None] * kv_seq_len
    selected_values = torch.index_select(
        value_states.reshape(-1, self.head_dim),
        dim=0,
        index=(selected_tokens + offsets).view(-1)
    ).view(bsz, self.num_heads, q_len, TOPK, self.head_dim)
    
    # weighted sum
    attn_output = selected_values * attn_weights.unsqueeze(-1)
    attn_output = attn_output.sum(dim=-2)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_topk_attention(model, topk=2):
    global TOPK
    TOPK = topk
    for name, module in model.named_modules():
        if "attn" in name and "attn." not in name:
            #print(name)
            module.forward = types.MethodType(llama_topk_attention_forward, module)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()
    enable_topk_attention(model, topk=16)

    # test
    #text = "What is deep learning?"
    text = "What is deep learning, and what is the major difference between deep leanring and machine learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
