"""
Co-shard: Weight sharded but not distribtued

Reference:
    SuperScaler: Supporting Flexible DNN Parallelization via a Unified Abstraction

Note: it can reduce the peak memory in attention and mlp modules, which however may not be the bottleneck
"""

import types
import functools
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb


use_checkpoint = True
if use_checkpoint:
    checkpoint_func = functools.partial(checkpoint, use_reentrant=False, preserve_rng_state=False)
else:
    def no_checkpoint(func, *args):
        return func(*args)
    checkpoint_func = no_checkpoint


def mlp_forward(self, x):
    slice = self.intermediate_size // self.config.pretraining_tp
    gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
    up_proj_slices = self.up_proj.weight.split(slice, dim=0)
    down_proj_slices = self.down_proj.weight.split(slice, dim=1)

    def forward_function(x, i):
        gate_proj = F.linear(x, gate_proj_slices[i])
        up_proj = F.linear(x, up_proj_slices[i])
        intermediate_states = self.act_fn(gate_proj) * up_proj
        down_proj = F.linear(intermediate_states, down_proj_slices[i])
        return down_proj

    output = checkpoint_func(
        forward_function, 
        x, 
        0, 
    )
    for i in range(1, self.config.pretraining_tp):
        output += checkpoint_func(
            forward_function, 
            x, 
            i, 
        )
    
    return output


def attn_forward(self, hidden_states, position_ids, **kwargs):
    bsz, q_len, _ = hidden_states.size()
    num_heads = self.num_heads // self.config.pretraining_tp
    num_key_value_heads = self.num_key_value_heads // self.config.pretraining_tp

    key_value_slicing = num_key_value_heads * self.head_dim
    query_slices = self.q_proj.weight.split(num_heads * self.head_dim, dim=0)
    key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
    value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
    o_proj_slices = self.o_proj.weight.split(num_heads * self.head_dim, dim=1)

    def forward_function(hidden_states, position_ids, i):
        query_states = F.linear(hidden_states, query_slices[i])
        key_states = F.linear(hidden_states, key_slices[i])
        value_states = F.linear(hidden_states, value_slices[i])
        
        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, num_heads * self.head_dim)
        attn_output = F.linear(attn_output, o_proj_slices[i])

        return attn_output

    attn_output = checkpoint_func(
        forward_function, 
        hidden_states, 
        position_ids, 
        0, 
    )
    for i in range(1, self.config.pretraining_tp):
        attn_output += checkpoint_func(
            forward_function, 
            hidden_states, 
            position_ids, 
            i,
        )

    return attn_output, None, None


def make_shard(model, tp=2):
    for name, module in model.named_modules():
        if 'mlp' in name and 'mlp.' not in name:
            assert module.intermediate_size % tp == 0
            module.config.pretraining_tp = tp
            module.forward = types.MethodType(mlp_forward, module)

        if 'attn' in name and 'attn.' not in name:
            assert module.num_key_value_heads % tp == 0
            module.config.pretraining_tp = tp
            module.forward = types.MethodType(attn_forward, module)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoConfig

    model_id = "lmsys/vicuna-7b-v1.3"
    config = AutoConfig.from_pretrained(model_id)

    # config
    config.num_hidden_layers = 2
    config.vocab_size = 4096
    memory_snapshot = False

    # load model
    model = AutoModelForCausalLM.from_config(config)
    model.to(device='cuda', dtype=torch.float16)

    # coshard
    tp = 8
    make_shard(model, tp)

    # fake data
    batch_size, seq_len = 8, 4096
    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    # one step
    if memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    outputs = model(input_ids=x, labels=x)
    loss = outputs.loss
    loss.backward()

    if memory_snapshot:
        torch.cuda.memory._dump_snapshot("snapshot.pickle") # open it at https://pytorch.org/memory_viz
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Coshard tp size: {tp}")
    #print(f"Memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    print(f"Memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

