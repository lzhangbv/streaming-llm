from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.profiler import profile, ProfilerActivity

try: 
    from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen
    torch_selective_checkpoint = True  #torch-2.2
except:
    from selective_checkpoint import checkpoint as custom_checkpoint, set_no_recompute_list
    torch_selective_checkpoint = False

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.ReLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        # ingore rope for simplicity
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

# model
hidden_size = 4096

use_mlp = True
if use_mlp:
    model = nn.Sequential(MLP(hidden_size, hidden_size*4))
    check_fn = lambda submodule: isinstance(submodule, MLP)
else:
    model = nn.Sequential(Attention(hidden_size, num_heads=32))
    check_fn = lambda submodule: isinstance(submodule, Attention)

model.cuda()


"""
Selective Checkpoint: skip recomputing compute-heavy operations, such as mm and sdpa
for example, we can recompute relu (swiglu) for mlp, or skip sdpa for long-context attention.
"""
no_recompute_list = [
    torch.ops.aten.mm.default, 
    torch.ops.aten._scaled_dot_product_efficient_attention.default, 
    torch.ops.aten._scaled_dot_product_flash_attention.default,
]
if not use_mlp:
    no_recompute_list.pop(0)  # only skip sdpa

def get_custom_policy():
    def custom_policy(mode, func, *args, **kwargs):
        return func in no_recompute_list
    return custom_policy

def selective_checkpoint_context_fn():
    return _pt2_selective_checkpoint_context_fn_gen(get_custom_policy())


# wrap checkpoint: reentrant vs. non-reentrant
# non_reentrant recompute only necessary activation inputs, 
# for example, there is no need to recompute last gemm in mlp. 
selective_checkpoint = True
non_reentrant = True

if selective_checkpoint and torch_selective_checkpoint:
    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT, context_fn=selective_checkpoint_context_fn)
    print('non-reentrant selective checkpoint')
elif selective_checkpoint:
    set_no_recompute_list(no_recompute_list=no_recompute_list)
    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_fn=custom_checkpoint)
    print('custom selective checkpoint')
elif non_reentrant:
    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    print('non-reentrant full checkpoint')
else:
    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT)
    print('reentrant full checkpoint')

# fake data
batch_size = 1
seq_len = 16000
data = torch.randn(size=(batch_size, seq_len, hidden_size)).cuda()
data.requires_grad_()

def benchmark_step():
    output = model(data)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()

# reference w/o checkpoint
benchmark_step()
grad_ref = data.grad
data.grad = None

# wrap checkpoint
apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper_fn, check_fn=check_fn)
#print(model)

# warmup w. checkpoint
benchmark_step()
print(f'The maximum difference is {torch.max(torch.abs(grad_ref - data.grad))}')

# profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    benchmark_step()
prof.export_chrome_trace("checkpoint_benchmark.json")

