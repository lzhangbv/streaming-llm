import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import _functional_collectives as funcol

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def maybe_init_dist():
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()
        if world_size < 2:
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank

def _apply_tp_linear(linear: nn.Linear, style: str) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    # Linear's weight matrix is transposed: (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    # shard
    sharded_weight = shard(linear.weight, shard_dim)
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

def _apply_tp_ffn(mlp):
    assert hasattr(mlp, "gate_proj")
    assert hasattr(mlp, "up_proj")
    assert hasattr(mlp, "down_proj")

    _apply_tp_linear(mlp.gate_proj, "colwise")
    _apply_tp_linear(mlp.up_proj, "colwise")
    _apply_tp_linear(mlp.down_proj, "rowwise")

    def hook(_module, _input, output):
        dist.all_reduce(output)

    mlp.register_forward_hook(hook) #all-reduce output

def _apply_tp_attn(attn):
    assert hasattr(attn, "q_proj")
    assert hasattr(attn, "k_proj")
    assert hasattr(attn, "v_proj")
    assert hasattr(attn, "o_proj")

    _apply_tp_linear(attn.q_proj, "colwise")
    _apply_tp_linear(attn.k_proj, "colwise")
    _apply_tp_linear(attn.v_proj, "colwise")
    _apply_tp_linear(attn.o_proj, "rowwise")

    # overwrite
    world_size = _get_world_size()
    attn.num_heads = attn.num_heads // world_size
    attn.hidden_size = attn.hidden_size // world_size
    attn.num_key_value_heads = attn.num_key_value_heads // world_size
 
    def hook(_module, _input, output):
        attn_output, attn_weights, past_key_value = output
        dist.all_reduce(attn_output)

    attn.register_forward_hook(hook) #all-reduce attn_output

def apply_tp(model):
    for name, module in model.named_modules():
        if "attn" in name and "attn." not in name:
            _apply_tp_attn(module)
        if "mlp" in name and "mlp." not in name:
            _apply_tp_ffn(module)

if __name__ == "__main__":
    """
    Run tp using the following command:
        CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tensor_parallel.py
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # init
    rank = maybe_init_dist()
    use_tp = (rank is not None)
    global print
    if use_tp and rank != 0:
        print = lambda *args, **kwargs: None        
    # model
    model_id = "/mnt/data/llama2-7b-chat/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    # tp
    if use_tp:
        apply_tp(model)
    model = model.to(device="cuda")
    # test
    print("Testing ...")
    prompt = "What is deep learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
