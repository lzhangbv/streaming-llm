import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, lambda_auto_wrap_policy, _or_policy
from torch.profiler import profile, ProfilerActivity

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


"""
FSDP for Base Model, and DDP for LoRA Adaptors. 
"""

class LinearLoRA(nn.Module):
    def __init__(self, linear, r, alpha, sync_lora=True):
        super().__init__()

        self.base = linear
        self.r = r
        self.alpha = alpha
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        device, dtype = self.base.weight.device, self.base.weight.dtype
        self.w_a = nn.Linear(self.in_features, r, bias=False, device=device, dtype=dtype)
        self.w_b = nn.Linear(r, self.out_features, bias=False, device=device, dtype=dtype)

        # sync params among ddp workers
        if sync_lora:
            dist.broadcast(self.w_a.weight.data, src=0)
        self.w_b.weight.data.fill_(0.)
    
    def forward(self, hidden_states):
        return self.base(hidden_states) + self.alpha * self.w_b(self.w_a(hidden_states))


def naive_fsdp_lora(model, r=32, alpha=1):
    # freeze model
    for name, p in model.named_parameters():
        p.requires_grad = False

    # add lora adaptor
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue

        linear_lora = LinearLoRA(m, r, alpha, sync_lora=False)
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)
        setattr(parent, name[len(parent_name) + 1:], linear_lora)
    
    # wrap fsdp
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])

    model = FSDP(
        model, 
        auto_wrap_policy=auto_wrap_policy, 
        device_id=torch.cuda.current_device(),
    )

    return model


def fsdp_lora(model, r=32, alpha=1):
    # freeze model
    for name, p in model.named_parameters():
        p.requires_grad = False

    # wrap fsdp
    llama_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    model = FSDP(
        model, 
        auto_wrap_policy=llama_auto_wrap_policy, 
        device_id=torch.cuda.current_device(),
    )
    
    # add lora adaptor
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue

        linear_lora = LinearLoRA(m, r, alpha)
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)
        setattr(parent, name[len(parent_name) + 1:], linear_lora)
    
    # wrap ddp
    model = DDP(
        model, 
        device_ids=[torch.cuda.current_device()], 
        broadcast_buffers=False, # avoid broadcast frozen params
        # bucket_cap_mb=25, 
    )
    
    return model
    

if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model_id = "lmsys/vicuna-7b-v1.3"

    # artifact config
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 8
    config.vocab_size = 1024
    config.hidden_size = 1024

    model = AutoModelForCausalLM.from_config(config)

    # wrap fsdp and lora
    # 1) naive wrap
    #model = naive_fsdp_lora(model, r=32, alpha=1)

    # 2) optimized wrap
    model = fsdp_lora(model, r=32, alpha=1)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0001,
        fused=True,
    )

    # fake data
    batch_size, seq_len = 16, 128
    data = torch.randint(low=0, high=1000, size=(batch_size, seq_len + 1))
    target = data[:, 1:].cuda()
    data = data[:, 0:seq_len].cuda()

    def one_step():
        optimizer.zero_grad()
        outputs = model(data)
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        loss = F.cross_entropy(logits, target.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # warmup
    one_step()

    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        one_step()

    if dist.get_rank() == 0:
        #print(model)

        print("num of param groups:", len(optimizer.param_groups))
        #print("keys in first group:", optimizer.param_groups[0].keys())
        print("num of params in first group:", len(optimizer.param_groups[0]['params']))

        # each FSDP module has one frozen flatten param; every lora weight is trainable
        frozen_params = []
        trainable_params = []
        for param in optimizer.param_groups[0]['params']:
            if param.requires_grad:
                trainable_params.append(param)
            else:
                frozen_params.append(param)
        print(f"num of frozen params: {len(frozen_params)}, num of trainable params: {len(trainable_params)}")

        prof.export_chrome_trace("fsdp_lora.json")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

