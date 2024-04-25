import os
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def _custom_construct_wrap_fn(
    root_module: nn.Module,
    target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]],
    fsdp_fn: Callable,
) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    This constructs the "wrap" function to pass to :func:`_post_order_apply`
    based on ``target_module_to_kwargs``, which should be constructed from the
    wrapping policy.
    """
    def fn(module: nn.Module) -> Optional[nn.Module]:
        # Explicitly avoid wrapping the root module since for FSDP, it is
        # handled by the caller
        if module in target_module_to_kwargs and module is not root_module:
            kwargs = target_module_to_kwargs[module]
            #print('offload params:', module.is_offload)
            kwargs['cpu_offload'] = CPUOffload(offload_params=module.is_offload)
            return fsdp_fn(module, **kwargs)
        return None
    return fn


def _custom_wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    assert wrapper_cls is not None
    
    #print('offload params:', module.is_offload)
    kwargs['cpu_offload'] = CPUOffload(offload_params=module.is_offload)

    if hasattr(module, "_wrap_overrides"):
        overrides = {**kwargs, **module._wrap_overrides}
        return wrapper_cls(module, **overrides)
    return wrapper_cls(module, **kwargs)


def make_adaptive_offload(model, transformer_layer_cls, k=2):
    """Offload one transformer layer in every `k` layers."""
    wrap_num = 0
    for name, module in model.named_modules():
        if isinstance(module, transformer_layer_cls):
            module.is_offload = True if (k > 0) and (wrap_num % k == 0) else False
            wrap_num += 1

    # one of these two functions is used to wrap fsdp, see: fsdp/_wrap_utils/_auto_wrap
    # _wrap is used in our test case
    torch.distributed.fsdp.wrap._construct_wrap_fn = _custom_construct_wrap_fn
    torch.distributed.fsdp.wrap._wrap = _custom_wrap


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model_id = "lmsys/vicuna-7b-v1.3"

    # artifact config
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 32
    config.vocab_size = 1024
    config.hidden_size = 1024

    model = AutoModelForCausalLM.from_config(config)

    llama_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # setup offload before wrap fsdp
    offload_freq = 2
    make_adaptive_offload(model, LlamaDecoderLayer, k=offload_freq)

    model = FSDP(
        model, 
        auto_wrap_policy=llama_auto_wrap_policy, 
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0001, 
        #fused=True, # fused_adam on cpu is not supported yet
    )

    # fake data
    batch_size, seq_len = 1, 32
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

    if dist.get_rank() == 0:
        n_offload = (config.num_hidden_layers + offload_freq - 1) // offload_freq if offload_freq > 0 else 0
        #print(model)
        print(f"{n_offload} out of {config.num_hidden_layers} transformer layers are offloaded.")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

