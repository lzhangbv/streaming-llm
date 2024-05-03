from collections import defaultdict
import torch
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import detach_variable


__all__ = ["set_no_recompute_list", "checkpoint"]

def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x

class NoCheckpointHook(torch.autograd.graph.saved_tensors_hooks):
    # discard actications in the first forward pass
    def __init__(self):
        def pack(x):
            return None

        def unpack(x):
            raise AssertionError("Did not expect to unpack in first forward pass")

        super().__init__(pack, unpack)

class NoRecomputationPushMode(TorchDispatchMode):
    # save outputs of no-recompute operations in the first forward pass
    def __init__(self, storage, policy):
        self.storage = storage
        self.policy = policy
    
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        out = func(*args, **kwargs or {})
        if self.policy(func):
            detach_out = tree_map(_detach, out)
            self.storage.append(detach_out)
        return out

class NoRecomputationPopMode(TorchDispatchMode):
    # skip no-recompute operations in the recompute forward pass
    def __init__(self, storage, policy):
        self.storage = storage
        self.policy = policy

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if self.policy(func):
            out = self.storage.pop(0)
        else:
            out = func(*args, **kwargs or {})
        return out
    
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, policy, *args):
        ctx.run_function = run_function
        ctx.policy = policy
        ctx.storage = []
        
        # enable_grad: make sure that forward and recompute are consistent
        # NoRecomputationPushMode: save outputs of no-recompute operations
        # NoCheckpointHook: discard all activations in the first forward pass
        with torch.enable_grad(), NoRecomputationPushMode(ctx.storage, ctx.policy), NoCheckpointHook():
            outputs = run_function(*args)
        
        ctx.save_for_backward(*args)
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)
        
        # NoRecomputationPopMode: get outputs of no-recompute operations w/o runing them
        with torch.enable_grad(), NoRecomputationPopMode(ctx.storage, ctx.policy):
            outputs = ctx.run_function(*detached_inputs)
        
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads

# no_recompute_list is empty (a.k.a full checkpoint), it can be set via set_policy
_no_recompute_list = []
def _policy(func):
    return func in _no_recompute_list

def set_no_recompute_list(no_recompute_list):
    global _no_recompute_list
    _no_recompute_list = no_recompute_list

# custom checkpoint with selective recomputation
def checkpoint(function, *args):
    return CheckpointFunction.apply(function, _policy, *args)

