"""
Delayed weight gradient computation with torch-ddp. 

Refer to: Zero-bubble Pipeline Parallelism
"""

# Important: 
# it is non-trivial to support delayed weight gradient computation with torch-ddp;
# because delayed weight gradient (outside backward pass) will change hook behaviours;
# we design grad_sync and non_grad_sync modes to support multi-forward-backward cases in ddp;
# we ensure that ddp's gradient reduction only happens at the last backward when they are ready;
# however, our solutions are too complex, we recommand to reimplement ddp for sake of simplicity.

import os
import queue
import types
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity


class WeightGradStore: 
    cache = []                         # cache activations of one linear layer in the backward
    weight_grad_queue = queue.Queue()  # put cached activations of all linear layers in one backward
    weight_queue = []                  # save one weight temporarily
    grad_sync = True                   # grad_sync or non_grad_sync mode

    @classmethod
    def set_grad_sync(cls, grad_sync):
        # It should be called before each forward pass
        cls.grad_sync = grad_sync

    @classmethod
    def get_grad_sync(cls):
        return cls.grad_sync

    @classmethod
    def cache_weight(cls, weight):
        assert len(cls.weight_queue) == 0
        cls.weight_queue.append(weight)

    @classmethod
    def pop_weight(cls):
        assert len(cls.weight_queue) == 1
        return cls.weight_queue.pop()

    @classmethod
    def put(cls, input, grad_output, weight, grad_sync):
        # Store activations of one linear layer in the backward
        cls.cache.append((input, grad_output, weight, grad_sync))

    @classmethod
    def flush(cls):
        # Store activations of all linear layers in one backward
        # It should be called after each backward pass
        if len(cls.cache) == 0:
            return
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        # Compute weight gradients of all linear layers cached in one backward
        @torch.no_grad()
        def compute_grad_weight(input, grad_output):
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = grad_output.t().matmul(input)
            return grad_weight

        stored_grads= cls.weight_grad_queue.get()

        for input, grad_output, weight, sync in stored_grads:
            grad_weight = compute_grad_weight(input, grad_output)

            if sync: 
                # In grad_sync mode, we use weight_temp to trigger weight's grad_acc hook; 
                with torch.enable_grad():
                    weight_temp = weight.expand_as(weight)
                weight_temp.backward(grad_weight)
            else:
                # In non_grad_sync mode, we add grad_weight to weight.grad under no_grad context; 
                with torch.no_grad():
                    if weight.grad is None:
                        weight.grad = grad_weight
                    else:
                        weight.grad.add_(grad_weight)
    
    @classmethod
    def pop_all(cls):
        # Execute all remaining weight gradient computations
        remaining_qsize = cls.weight_grad_queue.qsize()
        for i in range(remaining_qsize):
            cls.pop()


class LinearWithDelayedWeightGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
    ):
        grad_sync = WeightGradStore.get_grad_sync()

        if grad_sync:
            # pop weight from the cache
            weight = WeightGradStore.pop_weight()
        
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.grad_sync = grad_sync

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias

        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_sync = ctx.grad_sync

        grad_input = grad_output.matmul(weight)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        WeightGradStore.put(input, grad_output, weight, grad_sync)

        # In non_grad_sync mode, if we return grad_weight=None to the weight,
        # 1) it will not clean gradient value to None;
        # 2) but it will still trigger its grad hook in ddp. 
        # Please make sure that ddp is under no_sync to avoid incorrect gradient reduction.
        return grad_input, None, grad_bias


def linear_with_delayed_weight_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return LinearWithDelayedWeightGrad.apply(input, weight, bias)


def linear_forward(self, input):
    # To handle multi-forward-backward, we design two forward modes: grad_sync and non_grad_sync; 
    # 1) in grad_sync mode, we trigger its grad hook after computing weight_grad for ddp communication; 
    # 2) in non_grad_sync mode, we trigger its grad hook for ddp's param check during back-propagation;  
    grad_sync = WeightGradStore.get_grad_sync()

    # grad_sync mode
    if grad_sync:
        # we don't want to trigger grad hook during back-propagation, as weight grad is delayed and not ready;
        # if we feed weight to linear function, it will return grad_weight and trigger grad_hook at that moment;
        # to avoid that, we cache weight here and pop it inside linear function, and feed None as a placeholder.
        WeightGradStore.cache_weight(self.weight)
        _weight = None
        return linear_with_delayed_weight_grad(input, _weight, self.bias)
    
    # non_grad_sync mode
    else:
        # for each backward, ddp checks whether all grads are ready even with no_sync; 
        # in two consecutive backward passes, if we skip some grad hooks in the first pass, 
        # the second pass will raise an error, since other grad hooks are fired twice by ddp. 
        # to avoid that, we feed weight to linear function, and return None at backward to fire its hook.
        return linear_with_delayed_weight_grad(input, self.weight, self.bias)


def replace_linear_layers(model):
    # replace linear layers to support delayed weight gradient computation
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(
                linear_forward, 
                module, 
            )


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModelForCausalLM

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
    model.cuda()
    
    use_ddp = True
    if use_ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)
        module = model.module
    else:
        # for debug
        module = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # fake data
    batch_size, seq_len = 1, 256
    num_batches = 8
    data = torch.randint(low=0, high=1000, size=(batch_size * num_batches, seq_len + 1))
    target = data[:, 1:].cuda()
    data = data[:, 0:seq_len].cuda()

    def one_forward(data, target):
        # forward pass
        outputs = model(data)
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        loss = F.cross_entropy(logits, target.view(-1))
        return loss

    def one_backward(loss):
        # backward pass with delayed weight grad
        loss.backward()
        # cache activations of all linear layers in one backward
        WeightGradStore.flush()

    # tests
    def test1():
        # simple forward-backward
        # 1) Baseline
        optimizer.zero_grad()
        one_backward(one_forward(data, target))
        grad_ref = module.model.layers[0].self_attn.q_proj.weight.grad.clone()

        # 2) Delayed Weight Grad
        replace_linear_layers(model)

        optimizer.zero_grad()
        one_backward(one_forward(data, target))
        WeightGradStore.pop()
        
        if dist.get_rank() == 0:
            print(torch.max(torch.abs(module.model.layers[0].self_attn.q_proj.weight.grad - grad_ref)))
    
    def test2():
        # one-forward one-backward with micro-batches

        # 1) Baseline: 1F1B
        optimizer.zero_grad()
        for i in range(num_batches):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            # grad-sync for last micro-batch
            model.require_backward_grad_sync = (i == num_batches - 1)
            # one-forward one-backward
            one_backward(one_forward(micro_data, micro_target))
        grad_ref = module.model.layers[0].self_attn.q_proj.weight.grad.clone()

        # 2) Delayed Weight Grad
        replace_linear_layers(model)

        optimizer.zero_grad()
        for i in range(num_batches):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            # grad-sync for last micro-batch
            is_last = (i == num_batches - 1)
            WeightGradStore.set_grad_sync(is_last)
            model.require_backward_grad_sync = is_last 
            one_backward(one_forward(micro_data, micro_target))
            WeightGradStore.pop()
        WeightGradStore.pop_all()
        if dist.get_rank() == 0:
            print(torch.max(torch.abs(module.model.layers[0].self_attn.q_proj.weight.grad - grad_ref)))
        
    def test3():
        # all-forward all-backward with no_sync_grad, plus one-forward-one-backward with sync_grad

        # 1) Baseline: 1F1B
        optimizer.zero_grad()
        for i in range(num_batches):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            # grad-sync for last micro-batch
            model.require_backward_grad_sync = (i == num_batches - 1)
            # one-forward one-backward
            one_backward(one_forward(micro_data, micro_target))
        grad_ref = module.model.layers[0].self_attn.q_proj.weight.grad.clone()

        # 2) Delayed Weight Grad
        replace_linear_layers(model)

        optimizer.zero_grad()
        loss_list = []
        # no grad_sync for (n-1) micro-batches
        WeightGradStore.set_grad_sync(False)
        model.require_backward_grad_sync = False
        for i in range(num_batches-1):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            loss_list.append(one_forward(micro_data, micro_target))
        for i in range(num_batches-1):
            one_backward(loss_list.pop(0))
            WeightGradStore.pop()

        # grad_sync for last micro-batches
        WeightGradStore.set_grad_sync(True)
        model.require_backward_grad_sync = True
        i = num_batches - 1
        micro_data = data[i * batch_size: (i+1) * batch_size, :]
        micro_target = target[i * batch_size: (i+1) * batch_size, :]
        one_backward(one_forward(micro_data, micro_target))
        WeightGradStore.pop_all()

        if dist.get_rank() == 0:
            print(torch.max(torch.abs(module.model.layers[0].self_attn.q_proj.weight.grad - grad_ref)))

    def test4():
        # all-forward all-backward with micro-batches

        # 1) Baseline: 1F1B
        optimizer.zero_grad()
        for i in range(num_batches):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            # grad-sync for last micro-batch
            model.require_backward_grad_sync = (i == num_batches - 1)
            # one-forward one-backward
            one_backward(one_forward(micro_data, micro_target))
        grad_ref = module.model.layers[0].self_attn.q_proj.weight.grad.clone()

        # 2) Delayed Weight Grad
        replace_linear_layers(model)

        optimizer.zero_grad()
        loss_list = []
        # set require_backward_grad_sync = False for all forwards
        # if we set it True for last forward, the next backward (which is the first backward) will sync grads
        # but we want to sync grads at last backward, so we choose to manually enable it before last backward
        model.require_backward_grad_sync = False
        for i in range(num_batches):
            micro_data = data[i * batch_size: (i+1) * batch_size, :]
            micro_target = target[i * batch_size: (i+1) * batch_size, :]
            # no grad_sync for (n-1) micro-batches
            WeightGradStore.set_grad_sync((i == num_batches - 1))
            loss_list.append(one_forward(micro_data, micro_target))
        for i in range(num_batches):
            if i == num_batches - 1:
                # trigger sync grads in the last backward
                model.reducer.prepare_for_backward([])
            one_backward(loss_list.pop(0))
            WeightGradStore.pop()
        WeightGradStore.pop_all()

        if dist.get_rank() == 0:
            print(torch.max(torch.abs(module.model.layers[0].self_attn.q_proj.weight.grad - grad_ref)))

    def test5(case=0):
        # dive into ddp's grad_sync mechanism
        # if grad_sync=True, its post_forward will call prepare_for_backward
        # in the next backward, reducer will reduce gradients
       
        micro_data = data[0:batch_size, :]
        micro_target = target[0:batch_size, :]

        if case == 0:
            # case 0: gradient reduction is wrong if we change require_backward_grad_sync
            optimizer.zero_grad()
            model.require_backward_grad_sync = False
            loss1 = one_forward(micro_data, micro_target) # no prepare_for_backward
            model.require_backward_grad_sync = True
            loss2 = one_forward(micro_data, micro_target) # prepare_for_backward
            one_backward(loss1) # we want no_grad here, but it does happen
            one_backward(loss2) # we want sync_grad here, but it does not happen

        else:
            # case 1: gradient reduction is right if we call pre_for_backward by ourself
            optimizer.zero_grad()
            model.require_backward_grad_sync = False
            loss1 = one_forward(micro_data, micro_target) # no prepare_for_backward
            loss2 = one_forward(micro_data, micro_target) # no prepare_for_backward
            one_backward(loss1) # no sync grad
            model.reducer.prepare_for_backward([]) # prepare_for_forward, where unused_params are None
            one_backward(loss2) # sync grad here

    # warmup
    one_backward(one_forward(data, target))
    optimizer.zero_grad()

    # please choose one test function at each run
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        test1()
        #test2()
        #test3()
        #test4()
        #test5(case=0)
        #test5(case=1)

    if dist.get_rank() == 0:
        prof.export_chrome_trace("trace.json")

