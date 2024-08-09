import torch
from torch.utils._python_dispatch import TorchDispatchMode

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Block
from transformers.activations import ACT2FN


activation_dict = dict()

debug = False

class TrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        res = func(*args, **kwargs or {})
        if debug:
            print('operation:', func, ', allocated memory:', torch.cuda.memory_allocated())
        return res


class ActivationMode(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self):
        def pack(x):
            if not isinstance(x, torch.nn.Parameter):
                x_ = x.untyped_storage()
                name = str(x_.data_ptr())
                
                # note: x.untyped_storage().size() sometimes is not equal to x.element_size()*x.numel()
                # if a large tensor's slice is saved, it has small logical size but large physical size!
                if name not in activation_dict:
                    if debug:
                        print('saved activation:', x.dtype, x.shape, x.element_size()*x.numel(), x_.size(), name)
                    activation_dict[name] = x_.size()
                else:
                    #if debug:
                    #    print('duplicated activation:', x.dtype, x.shape, x.element_size()*x.numel(), x_.size())
                    pass

            return x

        def unpack(x):
            return x
        super().__init__(pack, unpack)


def get_config(name='gpt2', gelu_new=True):
    activation_function = 'gelu_new' if gelu_new else 'gelu_pytorch_tanh'

    if name == 'gpt2': #124M
        return GPT2Config(n_embd=768, n_head=12, n_layer=12, activation_function=activation_function)
    elif name == 'gpt2-medium': #355M
        return GPT2Config(n_embd=1024, n_head=16, n_layer=24, activation_function=activation_function)
    elif name == 'gpt2-large': #774M
        return GPT2Config(n_embd=1280, n_head=20, n_layer=36, activation_function=activation_function)
    elif name == 'gpt2-xl': #1.5B
        return GPT2Config(n_embd=1600, n_head=25, n_layer=48, activation_function=activation_function)


def count_parameter(model):
    print('tied embedding:', model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr())
    count = 0
    for name, param in model.named_parameters():
        count += param.numel()
        # if len(param.shape) > 1:
        #     print(name, param.shape)
    return count


def profile_model(name='gpt2', mode='autocast', peak_memory=True):
    config = get_config(name)
    model = GPT2LMHeadModel(config)
    print("Model:", name)

    batch_size, seq_len = 1, 1024
    config.batch_size, config.sequence_length = batch_size, seq_len

    assert mode in ['fp32', 'fp16', 'autocast']
    dtype = torch.float16 if mode == 'fp16' else torch.float32
    mixed_precision = (mode == 'autocast')
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Dtype: {dtype}, Mixed-precision: {mixed_precision}")

    mem_0 = torch.cuda.memory_allocated()
    assert mem_0 == 0

    model.to(device='cuda', dtype=dtype)
    mem_1 = torch.cuda.memory_allocated()
    print("Num of parameters:", count_parameter(model))

    # add buffer tensors for deduplication
    global activation_dict
    for name, buffer in model.named_buffers():
        name = str(buffer.untyped_storage().data_ptr())
        activation_dict[name] = 0
    # add lm head for deduplication
    name = str(model.lm_head.weight.untyped_storage().data_ptr())
    activation_dict[name] = 0

    optimizer = torch.optim.AdamW(model.parameters())
    assert torch.cuda.memory_allocated() == mem_1

    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    if mixed_precision:
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            with TrackingMode(), ActivationMode():
                outputs = model(input_ids=x, labels=x)
                loss = outputs.loss
    else:
        with TrackingMode(), ActivationMode():
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
    
    del outputs  # delete outputs
    mem_2 = torch.cuda.memory_allocated()
    print("\nLoad model:", mem_1)
    print("Forward:", mem_2)

    loss.backward()

    mem_3 = torch.cuda.memory_allocated()
    print("Backward:", mem_3)

    optimizer.step()
    optimizer.zero_grad()

    mem_4 = torch.cuda.memory_allocated()
    print("Update:", mem_4)

    # report activation memory
    print("\nDelta-mode activations:", mem_2 - mem_1)

    # add activation memory
    curr_use = 0
    for k, v in activation_dict.items():
        curr_use += v
    activation_dict = dict()
    # this is more accurate than delta-mode
    print("Track-mode activations:", curr_use)

    # estimate activation memory
    print("Estimated activations:", estimate_model_activation_memory(config, mode))

    # report peak memory
    if peak_memory:
        # for first iteration
        #print("\nFirst-iteration peak memory:", torch.cuda.max_memory_allocated())
        #print("Estimated first-iteration peak memory:", estimate_peak_memory(config, mode, first_iteration=True))

        # for second iteration
        if mixed_precision:
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                outputs = model(input_ids=x, labels=x)
                loss = outputs.loss
        else:
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
        del outputs
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        print("\nPeak memory:", torch.cuda.max_memory_allocated())
        print("Estimated peak memory:", estimate_peak_memory(config, mode))


def estimate_peak_memory(config, mode='autocast', first_iteration=False):
    b = config.batch_size
    s = config.sequence_length
    d = config.n_embd
    vocab_size = config.vocab_size
    n_layer = config.n_layer
    n_ctx = config.n_positions

    # model and optimizer states
    n_param = 12 * d * d * n_layer + d * (vocab_size + n_ctx)
    if mode == 'fp16':
        cost = n_param * 2 if first_iteration else n_param * 6  # first iteration did not count adam states
    else:
        cost = n_param * 4 if first_iteration else n_param * 12

    # forward activations
    cost += estimate_model_activation_memory(config, mode)

    # workspace memory for loss function (fwd+bwd), note that fused_xentropy helps here
    if mode == 'fp32':
        cost += 2 * 4 * b * (s-1) * vocab_size
    elif mode == 'fp16':
        cost += 2 * 2 * b * (s-1) * vocab_size
    elif mode == 'autocast':
        cost += 2 * 2 * b * (s-1) * vocab_size

    # peak memory may happen during optimizer-update (weight, grad, adam states, update workspace), note that fused_adam helps here
    if mode == 'fp16':
        update_cost = n_param * 2 * 5
    else:
        update_cost = n_param * 4 * 5

    return max(cost, update_cost)


def estimate_model_activation_memory(config, mode='autocast'):
    b = config.batch_size
    s = config.sequence_length
    d = config.n_embd
    vocab_size = config.vocab_size

    # embed_dropout, output_layernorm, lm_head, loss
    if mode == 'fp32':
        cost = b * s * d + 4 * b * s * d + 4 * b * s * d + 4 * b * (s-1) * vocab_size
    elif mode == 'fp16':
        cost = b * s * d + 2 * b * s * d + 2 * b * s * d + 2 * b * (s-1) * vocab_size
    elif mode == 'autocast':
        cost = b * s * d + 4 * b * s * d + 2 * b * s * d + 6 * b * (s-1) * vocab_size
        # fp16 head weight is included into activationn memory
        fp16_weight = d * vocab_size * 2
        cost += fp16_weight

    # transformer layers
    cost += estimate_layer_activation_memory(config, mode) * config.n_layer
    return cost


def estimate_layer_activation_memory(config, mode='autocast', eager_attn=True):
    b = config.batch_size
    s = config.sequence_length
    d = config.n_embd
    h = config.n_head
    gelu_new = (config.activation_function == 'gelu_new')

    if not eager_attn: 
        # flash-attention and sdpa did not include the item of b * h * s * s
        h = 0

    if mode == 'fp32':
        if gelu_new:
            cost = 114 * b * s * d + 9 * b * h * s * s
        else:
            cost = 66 * b * s * d + 9 * b * h * s * s
    elif mode == 'fp16':
        if gelu_new:
            cost = 58 * b * s * d + 5 * b * h * s * s
        else:
            cost = 34 * b * s * d + 5 * b * h * s * s  # it is the same to megatron-lm's estimation
    elif mode == 'autocast':
        if gelu_new:
            cost = 86 * b * s * d + 7 * b * h * s * s
        else:
            cost = 38 * b * s * d + 7 * b * h * s * s
        # fp16 weights are included into activation memory
        fp16_weight = d * d * 12 * 2
        cost += fp16_weight
    return cost


def profile_layer(name='gpt2', mode='autocast'):
    config = get_config(name)
    model = GPT2Block(config)
    print("Model:", name)

    batch_size, seq_len = 1, 1024
    hidden_dim = config.n_embd
    config.batch_size, config.sequence_length = batch_size, seq_len

    assert mode in ['fp32', 'fp16', 'autocast']
    dtype = torch.float16 if mode == 'fp16' else torch.float32
    mixed_precision = (mode == 'autocast')
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Dtype: {dtype}, Mixed-precision: {mixed_precision}")

    model.to(device='cuda', dtype=dtype)
    mem_1 = torch.cuda.memory_allocated()

    # add buffer tensors for deduplication
    global activation_dict
    for name, buffer in model.named_buffers():
        name = str(buffer.untyped_storage().data_ptr())
        activation_dict[name] = 0

    x = torch.randn((batch_size, seq_len, hidden_dim), device='cuda', dtype=dtype, requires_grad=True)

    if mixed_precision:
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            # tracking mode
            with TrackingMode(), ActivationMode():
                output = model(x)
    else:
        # tracking mode
        with TrackingMode(), ActivationMode():
            output = model(x)

    mem_2 = torch.cuda.memory_allocated()
    print("Delta-mode one-layer activations:", mem_2 - mem_1)

    # add activation memory
    curr_use = 0
    for k, v in activation_dict.items():
        curr_use += v
    activation_dict = dict()
    # this is more accurate than delta-mode
    print("Track-mode one-layer activations:", curr_use)

    # estimate activation memory
    print("Estimated one-layer activations:", estimate_layer_activation_memory(config, mode))


def profile_gelu(gelu_new=True, mode='autocast'):
    act1 = ACT2FN["gelu_new"] # used in gpt2
    act2 = ACT2FN["gelu_pytorch_tanh"]
    model = act1 if gelu_new else act2

    batch_size, seq_len, hidden_dim = 1, 1024, 768
    assert mode in ['fp32', 'fp16', 'autocast']
    input_dtype = torch.float32 if mode == 'fp32' else torch.float16
    mixed_precision = (mode == 'autocast')
    print("gelu_new:", gelu_new)
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Input dtype: {input_dtype}, Mixed-precision: {mixed_precision}")

    x = torch.randn((batch_size, seq_len, hidden_dim), device='cuda', dtype=input_dtype, requires_grad=True)
    
    # test
    #y1 = act1(x)
    #y2 = act2(x)
    #print("Difference between two gelu:", (y1-y2).abs().max())

    global debug
    debug = True

    if mixed_precision:
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            with TrackingMode(), ActivationMode():
                output = model(x)
    else:
        with TrackingMode(), ActivationMode():
            output = model(x)


if __name__ == '__main__':
    profile_model(name='gpt2', mode='autocast')
    #profile_layer(name='gpt2', mode='fp32')
    #profile_layer(name='gpt2', mode='fp16')
    #profile_layer(name='gpt2', mode='autocast')
    #profile_gelu(gelu_new=True, mode='autocast')
    #profile_gelu(gelu_new=False, mode='autocast')

