import torch
from torch.utils._python_dispatch import TorchDispatchMode

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Block


activation_dict = dict()


class TrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        res = func(*args, **kwargs or {})
        #print('operation:', func)
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
                    #print('saved activation:', x.dtype, x.shape, x.element_size()*x.numel(), x_.size())
                    activation_dict[name] = x_.size()
                else:
                    #print('duplicated activation:', x.dtype, x.shape, x.element_size()*x.numel(), x_.size())
                    pass

            return x

        def unpack(x):
            return x
        super().__init__(pack, unpack)


def get_config(name='gpt2'):
    if name == 'gpt2': #124M
        return GPT2Config(n_embd=768, n_head=12, n_layer=12)
    elif name == 'gpt2-medium': #355M
        return GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    elif name == 'gpt2-large': #774M
        return GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    elif name == 'gpt2-xl': #1.5B
        return GPT2Config(n_embd=1600, n_head=25, n_layer=48)


def count_parameter(model):
    print('tied embedding:', model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr())
    count = 0
    for name, param in model.named_parameters():
        count += param.numel()
        # if len(param.shape) > 1:
        #     print(name, param.shape)
    return count


def profile_model(name='gpt2', mode='autocast'):
    config = get_config(name)
    model = GPT2LMHeadModel(config)
    print("Model:", name)
    print("Num of parameters:", count_parameter(model))

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
    print("Load model:", mem_1)

    optimizer = torch.optim.AdamW(model.parameters())
    assert torch.cuda.memory_allocated() == mem_1

    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    if mixed_precision:
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
    else:
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss

    del outputs  # delete outputs
    mem_2 = torch.cuda.memory_allocated()
    print("Forward:", mem_2)

    loss.backward()

    mem_3 = torch.cuda.memory_allocated()
    print("Backward:", mem_3)

    optimizer.step()
    optimizer.zero_grad()

    mem_4 = torch.cuda.memory_allocated()
    print("Update:", mem_4)
    # print("Peak:", torch.cuda.max_memory_allocated())


def estimate_activation_memory(config, mode='autocast'):
    b = config.batch_size
    s = config.sequence_length
    d = config.n_embd
    h = config.n_head

    if mode == 'fp32':
        cost = 114 * b * s * d + 9 * b * h * s * s
    elif mode == 'fp16':
        cost = 58 * b * s * d + 5 * b * h * s * s
    elif mode == 'autocast':
        cost = 86 * b * s * d + 7 * b * h * s * s
        # fp16 weights are included into activation memory
        fp16_weight = d * d * 12 * 2
        cost += fp16_weight
    return cost


def profile_layer(name='gpt2', mode='fp16'):
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
    # this is more accurate than diff-mode
    print("Track-mode one-layer activations:", curr_use)

    # estimate activation memory
    print("Estimated one-layer activations:", estimate_activation_memory(config, mode))


if __name__ == '__main__':
    #profile_model(name='gpt2', mode='autocast')
    #profile_layer(name='gpt2', mode='fp32')
    #profile_layer(name='gpt2', mode='fp16')
    profile_layer(name='gpt2', mode='autocast')

