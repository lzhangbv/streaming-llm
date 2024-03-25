"""
Checkpoint SiLU: Recompute SiLU output for element-wise product
"""

import torch
import types

class SiLUCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y): 
        """out = silu(x) * y"""
        with torch.no_grad():
            act = torch.nn.functional.silu(x)
            out = act * y
        ctx.save_for_backward(x, y)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors

        x = x.detach().requires_grad_()
        
        with torch.enable_grad():
            a = torch.nn.functional.silu(x)

        with torch.no_grad():
            dy = dout * a
            da = dout * y

        torch.autograd.backward(a, da)

        return x.grad, dy


def silu_checkpoint(x, y):
    return SiLUCheckpointFunction.apply(x, y)


def silu_naive(x, y):
    return torch.nn.functional.silu(x) * y


def replace_silu(model):
    def mlp_forward(self, x):
        return self.down_proj(silu_checkpoint(self.gate_proj(x), self.up_proj(x)))

    for name, module in model.named_modules():
        if "mlp" in name and "mlp." not in name:
            module.forward = types.MethodType(mlp_forward, module)


def test():
    batch, dim = 4, 4096
    
    x = torch.randn((batch, dim), dtype=torch.float16, device='cuda')
    W1 = torch.randn((dim, dim), dtype=torch.float16, device='cuda', requires_grad=True)
    W2 = torch.randn((dim, dim), dtype=torch.float16, device='cuda', requires_grad=True)

    W1_ref = W1.clone().requires_grad_()
    W2_ref = W2.clone().requires_grad_()

    out = silu_checkpoint(torch.matmul(x, W1), torch.matmul(x, W2))
    out.sum().backward()

    out_ref = silu_naive(torch.matmul(x, W1_ref), torch.matmul(x, W2_ref))
    out_ref.sum().backward()

    print(f"The maximum difference of out is {torch.max(torch.abs(out - out_ref))}")
    print(f"The maximum difference of grad1 is {torch.max(torch.abs(W1.grad - W1.grad))}")
    print(f"The maximum difference of grad2 is {torch.max(torch.abs(W2.grad - W2.grad))}")


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    #test()
    
    # model
    model_id = "lmsys/vicuna-7b-v1.3"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2

    model = AutoModelForCausalLM.from_config(config)
    model.to(device='cuda')

    # freeze base model
    for name, p in model.named_parameters():
        p.requires_grad = False

    # replace silu
    replace_silu(model)

    # inputs
    batch_size, seq_len = 32, 128
    dim = config.hidden_size

    #torch.cuda.memory._record_memory_history(max_entries=100000)

    embeds = torch.randn((batch_size, seq_len, dim), device='cuda', requires_grad=True)
    labels = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')
    outputs = model(inputs_embeds=embeds, labels=labels)
    loss = outputs.loss
    del outputs
    loss.backward()

    #torch.cuda.memory._dump_snapshot("snapshot.pickle") # open it at https://pytorch.org/memory_viz
    #torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


