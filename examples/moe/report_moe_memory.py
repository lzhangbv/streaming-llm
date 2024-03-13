import torch

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.llama.modeling_llama import LlamaMLP

"""
MoE has increased weight-and-grad memory 'num_experts' times, and activation memory about 'topk' times.
"""

class MoEConfig:
    def __init__(self, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = "silu"
        self.pretraining_tp = 1

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    num_experts = 8
    topk = 2

    dim = 4096
    batch_size = 1
    seqlen = 2048

    config = MoEConfig(dim, 14336, num_experts, topk)

    if num_experts == 1:
        model = LlamaMLP(config)
    else:
        model = MixtralSparseMoeBlock(config)

    model.to(device=device, dtype=dtype)

    torch.cuda.memory._record_memory_history(max_entries=100000)

    x = torch.randn(batch_size, seqlen, dim, device=device, dtype=dtype)
    x.requires_grad_()

    if num_experts == 1:
        out = model(x)
    else:
        out, _ = model(x)

    loss = out.sum()
    loss.backward()

    torch.cuda.memory._dump_snapshot("snapshot.pickle") # open it at https://pytorch.org/memory_viz
    torch.cuda.memory._record_memory_history(enabled=None)
    

