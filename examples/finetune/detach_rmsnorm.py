"""
DetachRMSNorm: detach variance from the autograd graph  

Note:
    1) Native LlamaRMSNorm requires three input activations, while frozen version requires two input activations
    2) Fused RMSNorm requires one input activation, while memory-efficient version saves its output activation instead
    3) Detach RMSNorm requires one input activation, while frozen version requires no input activation (except small variance)
"""

import torch
from torch import nn


class DetachRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True).detach()
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # detach rmsnorm
    LlamaRMSNorm.forward = DetachRMSNorm.forward

    # config
    model_id = "lmsys/vicuna-7b-v1.3"

    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    
    model = AutoModelForCausalLM.from_config(config)
    model.to(device='cuda')
    
    # freeze base model
    for name, p in model.named_parameters():
        # skip freezing layernorm
        #if 'norm' in name:
        #    continue
        p.requires_grad = False

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

