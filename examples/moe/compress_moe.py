"""
Compress experts in Mixtral-8x7B MoE model using shared weight and LoRA. 
(this idea didn't work out as the delta_w is not low-rank) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

LORA_R = 32

class MixtralBLockSparseTop2MLPLora(nn.Module):
    def __init__(self, config, wExperts, w1, w2, w3):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # shared MLP with LoRA
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.w1_a = nn.Linear(self.hidden_dim, LORA_R, bias=False)
        self.w1_b = nn.Linear(LORA_R, self.ffn_dim, bias=False)
        self.w2_a = nn.Linear(self.ffn_dim, LORA_R, bias=False)
        self.w2_b = nn.Linear(LORA_R, self.hidden_dim, bias=False)
        self.w3_a = nn.Linear(self.hidden_dim, LORA_R, bias=False)
        self.w3_b = nn.Linear(LORA_R, self.ffn_dim, bias=False)
        
        # delta = original MLP - shared MLP
        delta_w1 = wExperts.w1.weight.data - w1.weight.data
        delta_w2 = wExperts.w2.weight.data - w2.weight.data
        delta_w3 = wExperts.w3.weight.data - w3.weight.data
        
        # approximate with SVD
        # delta_w1
        U, S, V = torch.svd_lowrank(delta_w1, LORA_R)
        self.w1_a.weight.data.copy_(torch.matmul(torch.diag(S), V.T))
        self.w1_b.weight.data.copy_(U)
        # delta_w2 
        U, S, V = torch.svd_lowrank(delta_w2, LORA_R)
        self.w2_a.weight.data.copy_(torch.matmul(torch.diag(S), V.T))
        self.w2_b.weight.data.copy_(U)
        # delta_w3
        U, S, V = torch.svd_lowrank(delta_w3, LORA_R)
        self.w3_a.weight.data.copy_(torch.matmul(torch.diag(S), V.T))
        self.w3_b.weight.data.copy_(U)

        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, hidden_states):
        x1 = self.w1(hidden_states)
        x1 = x1 + self.w1_b(self.w1_a(hidden_states))
        x3 = self.w3(hidden_states)
        x3 = x3 + self.w3_b(self.w3_a(hidden_states))
        x2 = self.act_fn(x1) * x3
        current_hidden_states = self.w2(x2)
        current_hidden_states = current_hidden_states + self.w2_b(self.w2_a(x2))
        return current_hidden_states

def get_means(experts, hidden_dim, ffn_dim):
    w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
    w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
    w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
    w1.weight.data.fill_(0)
    w2.weight.data.fill_(0)
    w3.weight.data.fill_(0)
    size = len(experts)
    for i in range(size):
        w1.weight.data += experts[i].w1.weight.data / size
        w2.weight.data += experts[i].w2.weight.data / size
        w3.weight.data += experts[i].w3.weight.data / size
    return w1, w2, w3

class MixtralSparseMoeBlockLora(nn.Module):
    def __init__(self, config, moe_model):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.gate.weight.data.copy_(moe_model.gate.weight.data)
        
        # create shared parameters
        self.w1, self.w2, self.w3 = get_means(moe_model.experts, self.hidden_dim, self.ffn_dim)
        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLPLora(config, moe_model.experts[i], self.w1, self.w2, self.w3) for i in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

for name, module in model.named_modules():
    if "block_sparse_moe" in name and "block_sparse_moe." not in name:
        print(name)
        _set_module(model, name, MixtralSparseMoeBlockLora(model.config, module))

# test
text = "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2). Based on this result, find x such that f(x) = 0."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
