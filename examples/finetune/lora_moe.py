"""
Expand Mistral-7B dense model into MoE using base weight and LoRA. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


LORA_R = 32
TOP_K = 2
NUM_EXPERTS = 8
AUX_LOSS_COEF = 0.02


def load_balancing_loss_func(
    routing_weights, expert_mask, num_experts, loss_coef):
    """
    Computes auxiliary load balancing loss as in Switch Transformer.
    """
    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    aux_loss = aux_loss * num_experts * loss_coef
    return aux_loss


class MoEAuxLoss(torch.autograd.Function):
    """Compute the grad for auxiliary loss"""
    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        scaled_aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, scaled_aux_loss_grad


class MixtralBLockSparseTop2MLPLora(nn.Module):
    def __init__(self, config, w1, w2, w3):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # base
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # LoRA
        self.w1_a = nn.Linear(self.hidden_dim, LORA_R, bias=False)
        self.w1_b = nn.Linear(LORA_R, self.ffn_dim, bias=False)
        self.w2_a = nn.Linear(self.ffn_dim, LORA_R, bias=False)
        self.w2_b = nn.Linear(LORA_R, self.hidden_dim, bias=False)
        self.w3_a = nn.Linear(self.hidden_dim, LORA_R, bias=False)
        self.w3_b = nn.Linear(LORA_R, self.ffn_dim, bias=False)
        
        # init LoRA B
        self.w1_b.weight.data.fill_(0.)
        self.w2_b.weight.data.fill_(0.)
        self.w3_b.weight.data.fill_(0.)

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


class MixtralSparseMoeBlockLora(nn.Module):
    def __init__(self, config, mlp): 
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.router_aux_loss_coef = AUX_LOSS_COEF

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # base model
        self.w1, self.w2, self.w3 = mlp.gate_proj, mlp.down_proj, mlp.up_proj

        # MoE with LoRAs
        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLPLora(config, self.w1, self.w2, self.w3) for i in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        all_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(all_routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)

        # auxiliary loss
        if self.router_aux_loss_coef > 0 and self.training: 
            aux_loss = load_balancing_loss_func(all_routing_weights, expert_mask, self.num_experts, self.router_aux_loss_coef)
            routing_weights = MoEAuxLoss.apply(routing_weights, aux_loss)
            #print(f"aux loss: {aux_loss.item()}")

        expert_mask = expert_mask.permute(2, 1, 0)
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
        return final_hidden_states


def make_lora_moe(model):
    # freeze base model
    for name, p in model.named_parameters():
        p.requires_grad = False
    
    # add router and lora adaptors
    for name, module in model.named_modules():
        if "mlp" in name and "mlp." not in name:
            print(name)
            moe_block_lora = MixtralSparseMoeBlockLora(model.config, module)
            parent_name = name.rsplit('.', 1)[0]
            parent = model.get_submodule(parent_name)
            setattr(parent, name[len(parent_name) + 1:], moe_block_lora)


def report_memory(model_id):
    config = AutoConfig.from_pretrained(model_id)
    model = LlamaMLP(config)
    model = MixtralSparseMoeBlockLora(config, model)

    model.to(device='cuda')

    batch_size, seq_len = 4, 2048
    dim = config.hidden_size
    x = torch.randn((batch_size, seq_len, dim), device='cuda')
    
    loss = model(x).mean()
    loss.backward()
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaMLP
    
    model_id = "mistralai/Mistral-7B-v0.1"

    #report_memory(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cuda')
    model.eval()

    make_lora_moe(model)
    model.to(device='cuda', dtype=torch.float16)

    text = "What is deep learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

