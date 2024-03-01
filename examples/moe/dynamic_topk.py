""" PyTorch Mixtral model with Dynamic Topk Routing."""

import torch
import torch.nn.functional as F
import types

# small threshold gives more top1 routing and worse performance
THRESHOLD = 0.3

def dynamic_topk_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

    # top1 mask: set the 2rd expert weight to 0
    top1_mask = routing_weights[:,0] - routing_weights[:,1] > THRESHOLD
    routing_weights[top1_mask, 1] = 0
    
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
    
    # top1 mask: set the 2rd expert assignment to 0
    expert_mask[top1_mask, 1, :] = 0
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
    return final_hidden_states, router_logits

def enable_dynamic_topk(model, threshold=0.3):
    global THRESHOLD
    THRESHOLD = threshold
    for name, module in model.named_modules():
        if "block_sparse_moe" in name and "block_sparse_moe." not in name:
            module.forward = types.MethodType(dynamic_topk_forward, module)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # set pad token id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    enable_dynamic_topk(model, threshold=0.3)
    
    text = "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2)."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
