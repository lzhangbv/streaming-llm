import torch
from torch.profiler import profile, ProfilerActivity

"""
Profile moe computation in different implementations, and study the effects of cpu-sync. 
"""
# for some operations, we need cpu-sync if output tensor shape is unknown
# and depends on on-device tensor elements, such as nonzero and masked_select.

# expert configs
num_experts = 8
topk = 2
num_local_experts = 2 # with ep


def mixtral_moe(hidden_states, max_prob, max_ind, weight):
    """
    Mixtral-MoE runs the loop over all experts, and it uses torch.where to get per-expert tokens. 
    """
    routing_weights = max_prob
    selected_experts = max_ind

    num_tokens, hidden_dim = hidden_states.shape
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    #print(expert_mask.shape) # (num_experts, topk, num_tokens)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(num_experts):
        # 1) get index with torch.where (cpu-sync), which is identical to torch.nonzero(x, as_tuple=True)
        idx, top_x = torch.where(expert_mask[expert_idx])
        #print(idx)   # which topk index to expert-i
        #print(top_x) # which token index to expert-i
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        
        # 2) single expert computation (dummy)
        current_state = torch.matmul(current_state, weight)
        current_state = torch.matmul(current_state, weight.t())
        current_state *= routing_weights[top_x, idx, None]

        # 3) index_add with index
        final_hidden_states.index_add_(0, top_x, current_state.to(hidden_states.dtype))

    return final_hidden_states


def megatron_moe_with_ep(hidden_states, max_prob, max_ind, weight): 
    """
    Megatron-MoE with all-gather token dispatcher. 
    """
    # 1) global gather
    global_local_mask = max_ind < num_local_experts

    # a) max_prob, max_ind -> local_prob, local_indices with masked_select (cpu-sync)
    local_indices = max_ind.masked_select(global_local_mask)
    local_probs = max_prob.masked_select(global_local_mask)

    # b) hidden_states -> local_hidden_states with nonzero (cpu-sync)
    global_local_map = global_local_mask.nonzero()[:, 0] # get all nonzero row ids
    global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
    local_hidden_states = torch.gather(hidden_states, 0, global_local_map)

    # 2) local permutation
    tokens_per_expert = torch.histc(
        local_indices.float(), 
        bins=num_local_experts, 
        min=0, 
        max=num_local_experts-1,
        ).to(torch.long)
    # print(tokens_per_expert)

    sorted_indices = torch.argsort(local_indices, dim=0)
    sorted_indices = sorted_indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
    permuted_local_hidden_states = torch.gather(local_hidden_states, 0, sorted_indices)

    # 3) local expert computation (dummy)
    local_outputs = torch.matmul(permuted_local_hidden_states, weight)
    local_outputs = torch.matmul(local_outputs, weight.t())

    # 4) local unpermutation
    unpermuted_local_outputs = torch.zeros_like(local_outputs)
    unpermuted_local_outputs = local_outputs.scatter(0, sorted_indices, local_outputs)

    # 5) weighted sum
    unpermuted_local_outputs = unpermuted_local_outputs * local_probs.view(-1, 1)

    # 6) global scatter
    unpermuted_global_outputs = torch.zeros(
        (hidden_states.shape[0], unpermuted_local_outputs.shape[-1]), 
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    outputs = unpermuted_global_outputs.scatter_add(0, global_local_map, unpermuted_local_outputs)

    return outputs


def tutel_moe(hidden_states, logits, weight, cf):
    """
    Tutel-MoE with fast encode/decode, and capacity = (num_tokens * topk * cf) / num_experts.
    """
    scores = torch.nn.functional.softmax(logits, dim=1)
    # permutation
    crit, _ = extract_critical(
        scores,
        top_k=topk,
        loss_fn=None,
        capacity_factor=cf,
    )
    dispatched_input = fast_encode(hidden_states, crit, True)
    #print(dispatched_input.shape) #(num_experts, capacity, dim)

    # batch expert computation (dummy), todo: torch.bmm
    dispatched_outputs = torch.matmul(dispatched_input, weight)
    dispatched_outputs = torch.matmul(dispatched_outputs, weight.t())

    # unpermutation
    outputs = fast_decode(dispatched_outputs, crit, True)
    return outputs


def custom_moe(hidden_states, max_prob, max_ind, weight):
    """
    Custom MoE: permutation, grouped gemm, unpermutation
    """
    batch_size, dim = hidden_states.shape
    max_prob = max_prob.view(-1)
    max_ind = max_ind.view(-1)

    # tokens per expert
    tokens_per_expert = torch.histc(
        max_ind.float(),
        bins=num_experts,
        min=0,
        max=num_experts-1,
        ).to(torch.long)
    #print(tokens_per_expert)

    # permutation
    perm_ids = torch.argsort(max_ind)
    sorted_hidden_states = torch.index_select(hidden_states, 0, perm_ids // topk)

    # moe computation (dummy)
    grouped_gemm = True

    if not grouped_gemm: 
        # send tokens_per_expert to cpu to get split tensor shapes
        split_inputs = sorted_hidden_states.split(tokens_per_expert.tolist())
        outputs = []
        for split_input in split_inputs:
            outputs.append(torch.matmul(torch.matmul(split_input, weight), weight.t()))
        sorted_outputs = torch.cat(outputs)

    else:
        # todo: grouped gemm with tokens_per_expert
        sorted_outputs = torch.matmul(sorted_hidden_states, weight)
        sorted_outputs = torch.matmul(sorted_outputs, weight.t())

    # unpermutation
    unperm_ids = torch.argsort(perm_ids)
    outputs = torch.index_select(sorted_outputs, 0, unperm_ids)

    # weighted sum
    outputs *= max_prob.view(-1, 1)
    outputs = outputs.view(batch_size, topk, dim).sum(dim=1)
    return outputs


if __name__ == "__main__": 
    batch_size, dim = 4096, 4096

    # inputs
    weight = torch.randn((dim, dim * 4), dtype=torch.float32, device='cuda').div_(dim)
    hidden_states = torch.randn((batch_size, dim), dtype=torch.float32, device='cuda')
    weight_dummy = torch.randn((dim, dim), dtype=torch.float32, device='cuda')
    logits = torch.randn((batch_size, num_experts), dtype=torch.float32, device='cuda') 

    max_logit, max_ind = torch.topk(logits, k=topk, dim=1)
    max_prob = torch.softmax(max_logit, dim=-1)

    # warmup
    outputs = mixtral_moe(hidden_states, max_prob, max_ind, weight)
    #outputs = megatron_moe_with_ep(hidden_states, max_prob, max_ind, weight)
    #outputs = custom_moe(hidden_states, max_prob, max_ind, weight)
    
    #from tutel.impls.fast_dispatch import fast_decode, fast_encode, extract_critical
    #outputs = tutel_moe(hidden_states, logits, weight, cf=1.2)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # make cpu run faster than gpu
        for i in range(10):
            torch.matmul(hidden_states, weight_dummy)

        outputs = mixtral_moe(hidden_states, max_prob, max_ind, weight)
        #outputs = megatron_moe_with_ep(hidden_states, max_prob, max_ind, weight)
        #outputs = custom_moe(hidden_states, max_prob, max_ind, weight)
        #outputs = tutel_moe(hidden_states, logits, weight, cf=1.2)

    prof.export_chrome_trace("trace.json")


