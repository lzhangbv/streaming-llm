from transformers import AutoModelForCausalLM
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
moe_model = AutoModelForCausalLM.from_pretrained(model_id)
moe_model.eval()

def get_attn_weights(model):
    qs = []
    ks = []
    vs = []
    os = []
    for name, module in model.named_modules():
        if "q_proj" in name:
            #print(name)
            qs.append(module.weight.data)
        if "k_proj" in name:
            #print(name)
            ks.append(module.weight.data)
        if "v_proj" in name:
            #print(name)
            vs.append(module.weight.data)
        if "o_proj" in name:
            #print(name)
            os.append(module.weight.data)
    return qs, ks, vs, os

def get_mlp_weights(model):
    w1 = []
    w2 = []
    w3 = []
    for name, module in model.named_modules():
        if "mlp" in name and "mlp." not in name:
            #print(name)
            w1.append(module.gate_proj.weight.data)
            w2.append(module.down_proj.weight.data)
            w3.append(module.up_proj.weight.data)
    return w1, w2, w3

def get_moe_weights(model, idx=0):
    w1 = []
    w2 = []
    w3 = []
    for name, module in model.named_modules():
        if "block_sparse_moe" in name and "block_sparse_moe." not in name:
            #print(name)
            w1.append(module.experts[idx].w1.weight.data)
            w2.append(module.experts[idx].w2.weight.data)
            w3.append(module.experts[idx].w3.weight.data)
    return w1, w2, w3

def get_moe_avg_weights(model, num_experts=8):
    w1 = []
    w2 = []
    w3 = []
    for name, module in model.named_modules():
        if "block_sparse_moe" in name and "block_sparse_moe." not in name:
            #print(name)
            result = module.experts[0].w1.weight.data / num_experts
            for i in range(1, num_experts):
                result +=  module.experts[i].w1.weight.data / num_experts
            w1.append(result)

            result = module.experts[0].w2.weight.data / num_experts
            for i in range(1, num_experts):
                result +=  module.experts[i].w2.weight.data / num_experts
            w2.append(result)

            result = module.experts[0].w3.weight.data / num_experts
            for i in range(1, num_experts):
                result +=  module.experts[i].w3.weight.data / num_experts
            w3.append(result)
    return w1, w2, w3

def cosine_similarity(x, y):
    x = x.view(-1)
    y = y.view(-1)
    return torch.nn.functional.cosine_similarity(x, y, dim=0)

def svd(x, y, alpha=1, rank=64):
    #alpha = torch.norm(x) / torch.norm(y)
    #print(alpha)

    delta = x - alpha * y
    
    U, S, V = torch.svd_lowrank(delta, rank)
    #print(U.shape, S.shape, V.shape)
    x_approx = U @ torch.diag(S) @ V.T + alpha * y
    
    #U, S, V = torch.linalg.svd(delta)
    #print(U.shape, S.shape, V.shape)
    #x_approx = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :] + alpha * y
    return torch.dist(x, x_approx).item()

# attn weights
# qs, ks, vs, os = get_attn_weights(model)
# moe_qs, moe_ks, moe_vs, moe_os = get_attn_weights(moe_model)

#for i in range(len(qs)):
    #print(cosine_similarity(qs[i], moe_qs[i]).item())
    #print(cosine_similarity(ks[i], moe_ks[i]).item())
    #print(cosine_similarity(vs[i], moe_vs[i]).item())
    #print(cosine_similarity(os[i], moe_os[i]).item())

    #print(f"Layer {i}, Q delta: {svd(moe_qs[i], qs[i])}")
    #print(f"Layer {i}, K delta: {svd(moe_ks[i], ks[i])}")
    #print(f"Layer {i}, V delta: {svd(moe_vs[i], vs[i])}")
    #print(f"Layer {i}, O delta: {svd(moe_os[i], os[i])}")

# mlp weights
#w1, w2, w3 = get_mlp_weights(model)
#w1, w2, w3 = get_moe_weights(moe_model, idx=0)
w1, w2, w3 = get_moe_avg_weights(moe_model)
moe_w1, moe_w2, moe_w3 = get_moe_weights(moe_model, idx=1)


for i in range(len(w1)):
    print(cosine_similarity(w1[i], moe_w1[i]).item())
    print(cosine_similarity(w2[i], moe_w2[i]).item())
    print(cosine_similarity(w3[i], moe_w3[i]).item())

    #print(f"Layer {i}, W1 delta: {svd(moe_w1[i], w1[i])}")
    #print(f"Layer {i}, W2 delta: {svd(moe_w2[i], w2[i])}")
    #print(f"Layer {i}, W3 delta: {svd(moe_w3[i], w3[i])}")
