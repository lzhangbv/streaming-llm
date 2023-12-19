from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def svd(x, y, alpha=1, rank=4):
    #alpha = torch.norm(x) / torch.norm(y)
    #print(alpha)

    delta = x - alpha * y
    
    U, S, V = torch.svd_lowrank(delta, rank)
    #print(U.shape, S.shape, V.shape)
    x_approx = U @ torch.diag(S) @ V.T + alpha * y
    
    #U, S, V = torch.linalg.svd(delta)
    #print(U.shape, S.shape, V.shape)
    #x_approx = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :] + alpha * y

    # print((S[:rank] / S[0]).tolist())
    return x_approx

def set_attn_weights(model, moe_model):
    qs, ks, vs, os = get_attn_weights(model)
    moe_qs, moe_ks, moe_vs, moe_os = get_attn_weights(moe_model)
    for i in range(len(qs)):
        q_approx = svd(moe_qs[i], qs[i])
        moe_qs[i].copy_(q_approx)
        k_approx = svd(moe_ks[i], ks[i])
        moe_ks[i].copy_(k_approx)
        v_approx = svd(moe_vs[i], vs[i])
        moe_vs[i].copy_(v_approx)
        o_approx = svd(moe_os[i], os[i])
        moe_os[i].copy_(o_approx)

def set_mlp_weights(model, moe_model):
    w1, w2, w3 = get_mlp_weights(model)
    moe_w1, moe_w2, moe_w3 = get_mlp_weights(moe_model)
    for i in range(len(w1)):
        w1_approx = svd(moe_w1[i], w1[i])
        moe_w1[i].copy_(w1_approx)
        w2_approx = svd(moe_w2[i], w2[i])
        moe_w2[i].copy_(w2_approx)
        w3_approx = svd(moe_w3[i], w3[i])
        moe_w3[i].copy_(w3_approx)

def set_moe_weights(model, moe_model, num_experts=8):
    w1, w2, w3 = get_mlp_weights(model)
    for moe_idx in range(num_experts):
        moe_w1, moe_w2, moe_w3 = get_moe_weights(moe_model, idx=moe_idx)
        for i in range(len(w1)):
            w1_approx = svd(moe_w1[i], w1[i])
            moe_w1[i].copy_(w1_approx)
            w2_approx = svd(moe_w2[i], w2[i])
            moe_w2[i].copy_(w2_approx)
            w3_approx = svd(moe_w3[i], w3[i])
            moe_w3[i].copy_(w3_approx)

def test_weight_mixture(model, moe_model):
    set_attn_weights(model, moe_model)
    set_mlp_weights(model, moe_model)
    # set_moe_weights(model, moe_model)

    text = "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2)."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = moe_model.generate(**inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def compute_attn_similarity(model, moe_model):
    # attn weights
    qs, ks, vs, os = get_attn_weights(model)
    moe_qs, moe_ks, moe_vs, moe_os = get_attn_weights(moe_model)

    for i in range(len(qs)):
        print(cosine_similarity(qs[i], moe_qs[i]).item())
        print(cosine_similarity(ks[i], moe_ks[i]).item())
        print(cosine_similarity(vs[i], moe_vs[i]).item())
        print(cosine_similarity(os[i], moe_os[i]).item())

        # print(f"Layer {i}, Q delta: {torch.dist(moe_qs[i], svd(moe_qs[i], qs[i])).item()}")
        # print(f"Layer {i}, K delta: {torch.dist(moe_ks[i], svd(moe_ks[i], ks[i])).item()}")
        # print(f"Layer {i}, V delta: {torch.dist(moe_vs[i], svd(moe_vs[i], vs[i])).item()}")
        # print(f"Layer {i}, O delta: {torch.dist(moe_os[i], svd(moe_os[i], os[i])).item()}")

def compute_mlp_similarity(model, moe_model):
    # mlp weights
    w1, w2, w3 = get_mlp_weights(model)
    # moe weights
    #w1, w2, w3 = get_moe_weights(moe_model, idx=0)
    #w1, w2, w3 = get_moe_avg_weights(moe_model)

    # other mlp weights
    moe_w1, moe_w2, moe_w3 = get_mlp_weights(moe_model)
    # other moe weights
    # moe_w1, moe_w2, moe_w3 = get_moe_weights(moe_model, idx=1)

    for i in range(len(w1)):
        print(cosine_similarity(w1[i], moe_w1[i]).item())
        print(cosine_similarity(w2[i], moe_w2[i]).item())
        print(cosine_similarity(w3[i], moe_w3[i]).item())

        # print(f"Layer {i}, W1 delta: {torch.dist(moe_w1[i], svd(moe_w1[i], w1[i])).item()}")
        # print(f"Layer {i}, W2 delta: {torch.dist(moe_w2[i], svd(moe_w2[i], w2[i])).item()}")
        # print(f"Layer {i}, W3 delta: {torch.dist(moe_w3[i], svd(moe_w3[i], w3[i])).item()}")

### main ###
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
moe_model = AutoModelForCausalLM.from_pretrained(model_id)
moe_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

compute_attn_similarity(model, moe_model)
# compute_mlp_similarity(model, moe_model)
# test_weight_mixture(model, moe_model)
