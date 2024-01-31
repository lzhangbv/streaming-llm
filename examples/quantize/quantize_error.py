import time

import torch
import torch.nn as nn
import numpy as np

errs = []

def fast_quant(W, name, bits=4, groupsize=128):
    #W = W.clone()
    N, K = W.shape
    assert K % groupsize == 0
    # get scales, zeros, and g_idx
    maxq = 2 ** bits - 1
    W = W.view(-1, groupsize) #(group_num, group_size)
    xmin = torch.clamp(torch.min(W, dim=1)[0], max=0)
    xmax = torch.clamp(torch.max(W, dim=1)[0], min=0)
    scales = (xmax - xmin).clamp(min=1e-5) / maxq
    zeros = torch.round(-xmin / scales)
    # quantize weight
    izeros = zeros.to(torch.int32)
    weight = W / scales[:, None] + zeros[:, None]
    iweight = torch.clamp(torch.round(weight), 0, maxq).to(torch.int32)
    # round error
    round_err = iweight - weight
    #print(f"{name}, round error: {round_err}")
    #hist = torch.histogram(round_err.view(-1), bins=10, range=(-0.5, 0.5), density=True)[0]
    #print(f"{name}, round error histogram: {hist}")
    # quantization error
    err = (iweight - zeros[:, None]) * scales[:, None] - W
    err = (err.norm() / W.norm()).item() #ratio to wnorm
    #err = (W.abs() <= err.abs()).sum().item() / W.numel() #percent of weights with smaller sparse error
    errs.append(err)
    #print(f"{name}, quant error: {err:.03f}")

def make_quant(model, bits=4, groupsize=128):
    """
    Compute quantization errors of linear layers in a model. 
    """
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        # quantize weight
        m_groupsize = m.in_features if groupsize == -1 else groupsize
        fast_quant(m.weight.data, name, bits, m_groupsize)

    global errs
    print(f"bits={bits}, groupsize={groupsize}") 
    print(f"Mean quantization error: {np.mean(errs):.03f}")
    errs = []

def make_noise(model, ratio=0.1):
    """
    Add some noise to linear layers in a model. 
    """
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        W = m.weight.data
        noise = torch.rand(W.shape, dtype=W.dtype, device=W.device) - 0.5
        scale = ratio * torch.norm(W) / torch.norm(noise)
        W.add_(noise, alpha=scale)   
 
if __name__ == "__main__":
    """Study the effect of quantization error"""
    # load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stime = time.time()
    model_id = "/mnt/data/llama2-7b-chat/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Time to load: {time.time()-stime:.02f}")
    stime = time.time()
    #make_quant(model, 4, -1)
    make_quant(model, 4, 128)
    #make_quant(model, 4, 64)
    #make_quant(model, 4, 32)
    print(f"Time to quantize: {time.time()-stime:.02f}")

    # add noise
    #make_noise(model, ratio=0.1)
    #print("Testing ...")
    #prompt = "What is deep learning?"
    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #outputs = model.generate(**inputs, max_new_tokens=25)
    #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

