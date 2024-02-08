import time

import torch
import torch.nn as nn
import numpy as np

def make_noise_linear(model, ratio=0.1, target=None):
    """
    Add some noise to linear layers in a model. 
    """
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        
        if (target is None) or ((target is not None) and (target in name)):
            print(name)
            W = m.weight.data
            noise = torch.rand(W.shape, dtype=W.dtype, device=W.device) - 0.5
            scale = ratio * torch.norm(W) / torch.norm(noise)
            W.add_(noise, alpha=scale)   
 
def make_noise_norm(model, ratio=0.1, target="layernorm"):
    """
    Add some noise to other modules, such as rmsnorm, embedding, and lm_head. 
    """
    for name, m in model.named_modules():
        if target in name:
            print(name)
            W = m.weight.data
            noise = torch.rand(W.shape, dtype=W.dtype, device=W.device) - 0.5
            scale = ratio * torch.norm(W) / torch.norm(noise)
            W.add_(noise, alpha=scale)

if __name__ == "__main__":
    """Make noise on target model weights"""
    # load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stime = time.time()
    model_id = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Time to load: {time.time()-stime:.02f}")

    # target: [q_proj, k_poj, v_proj, o_proj, gate_proj, up_proj, down_proj, attn, mlp, proj]
    make_noise_linear(model, ratio=0.1, target="proj")

    # target: [layernorm, model.norm, embed, lm_head]
    #make_noise_norm(model, ratio=0.1, target="layernorm")

    print("Testing ...")
    prompt = "What is deep learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

