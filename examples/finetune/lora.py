"""
LoRA and LoRA-FA.   
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class LinearLoRA(nn.Module):
    def __init__(self, linear, r, alpha, lora_fa):
        super().__init__()

        self.base = linear
        self.r = r
        self.alpha = alpha
        self.lora_fa = lora_fa
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.w_a = nn.Linear(self.in_features, r, bias=False)
        self.w_b = nn.Linear(r, self.out_features, bias=False)

        self.w_b.weight.data.fill_(0.)
        if lora_fa:
            self.w_a.weight.requires_grad = False
    
    def forward(self, hidden_states):
        return self.base(hidden_states) + self.alpha * self.w_b(self.w_a(hidden_states))


def make_lora(model, r=32, alpha=1, lora_fa=False):
    # freeze base model
    for name, p in model.named_parameters():
        p.requires_grad = False
    
    # add lora adaptors
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue

        linear_lora = LinearLoRA(m, r, alpha, lora_fa)
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)
        setattr(parent, name[len(parent_name) + 1:], linear_lora)

    #for name, p in model.named_parameters():
    #    if p.requires_grad:
    #        print(name)

def test(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cuda:0')
    model.eval()
    
    make_lora(model, lora_fa=True)
    model.to(dtype=torch.float16, device='cuda')

    text = "What is deep learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    model_id = "lmsys/vicuna-7b-v1.3"
    
    # test
    #test(model_id)

    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_config(config)

    make_lora(model, lora_fa=True)
    model.to(device='cuda')

    batch_size = 64
    seq_len = 128
    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len)).cuda()

    #torch.cuda.memory._record_memory_history(max_entries=100000)

    outputs = model(input_ids=x, labels=x)
    loss = outputs.loss
    del outputs
    loss.backward()
    
    #torch.cuda.memory._dump_snapshot("snapshot.pickle") # open it at https://pytorch.org/memory_viz
    #torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

