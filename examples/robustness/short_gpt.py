import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
"""
def delete_layer(model, start, number): 
    layers = torch.nn.ModuleList()
    layer_idx = 0
    for i, layer in enumerate(model.model.layers):
        if i >= start and i < start + number:
            print(f"delete layer {i}")
            continue
        
        layer.self_attn.layer_idx = layer_idx
        layers.append(layer)
        layer_idx += 1

    model.model.layers = layers

if __name__ == "__main__":
    model_id = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()

    number = 4
    start = 31 - number # delete layers 27, 28, 29, 30 (exclude last layer)
    delete_layer(model, start, number)
    model.to("cuda")

    # test
    text = "What is deep learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

