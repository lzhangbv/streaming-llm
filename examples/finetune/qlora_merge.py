"""
QLoRA-Merge: Reducing Quantization Noise of QLoRA Finetuning.

1) motivation of model merge
without mering, runing QLoRA model needs to dequant the weight and add the lora result:
    out = x @ dequant(quant_w) + x @ lora_a @ lora_b, 
which is much slower than fp16 model inference. 

If we merge quant model and lora adaptor into a fp16 model, it can accelerate inference.
If memory cost is an issue, after merge, we can use better quant algorithm and faster fused w4a16 kernel.   

2) current solution of model merge
We can simply get the merged QLoRA weight as follows:
    qlora_w = dequant(quant(base_w)) + lora_a @ lora_b. 
Note that quantization configs are the same as QLoRA fine-tuning process. 
    
3) limitation of current solution
On one hand, quant-and-dequant introduced quant noise, that is
    dequant(quant(base_w)) = base_w + quant_noise, 
a.k.a.
    qlora_w = base_w + quant_noise + lora_a @ lora_b, 
thus, performance of QLoRA is lagged behind by quantization noise. 

On the other hand, if we merge lora adaptor into the base model, that is
    merge_w = base_w + lora_a @ lora_b, 
it performs even worse, because LoRA was trained for quantized model, not base model.  

4) a better merge solution
To improve QLoRA-Merge process, we propose a new method: interpolation bwtween base and qlora weights.

Formally, we have:
    merge_w = a * qlora_w + (1-a) * base_w = base_w + a * (quant_noise + lora_a @ lora_b)
where 0 < a < 1 (default: 0.5) controls the percentage of fine-tuned weight. 

By doing so, we can reduce the effect of quantization noise, 
and alleviate overfitting of QLoRA fine-tuning (as free lunch). 
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes.functional import quantize_4bit, dequantize_4bit


@torch.no_grad()
def qlora_merge(
    model_path, 
    adapter_path, 
    bnb_4bit_compute_dtype, 
    bnb_4bit_use_double_quant, 
    bnb_4bit_quant_type,
    merge_alpha = 0.5,
):
    assert bnb_4bit_quant_type in ['fp4', 'nf4'], f"4-bit quantization data type {bnb_4bit_quant_type} is not implemented."
    assert merge_alpha >= 0. and merge_alpha <= 1.
    
    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=bnb_4bit_compute_dtype,
    )
    
    # load adaptor config
    adaptor_config = json.load(open(Path(adapter_path) / 'adapter_config.json'))
    assert adaptor_config['peft_type'] == 'LORA'

    lora_rank = int(adaptor_config['r'])
    lora_alpha = adaptor_config['lora_alpha']
    lora_alpha = lora_alpha / lora_rank
    fan_in_fan_out = adaptor_config['fan_in_fan_out']

    modules_to_save = adaptor_config['modules_to_save']
    has_save_modules = modules_to_save is not None

    target_modules = adaptor_config['target_modules']
    assert target_modules is not None
    
    # load adaptor weight
    if (Path(adapter_path) / 'adapter_model.bin').exists():
        checkpoint = torch.load(Path(adapter_path) / 'adapter_model.bin', map_location='cpu')
    else:
        FileNotFoundError(f"Could not find adaptor checkpoint at {adapter_path}; please ensure it contains a `adapter_model.bin` file.")

    for name, module in model.named_modules():
        # assume the name pattern is something like model.layers.0.self_attn.q_proj
        if has_save_modules and any(name.endswith(f"{module_to_save}") for module_to_save in modules_to_save):
            key_name = f"base_model.model.{name}.weight"
            adaptor_w = checkpoint.pop(key_name).to(dtype=bnb_4bit_compute_dtype)
            if module.weight.data.shape == adaptor_w.shape:
                module.weight.data.copy_(adaptor_w)
            else:
                # shape can be different, such as expanded token embedding
                module.weight.data = adaptor_w.to(device='cpu')

            if hasattr(module, "bias") and module.bias is not None:
                key_name = f"base_model.model.{name}.bias"
                adaptor_b = checkpoint.pop(key_name).to(dtype=bnb_4bit_compute_dtype)
                module.bias.data.copy_(adaptor_b)

        elif any(name.endswith(f"{target_module}") for target_module in target_modules):
            base_w = module.weight.data.to(device='cuda', dtype=bnb_4bit_compute_dtype)
            assert base_w.shape[0] == module.out_features and base_w.shape[1] == module.in_features
            
            # quant and dequant
            quant_w, quant_state = quantize_4bit(
                base_w, 
                compress_statistics=bnb_4bit_use_double_quant, 
                quant_type=bnb_4bit_quant_type,
            )
            dequant_w = dequantize_4bit(
                quant_w, 
                quant_state=quant_state, 
                quant_type=bnb_4bit_quant_type,
            ).to(bnb_4bit_compute_dtype)
            
            # lora weight
            lora_a_name = f"base_model.model.{name}.lora_A.weight"
            lora_b_name = f"base_model.model.{name}.lora_B.weight"
            lora_a_w = checkpoint.pop(lora_a_name).to(device='cuda', dtype=torch.float32)
            lora_b_w = checkpoint.pop(lora_b_name).to(device='cuda', dtype=torch.float32)
            assert lora_b_w.shape[1] == lora_rank and lora_a_w.shape[0] == lora_rank

            lora_w = (lora_b_w @ lora_a_w) * lora_alpha
            if fan_in_fan_out:
                lora_w = lora_w.T
            lora_w = lora_w.to(dtype=bnb_4bit_compute_dtype)

            # merge
            merge_w = merge_alpha * (dequant_w + lora_w) + (1 - merge_alpha) * base_w
            module.weight.data.copy_(merge_w)
    
    # final check
    assert len(checkpoint) == 0, f"There are some parameters left: {list(checkpoint.keys())}."
    
    return model


if __name__ == "__main__":
    # config
    model_path = 'huggyllama/llama-7b'
    adapter_path = 'timdettmers/guanaco-7b'

    # merge model and adaptor
    model = qlora_merge(
        model_path, 
        adapter_path, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        merge_alpha=0.5,
    )

    # move to gpu
    model.to(device='cuda')

    # test
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "What is deep learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

