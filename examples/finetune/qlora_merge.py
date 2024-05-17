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
import transformers
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
    **kwargs,
):
    assert bnb_4bit_quant_type in ['fp4', 'nf4'], f"4-bit quantization data type {bnb_4bit_quant_type} is not implemented."
    assert merge_alpha >= 0. and merge_alpha <= 1.
    
    # skip model init
    def skip(*args, **kwargs):
        pass
    old_init = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = skip, skip, skip
    old_init_weights = transformers.modeling_utils._init_weights
    transformers.modeling_utils._init_weights = False

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=bnb_4bit_compute_dtype,
        **kwargs,
    )

    (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_) = old_init
    transformers.modeling_utils._init_weights = old_init_weights

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
    elif (Path(adapter_path) / 'adapter_model.safetensors').exists():
        from safetensors.torch import load_file as safe_load
        checkpoint = safe_load(Path(adapter_path) / 'adapter_model.safetensors', device='cpu')
    else:
        FileNotFoundError(f"Could not find adaptor checkpoint at {adapter_path}; please ensure it contains a `adapter_model.bin` or `adapter_model.safetensors` file.")

    for name, module in model.named_modules():
        # assume the name pattern is something like model.layers.0.self_attn.q_proj
        if has_save_modules and any(name.endswith(f"{module_to_save}") for module_to_save in modules_to_save):
            key_name = f"base_model.model.{name}.weight"
            adaptor_w = checkpoint.pop(key_name).to(dtype=bnb_4bit_compute_dtype)
            if module.weight.data.shape == adaptor_w.shape:
                weight = merge_alpha * adaptor_w + (1 - merge_alpha) * module.weight
                module.weight.data.copy_(weight)
            else:
                # shape can be different, such as expanded token embedding
                # todo: how to interpolate the expanded token embedding
                if merge_alpha > 0:
                    module.weight.data = adaptor_w.to(device='cpu')

            if hasattr(module, "bias") and module.bias is not None:
                key_name = f"base_model.model.{name}.bias"
                adaptor_b = checkpoint.pop(key_name).to(dtype=bnb_4bit_compute_dtype)
                bias = merge_alpha * adaptor_b + (1 - merge_alpha) *  module.bias
                module.bias.data.copy_(bias)

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
    model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    adapter_path = 'namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA'

    # merge model and adaptor
    model = qlora_merge(
        model_path, 
        adapter_path, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        merge_alpha=0.5,
        rope_theta=200e6, # expand rope base, comment it out if merge_alpha=0
    )

    # move to gpu(s)
    multigpu = True
    if multigpu:
        from accelerate import infer_auto_device_map, dispatch_model
        device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'])
        model = dispatch_model(model, device_map=device_map)
    else:
        model.to(device='cuda')

    # test
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    messages = [{"role": "user", "content": "What is deep learning?"}]

    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", 
    ).to(model.device)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators)
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

