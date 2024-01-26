"""
This script is to detect possible qzeros overflow in AutoGPTQ zeropoint checkpoint. 
The risk is caused by "zeros -= 1" before packing to int32 format, 
    see https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_triton.py#L127
For example, in int4 packing, zeros of [0, 1, 2, 3, -1, 5, 6, 7] will become [0, 1, 2, 3, 15, 15, 15, 15] after pack and unpack operations.

In most GPTQ checkpoints, such as https://huggingface.co/TheBloke, it used symmetric quantization, where zeros are all (maxq + 1) / 2 - 1. 
It avoids the risk of overflow, however, symmetric quantizatioin usually performs worse than zeropoint quantization, as suggested in GPTQ paper. 

Besides, for other zeropoint quantized checkpoints, such as AWQ, 
the risk of overflow makes it challenging to convert AWQ-stype checkpoint into GPTQ-stype to use efficient GPTQ kernels, such as exllamav2.
While AutoAWQ supports this kind of conversion, 
    see https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/exllamav2.py
the performance is very poor because of the overflow problem. 
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load

def unpack_zeros(qzeros, bits=4):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :])
    izeros = izeros.view(izeros.shape[0], -1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    return izeros

def detect_gptq_overflow(checkpoint):
    # quant config
    quant_config = json.load(open(Path(checkpoint) / 'quantize_config.json'))
    wbits = quant_config['bits']
    groupsize = quant_config['group_size']
    sym = quant_config['sym']
    print(f"GPTQ: sym={sym}, bits={wbits}, groupsize={groupsize}")
    # load checkpoint
    if (Path(checkpoint) / 'model.safetensors').exists():
        checkpoint = safe_load(Path(checkpoint) / 'model.safetensors')
    elif (Path(checkpoint) / f'gptq_model-{wbits}bit-{groupsize}g.safetensors').exists():
        checkpoint = safe_load(Path(checkpoint) / f'gptq_model-{wbits}bit-{groupsize}g.safetensors')
    else:
        raise FileNotFoundError(f"Could not find model checkpoint at {checkpoint}; please ensure it contains a `model.safetensors` file.")
    # detect overflow
    maxq = 2 ** wbits - 1
    sym_zp = (maxq + 1) / 2 - 1
    num_overflow = 0
    for name in list(checkpoint.keys()):
        if "qzeros" in name:
            izeros = unpack_zeros(checkpoint[name], wbits)
            #print(f"{name}: izeros min={izeros.min()}, max={izeros.max()}")
            if not sym: 
                # izeros should be in the range of [-1, maxq-1] with "izeros -= 1"
                # overflow happens if izeros = -1, and it will become maxq after pack-and-unpack
                izeros = izeros.view(-1, 32 // wbits)  #(pack_num, pack_size)
                pack_idx = ((izeros == maxq).sum(dim=1) > 0)
                if pack_idx.sum() > 0:
                    num_overflow += 1
                    print(f"Detect qzeros overflow in {name}:\n{izeros[pack_idx]}")
            else:
                # izeros should be (maxq + 1) / 2 - 1 with "izeros -= 1"
                # overflow will never happen for symmetric quantization
                assert (~(izeros == sym_zp)).sum().item() == 0
    if num_overflow > 0:
        print(f"Overflow is detected in {num_overflow} qlinear layers.")
    else:
        print("No overflow is detected.")

def detect_awq_overflow(checkpoint):
    # quant config
    quant_config = json.load(open(Path(checkpoint) / 'quant_config.json'))
    wbits = quant_config['w_bit']
    zp = quant_config['zero_point']
    assert quant_config['version'] == 'GEMM', "Only GEMM version is supported"
    assert (wbits == 4) and zp, "Only int4 zeropoint AWQ is supported"
    groupsize = quant_config['q_group_size']
    print(f"AWQ: sym={not zp}, bits={wbits}, groupsize={groupsize}")
    # load checkpoint
    if (Path(checkpoint) / 'model.safetensors').exists():
        checkpoint = safe_load(Path(checkpoint) / 'model.safetensors')
    else:
        raise FileNotFoundError(f"Could not find model checkpoint at {checkpoint}; please ensure it contains a `model.safetensors` file.")
    # detect overflow
    num_overflow = 0
    for name in list(checkpoint.keys()):
        if "qzeros" in name:
            izeros = unpack_zeros(checkpoint[name], wbits)
            #print(f"{name}: izeros min={izeros.min()}, max={izeros.max()}")

            # izeros should be in the range of [0, maxq]
            # if any value of izeros is 0, it will become -1 for GPTQ-style packing
            izeros -= 1
            izeros = izeros.view(-1, 32 // wbits) #(pack_num, pack_size)
            pack_idx = ((izeros < 0).sum(dim=1) > 0)
            if pack_idx.sum() > 0:
                num_overflow += 1
                print(f"Detect qzeros overflow in {name}:\n{izeros[pack_idx]}")
    if num_overflow > 0:
        print(f"Overflow is detected in {num_overflow} qlinear layers.")
    else:
        print("No overflow is detected.")

if __name__ == "__main__":
    # (1) detect overflow in gptq-style zeropoint checkpoint
    checkpoint = "/mnt/data/llama2-7b-chat-gptq-zp"
    detect_gptq_overflow(checkpoint)

    # (2) detect overflow in awq-style zeropoint checkpoint
    #checkpoint = "/mnt/data/llama2-7b-chat-awq-zp"
    #detect_awq_overflow(checkpoint)

