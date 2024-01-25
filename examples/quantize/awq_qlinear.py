import math
import json
from pathlib import Path
import time

import torch
import torch.nn as nn

import triton
import triton.language as tl

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_awq_model(checkpoint, device='cuda'):
    # quant config
    quant_config = json.load(open(Path(checkpoint) / 'quant_config.json'))
    wbits = quant_config['w_bit']
    assert quant_config['version'] == 'GEMM', "Only GEMM version is supported"
    assert wbits == 4, "Only int4 AWQ is supported"
    groupsize = quant_config['q_group_size']
    print(f"AWQ: bits={wbits}, groupsize={groupsize}")
    # model config
    config = AutoConfig.from_pretrained(checkpoint)
    # build model: weights are zeros/ones, fp16, cpu
    def skip(*args, **kwargs):
        pass
    old_init = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = skip, skip, skip
    old_init_weights = transformers.modeling_utils._init_weights
    transformers.modeling_utils._init_weights = False

    torch.set_default_dtype(torch.float16)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float32)

    (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_) = old_init
    transformers.modeling_utils._init_weights = old_init_weights
    # make quant
    make_quant(model, wbits, groupsize)
    #print(model.state_dict().keys())
    # load checkpoint
    print('Loading model ...')
    t0 = time.time()
    if (Path(checkpoint) / 'model.safetensors').exists():
        from safetensors.torch import load_file as safe_load
        checkpoint = safe_load(Path(checkpoint) / 'model.safetensors')
        for name in list(checkpoint.keys()):
            if "rotary_emb.inv_freq" in name:
                del checkpoint[name]
            if "qweight" in name:
                #checkpoint[name] = _convert_awq_qweight(checkpoint[name], wbits)
                checkpoint[name] = _fast_convert_awq_qweight(checkpoint[name], wbits)
            if "qzeros" in name:
                #checkpoint[name] = _convert_awq_qzeros(checkpoint[name], wbits)
                checkpoint[name] = _fast_convert_awq_qzeros(checkpoint[name], wbits)
        model.load_state_dict(checkpoint, strict=False) #missing g_idx in checkpoints
    else:
        raise FileNotFoundError(f"Could not find model checkpoint at {checkpoint}; please ensure it contains a `model.safetensors` file.")
    print(f"Time to load: {time.time()-t0:.02f}")
    # remove zero bias
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            if m.bias is not None and (m.bias == 0).all():
                m.bias = None
                #print(f"Removed bias from {name}")
    # to device
    model = model.to(device)
    model.eval()
    return model

def _convert_awq_qweight(awq_qweight, bits=4):
    assert bits == 4, "Only 4-bit AWQ is supported"
    assert awq_qweight.dtype == torch.int32
    # unpack
    maxq = 2 ** bits - 1
    awq_qweight = torch.repeat_interleave(awq_qweight, dim=1, repeats=8)  #(K, N//8) -> (K, N)
    N = awq_qweight.shape[1]
    shifter = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=awq_qweight.device, dtype=torch.int32).reshape(1, -1)
    shifter = shifter.repeat(1, N//8) #(1, 8) -> (1, N)
    shifter = shifter * bits
    intweight = (awq_qweight >> shifter) & maxq #(K, N)
    # pack
    K = intweight.shape[0]
    assert K % 8 == 0
    qweight = torch.zeros((intweight.shape[0] // 8, intweight.shape[1]), dtype=torch.int32)
    i = 0
    row = 0
    while row < qweight.shape[0]:
        for j in range(i, i + (32 // bits)):
            qweight[row] |= intweight[j] << (bits * (j - i))
        i += 32 // bits
        row += 1

    return qweight

def _convert_awq_qzeros(awq_qzeros, bits=4):
    assert bits == 4, "Only 4-bit AWQ is supported"
    assert awq_qzeros.dtype == torch.int32
    # unpack
    maxq = 2 ** bits - 1
    awq_qzeros = torch.repeat_interleave(awq_qzeros, dim=1, repeats=8)  #(K//g, N//8) -> (K//g, N)
    N = awq_qzeros.shape[1]
    shifter = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=awq_qzeros.device, dtype=torch.int32).reshape(1, -1)
    shifter = shifter.repeat(1, N//8) #(1, 8) -> (1, N)
    shifter = shifter * bits
    zeros = (awq_qzeros >> shifter) & maxq #(K//g, N)
    # Note: we remove "zeros -= 1" as it could make big errors
    # pack
    qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 8), dtype=torch.int32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + (32 // bits)):
            qzeros[:, col] |= zeros[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    return qzeros

def _fast_convert_awq_qweight(awq_qweight, bits=4):
    assert bits == 4, "Only 4-bit AWQ is supported"
    assert awq_qweight.dtype == torch.int32
    # unpack
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=awq_qweight.device, dtype=torch.int32) * bits
    iweight = torch.bitwise_right_shift(awq_qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweight = iweight.view(iweight.shape[0], -1)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    # pack
    shifts = torch.arange(0, 32, bits, device=iweight.device)
    iweight = iweight.view(iweight.shape[0] // (32 // bits), 32 // bits, -1)
    qweight = (torch.bitwise_left_shift(iweight, shifts[None, :, None]).sum(dim=1).to(torch.int32))
    return qweight

def _fast_convert_awq_qzeros(awq_qzeros, bits=4):
    assert bits == 4, "Only 4-bit AWQ is supported"
    assert awq_qzeros.dtype == torch.int32
    # unpack
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=awq_qzeros.device, dtype=torch.int32) * bits
    izeros = torch.bitwise_right_shift(awq_qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    izeros = izeros.view(izeros.shape[0], -1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    # Note: we remove "izeros -= 1" as it could make big errors
    # pack
    shifts = torch.arange(0, 32, bits, device=izeros.device)
    izeros = izeros.view(-1, izeros.shape[1] // (32 // bits), 32 // bits)
    qzeros = (torch.bitwise_left_shift(izeros, shifts[None, None, :]).sum(dim=-1).to(torch.int32))
    return qzeros

def make_quant(model, bits=4, groupsize=128):
    """
    Replace linear layers in a model with qlinear layers. 
    Except for the lm_head. 
    """
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        # replace
        qlayer = QuantLinear(bits, groupsize, m.in_features, m.out_features, m.bias is not None)
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)
        setattr(parent, name[len(parent_name) + 1:], qlayer)

class QuantLinear(nn.Module):
    def __init__(self, bits: int, groupsize: int, infeatures: int, outfeatures: int, bias: bool):
        super().__init__()

        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        
        groupsize = infeatures if groupsize == -1 else groupsize
        
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize

        features_per_int = 32 // bits

        assert outfeatures % features_per_int == 0, "outfeatures must be a multiple of features_per_int"

        self.register_buffer('qweight', torch.empty((infeatures // features_per_int, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.empty((math.ceil(infeatures / groupsize), outfeatures // features_per_int), dtype=torch.int32))
        self.register_buffer('scales', torch.empty((math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // groupsize for i in range(infeatures)], dtype=torch.int32)) # it is needed for Act-Order
        self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16)) # remove later if values are zero

    def forward(self, x):
        y = w4a16_matmul(x, self.qweight, self.scales, self.qzeros, self.g_idx, self.groupsize)
        if self.bias is not None:
            y = y + self.bias
        return y

@triton.jit
def _zp_dequant_matmul_kernel(
    a_ptr, b_ptr, c_ptr,   #pointers to matrices
    scales_ptr, zeros_ptr, #pointers to scales and zeros
    g_ptr,                 #group ids for in-feature rows
    M, N, K,               #matrix dimensions
    stride_am, stride_ak,  
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x dequant(B).
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    zeros is of shape (G, N//8) int32
    g_ptr is of shape (K) int32
    """
    # map prograim id to the block of C(m, n)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + (
        (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // 8)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & 0xF
        # Note: we remove "zeros = (zeros + 1)" as it is not needed for auto-awq checkpoint

        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & 0xF  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def w4a16_matmul(a, qweight, scales, qzeros, g_idx, group_size):
    block_size_m=16
    block_size_n=32
    block_size_k=128

    x = a.view(-1, a.shape[-1])
    M, K = x.shape
    N = qweight.shape[1]

    # shape constraints
    assert x.shape[-1] == (qweight.shape[0] * 8), "Incompatible dimensions"
    assert x.is_contiguous(), "A must be contiguous"
    assert K % block_size_k == 0, "K must be a multiple of block_size_k"
    assert K % group_size == 0, "K must be a multiple of group size"
    assert N % block_size_n == 0, "N must be a multiple of block_size_n"
    assert group_size % block_size_k == 0, "Group size must be a multiple of block_size_k"

    # allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    _zp_dequant_matmul_kernel[grid](
        x, qweight, c,
        scales, qzeros, g_idx, 
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), 
        qzeros.stride(0),
        block_size_m, block_size_n, block_size_k,
        GROUP_SIZE_M=8,
        num_warps=4, num_stages=3,
    )
    c = c.view(a.shape[:-1] + (N,))
    return c

if __name__ == "__main__":
    """
    We convert AWQ checkpoint to GPTQ-stype qweight and qzeros, and then use w4a16 matmul for model inference. 
    Important: we remove qzeros-1 for packing, as well as qzeros+1 for w4a16, to avoid possible conversion errors. 
    """
    # load
    model_id = "/mnt/data/llama2-7b-chat-awq/"
    model = load_awq_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # test
    print("Testing ...")
    prompt = "What is deep learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
