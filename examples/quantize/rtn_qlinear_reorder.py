import time
import functools

import torch
import torch.nn as nn

import triton
import triton.language as tl


def fast_quant_with_reorder(W, act_scale, bits=4, groupsize=128):
    N, K = W.shape
    assert bits == 4
    assert K % groupsize == 0
    assert N % 8 == 0
    # reorder
    perm = torch.argsort(act_scale, descending=True)
    W = W[:, perm]
    invperm = torch.argsort(perm)
    # get scales, zeros, and g_idx
    maxq = 2 ** bits - 1
    W = W.view(-1, groupsize) #(group_num, group_size)
    xmin = torch.clamp(torch.min(W, dim=1)[0], max=0)
    xmax = torch.clamp(torch.max(W, dim=1)[0], min=0)
    scales = (xmax - xmin).clamp(min=1e-5) / maxq
    zeros = torch.round(-xmin / scales)
    g_idx = torch.tensor([i // groupsize for i in range(K)], dtype=torch.int32, device=W.device)
    g_idx = g_idx[invperm]
    # quantize weight
    izeros = zeros.to(torch.int32)
    iweight = torch.clamp(torch.round(W / scales[:, None]) + zeros[:, None], 0, maxq).to(torch.int32)
    iweight = iweight.view(N, K)[:, invperm]
    # pack iweight
    iweight = iweight.t().contiguous() #[K, N]
    shifts = torch.arange(0, 32, bits, device=iweight.device)
    iweight = iweight.view(iweight.shape[0] // (32 // bits), 32 // bits, -1)
    qweight = (torch.bitwise_left_shift(iweight, shifts[None, :, None]).sum(dim=1).to(torch.int32))
    # pack zeros
    scales = scales.view(N, K//groupsize).t().contiguous() #[K//groupsize, N]
    izeros = izeros.view(N, K//groupsize).t().contiguous() #[K//groupsize, N]
    izeros = izeros.view(-1, izeros.shape[1] // (32 // bits), 32 // bits)
    qzeros = (torch.bitwise_left_shift(izeros, shifts[None, None, :]).sum(dim=-1).to(torch.int32))
    return qweight, scales, qzeros, g_idx

def make_quant(model, tokenizer, inputs=None, bits=4, groupsize=128):
    """
    Enable act-reorder for RTN quantization. 
    """
    assert groupsize > 0, "Only group-wise quantization is supported"
    # register hook function
    act_scales = {}
    hooks = []

    def act_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x = x.view(-1, x.shape[-1])
        act_scales[name] = torch.norm(x, dim=0) #gptq style
        #act_scales[name] = torch.max(x.abs(), dim=0)[0] #smoothquant style
    
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        hooks.append(
            m.register_forward_hook(functools.partial(act_hook, name=name))
        )
        
    # get act scales, a.k.a diag(H)
    stime = time.time()
    if inputs is None:
        inputs = "Deep learning is the subset of machine learning methods based on artificial neural networks with representation learning."
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    model(input_ids.to(model.device))
    print(f"Time to get act scales: {time.time()-stime:.02f}")

    # remove hook function
    for h in hooks:
        h.remove()
    
    # quantize with act-reorder
    stime = time.time()
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if 'lm_head' in name:
            continue
        # quantize
        qweight, scales, qzeros, g_idx = fast_quant_with_reorder(m.weight.data, act_scales[name], bits, groupsize)
        # replace
        qlayer = QuantLinear(
            bits, groupsize, m.in_features, m.out_features, 
            qweight, scales, qzeros, g_idx, m.bias
        )
        parent_name = name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)
        setattr(parent, name[len(parent_name) + 1:], qlayer)
    print(f"Time to quantize: {time.time()-stime:.02f}")

class QuantLinear(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures, qweight, scales, qzeros, g_idx, bias):
        super().__init__()

        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize

        features_per_int = 32 // bits

        assert outfeatures % features_per_int == 0, "outfeatures must be a multiple of features_per_int"
        assert infeatures // features_per_int == qweight.shape[0]
        assert infeatures // groupsize == scales.shape[0]
        assert infeatures // groupsize == qzeros.shape[0]
        assert outfeatures // features_per_int == qzeros.shape[1]
        assert g_idx.shape[0] == infeatures

        self.register_buffer('qweight', qweight)
        self.register_buffer('qzeros', qzeros)
        self.register_buffer('scales', scales)
        self.register_buffer('g_idx', g_idx)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

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
    We convert fp16 checkpoint to GPTQ-stype qweight and qzeros (w/o -/+1), and then use w4a16 matmul for model inference. 
    Note: we use the round-to-nearest (rtn) method with act-reorder
    """
    # load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stime = time.time()
    model_id = "/mnt/data/llama2-7b-chat/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Time to load: {time.time()-stime:.02f}")
    make_quant(model, tokenizer)
    model.to(device="cuda")
    # test
    print("Testing ...")
    prompt = "What is deep learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=25)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
