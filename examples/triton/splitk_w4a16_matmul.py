import torch

import triton
from triton import language as tl


@triton.jit()
def _zp_dequant_matmul_kernel(
        a_ptr, b_ptr, c_ptr, 
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
        group_m: tl.constexpr, split_k: tl.constexpr
    ):
    """
    Compute the matrix multiplication C = A x dequant(B).
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    zeros is of shape (G, N//8) int32
    """

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_k = tl.cdiv(K, block_k*split_k)

    # re-order program ID for better L2 performance
    grid_m = tl.cdiv(M, block_m)
    grid_n = tl.cdiv(N, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    # offsets    
    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    # it tells the compiler that these elements are contiguous in memory
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, num_pid_k):
        # load a and b
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # load scales and zeros
        g_id = (k * split_k + pid_k) // (groupsize // block_k)
        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        # unpack b and zeros
        b = (b >> shifter[:, None]) & 0xF
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1)

        # dequant b
        b = (b - zeros[None, :]) * scales[None, :]

        acc += tl.dot(a, b)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    acc.to(tl.float16)

    offs_cm = pid_m*block_m + tl.arange(0, block_m)
    offs_cn = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)

@triton.jit()
def _sym_dequant_matmul_kernel(
        a_ptr, b_ptr, c_ptr, 
        scales_ptr, 
        ZERO,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        groupsize,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
        group_m: tl.constexpr, split_k: tl.constexpr
    ):
    """
    Compute the matrix multiplication C = A x dequant(B).
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    ZERO is 8 for symmetric quantization, where 2**(bits-1)=8
    """

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_k = tl.cdiv(K, block_k*split_k)

    # re-order program ID for better L2 performance
    grid_m = tl.cdiv(M, block_m)
    grid_n = tl.cdiv(N, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    # offsets    
    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    # it tells the compiler that these elements are contiguous in memory
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n

    shifter = (offs_k % 8) * 4

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, num_pid_k):
        # load a and b
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # load scales and zeros
        g_id = (k * split_k + pid_k) // (groupsize // block_k)
        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        # unpack b and zeros
        b = (b >> shifter[:, None]) & 0xF

        # dequant b
        b = (b - ZERO) * scales[None, :]

        acc += tl.dot(a, b)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    acc.to(tl.float16)

    offs_cm = pid_m*block_m + tl.arange(0, block_m)
    offs_cn = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)

def splitk_w4a16_matmul(x, qweight, scales, qzeros, group_size, sym=False):
    """
    The split-k optimization is based on: 
        https://github.com/openai/triton/blob/main/python/triton/ops/matmul.py
    """
    block_size_m=16
    block_size_n=32
    block_size_k=128
    group_m = 8
    split_k = 4

    M, K = x.shape
    N = qweight.shape[1]

    # shape constraints
    assert x.shape[-1] == (qweight.shape[0] * 8), "Incompatible dimensions"
    assert x.is_contiguous(), "A must be contiguous"
    assert K % (block_size_k * split_k) == 0, "K must be a multiple of block_size_k * split_k"
    assert K % group_size == 0, "K must be a multiple of group size"
    assert N % block_size_n == 0, "N must be a multiple of block_size_n"
    assert group_size % block_size_k == 0, "Group size must be a multiple of block_size_k"

    # allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['block_m']) * triton.cdiv(N, META['block_n']), META['split_k'])

    if sym:
        zeropoint = 8
        _sym_dequant_matmul_kernel[grid](
            x, qweight, c,
            scales,
            zeropoint,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            group_size,
            block_size_m, block_size_n, block_size_k,
            group_m, split_k,
            num_warps=4, num_stages=3,
        )
    else:
        _zp_dequant_matmul_kernel[grid](
            x, qweight, c,
            scales, qzeros,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            qzeros.stride(0), qzeros.stride(1),
            group_size,
            block_size_m, block_size_n, block_size_k,
            group_m, split_k,
            num_warps=4, num_stages=3,
        )
    return c

def torch_w4a16_matmul(x, qweight, scales, qzeros, group_size, sym=False):
    # unpack qweight
    qweight = torch.repeat_interleave(qweight, dim=0, repeats=8)  #(K//8, N) -> (K, N)
    K = qweight.shape[0]
    shifter = torch.arange(0, K, device=qweight.device, dtype=torch.int32).reshape(-1, 1) #(K, 1)
    shifter = (shifter % 8) * 4
    qweight = (qweight >> shifter) & 0xF
    # unpack qzeros and scales
    if sym:
        qzeros = 8
    else:
        qzeros = torch.repeat_interleave(qzeros, dim=1, repeats=8) #(K/g, N/8) -> (K/g, N)
        N = qzeros.shape[1]
        shifter = torch.arange(0, N, device=qzeros.device, dtype=torch.int32).reshape(1, -1) #(1, N)
        shifter = (shifter % 8) * 4
        qzeros = (qzeros >> shifter) & 0xF
        qzeros = qzeros + 1
        qzeros = torch.repeat_interleave(qzeros, dim=0, repeats=group_size) #(K/g, N) -> (K, N)
    scales = torch.repeat_interleave(scales, dim=0, repeats=group_size) #(K/g, N) -> (K, N)
    # dequant and matmul
    weight = (qweight - qzeros) * scales
    output = torch.matmul(x, weight.to(x.dtype))
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1, 4096, 4096
    group_size = 128

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    qweight = torch.randint(low=-2147483648, high=2147483647, size=(K//8, N), device='cuda', dtype=torch.int32)
    scales = torch.randn((K//group_size, N), device='cuda', dtype=torch.float16)
    qzeros = torch.randint(low=-2147483648, high=2147483647, size=(K//group_size, N//8), device='cuda', dtype=torch.int32)
    ref_weight = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # test
    print("Zeropoint quantization.")
    triton_output = splitk_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    print("\nSymmetric quantization.")
    triton_output = splitk_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    # benchmark
    print(f"\nBenchmark with bs={M}.")
    print("fp16:", triton.testing.do_bench(lambda: torch.matmul(a, ref_weight)))
    print("torch zp:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)))
    print("triton zp:", triton.testing.do_bench(lambda: splitk_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)))
    print("torch sym:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)))
    print("triton sym:", triton.testing.do_bench(lambda: splitk_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)))

