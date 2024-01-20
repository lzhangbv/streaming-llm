import torch

import triton
import triton.language as tl


@triton.jit
def _zp_dequant_kernel(
    Q, Out, 
    scales_ptr, zeros_ptr, 
    stride_qk, stride_qn,  
    stride_ok, stride_on,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, 
    BLOCK_SIZE_N: tl.constexpr, 
):
    """
    Dequant qweight to output matrix. 
    Q is of shape (K//8, N) int32
    Out is of shape (K, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    zeros is of shape (G, N//8) int32
    """
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    gid = pid_k // groupsize

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # pointers
    offs_q = (pid_k // 8) * stride_qk + offs_n * stride_qn
    offs_scales = gid * stride_scales_g + offs_n * stride_scales_n
    offs_zeros = gid * stride_zeros_g + (offs_n // 8) * stride_zeros_n

    # shifter
    shifter = (pid_k % 8) * 4
    zeros_shifter = (offs_n % 8) * 4

    # load
    weight = tl.load(Q + offs_q)
    scales = tl.load(scales_ptr + offs_scales)
    zeros = tl.load(zeros_ptr + offs_zeros)

    # unpack weight and zeros
    weight = (weight >> shifter) & 0xF
    zeros = (zeros >> zeros_shifter) & 0xF
    zeros = (zeros + 1)

    # dequant weight
    weight = (weight - zeros) * scales

    # store the result
    offs_o = pid_k * stride_ok + offs_n * stride_on
    tl.store(Out + offs_o, weight)

@triton.jit
def _sym_dequant_kernel(
    Q, Out, 
    scales_ptr, 
    ZERO,  
    stride_qk, stride_qn,  
    stride_ok, stride_on,
    stride_scales_g, stride_scales_n,
    groupsize, 
    BLOCK_SIZE_N: tl.constexpr, 
):
    """
    Dequant qweight to output matrix. 
    Q is of shape (K//8, N) int32
    Out is of shape (K, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    ZERO is 8, where 2 ** (bits-1) = 8
    """
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    gid = pid_k // groupsize

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # pointers
    offs_q = (pid_k // 8) * stride_qk + offs_n * stride_qn
    offs_scales = gid * stride_scales_g + offs_n * stride_scales_n

    # shifter
    shifter = (pid_k % 8) * 4

    # load
    weight = tl.load(Q + offs_q)
    scales = tl.load(scales_ptr + offs_scales)

    # unpack weight and zeros
    weight = (weight >> shifter) & 0xF

    # dequant weight
    weight = (weight - ZERO) * scales

    # store the result
    offs_o = pid_k * stride_ok + offs_n * stride_on
    tl.store(Out + offs_o, weight)

def w4a16_matmul(x, w, qweight, scales, qzeros, group_size, sym=False):
    block_size_n=128
    K = x.shape[1]
    N = qweight.shape[1]

    # shape constraints
    assert x.shape[-1] == (qweight.shape[0] * 8), "Incompatible dimensions"
    assert x.shape[-1] == w.shape[0], "Incompatible dimensions"
    assert w.shape[-1] == qweight.shape[-1], "Incompatible dimensions"
    assert K % group_size == 0, "K must be a multiple of group size"
    assert N % block_size_n == 0, "N must be a multiple of block_size_n"

    grid = (K, N // block_size_n)

    # dequant qweight to w
    if sym:
        zeropoint = 8
        _sym_dequant_kernel[grid](
            qweight, w, 
            scales, zeropoint,
            qweight.stride(0), qweight.stride(1),
            w.stride(0), w.stride(1),
            scales.stride(0), scales.stride(1),
            group_size,
            block_size_n,
            num_warps=2, num_stages=4,
        )
    else:
        _zp_dequant_kernel[grid](
            qweight, w, 
            scales, qzeros,
            qweight.stride(0), qweight.stride(1),
            w.stride(0), w.stride(1),
            scales.stride(0), scales.stride(1),
            qzeros.stride(0), qzeros.stride(1),
            group_size,
            block_size_n,
            num_warps=2, num_stages=4,
        )
    c = torch.matmul(x, w)
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

    print("Zeropoint quantization.")
    triton_output = w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=False)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    print("\nSymmetric quantization.")
    triton_output = w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=True)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    # benchmark
    print(f"\nBenchmark with bs={M}.")
    print("fp16:", triton.testing.do_bench(lambda: torch.matmul(a, ref_weight)))
    print("torch zp:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)))
    print("triton zp:", triton.testing.do_bench(lambda: w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=False)))
    print("torch sym:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)))
    print("triton sym:", triton.testing.do_bench(lambda: w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=True)))

