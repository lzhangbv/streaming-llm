import torch

import triton
import triton.language as tl

@triton.jit
def _dequant_matmul_kernel(
    a_ptr, b_ptr, c_ptr,   #pointers to matrices
    scales_ptr, zeros_ptr, #pointers to scales and zeros
    M, N, K,               #matrix dimensions
    stride_am, stride_ak,  
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, 
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

    # pointers for the first blocks of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    # it repeats elements along the K axis 8 times, as we pack 8-int4 to 1-int32 in the matrix of B
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    # pointers for the first blocks of scales and zeros
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)   # (BLOCK_SIZE_N,)

    # shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    # calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        # load a and b
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)                          # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        # load scales and zeros
        g_id = k // (groupsize // BLOCK_SIZE_K)
        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)   # (BLOCK_SIZE_N,)

        # unpack b and zeros
        b = (b >> shifter[:, None]) & 0xF
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) # as auto-gptq stores the values of zeros-1

        # dequant b
        b = (b - zeros[None, :]) * scales[None, :]

        # matmul
        accumulator += tl.dot(a, b)

        # update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk


    # store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def w4a16_matmul(x, qweight, scales, qzeros, group_size):
    block_size_m=16
    block_size_n=16
    block_size_k=64

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
    _dequant_matmul_kernel[grid](
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
        GROUP_SIZE_M=8,
        num_warps=2, num_stages=4,
    )
    return c

def torch_w4a16_matmul(x, qweight, scales, qzeros, group_size):
    # unpack qweight
    qweight = torch.repeat_interleave(qweight, dim=0, repeats=8)  #(K//8, N) -> (K, N)
    K = qweight.shape[0]
    shifter = torch.arange(0, K, device=qweight.device, dtype=torch.int32).reshape(-1, 1) #(K, 1)
    shifter = (shifter % 8) * 4
    qweight = (qweight >> shifter) & 0xF
    # unpack qzeros and scales
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
    M, N, K = 16, 4096, 4096
    group_size = 128

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    qweight = torch.randint(low=-2147483648, high=2147483647, size=(K//8, N), device='cuda', dtype=torch.int32)
    scales = torch.randn((K//group_size, N), device='cuda', dtype=torch.float16)
    qzeros = torch.randint(low=-2147483648, high=2147483647, size=(K//group_size, N//8), device='cuda', dtype=torch.int32)

    triton_output = w4a16_matmul(a, qweight, scales, qzeros, group_size)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size)

    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

