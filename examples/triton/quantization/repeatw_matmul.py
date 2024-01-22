import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _repeat8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,   #pointers to matrices
    M, N, K,               #matrix dimensions
    repeat,                #repeat=8 by default
    stride_am, stride_ak,  
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x repeat(B, 8).
    A is of shape (M, K) float16
    B is of shape (K//8, N) float16 or float32
    C is of shape (M, N) float16
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
    b_ptrs = b_ptr + ((offs_k[:, None] // repeat) * stride_bk + offs_bn[None, :] * stride_bn)

    # calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        # load a and b
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)                          # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        # matmul
        b = b.to(tl.float16)
        accumulator += tl.dot(a, b)

        # update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // repeat) * stride_bk

    # store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def repeat8_matmul(x, qweight, repeat):
    M, K = x.shape
    N = qweight.shape[1]

    # shape constraints
    assert x.shape[-1] == (qweight.shape[0] * repeat), "Incompatible dimensions"
    assert x.is_contiguous(), "A must be contiguous"

    # allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    _repeat8_matmul_kernel[grid](
        x, qweight, c,
        M, N, K,
        repeat,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def torch_repeat8_matmul(x, qweight, repeat=8):
    # unpack qweight
    weight = torch.repeat_interleave(qweight, dim=0, repeats=repeat)  #(K//8, N) -> (K, N)
    output = torch.matmul(x, weight.to(x.dtype))
    return output

if __name__ == "__main__":
    # It is to study the performance bound of w4a16 matmul operator w/o dequant
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        torch.manual_seed(0)
        N, K = 4096, 4096
        repeat, dtype = 8, torch.float32 #w4a16 simulation w/o scale and zero
        #repeat, dtype = 1, torch.float16 #normal fp16 matmul

        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        qweight = torch.randn((K//repeat, N), device='cuda', dtype=dtype)

        triton_output = repeat8_matmul(a, qweight, repeat)
        torch_output = torch_repeat8_matmul(a, qweight, repeat)
        #print(f"triton_output={triton_output}")
        #print(f"torch_output={torch_output}")
        print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

        # benchmark
        print(f"\nBenchmark with bs={M}.")
        print("Time (ms):", triton.testing.do_bench(lambda: repeat8_matmul(a, qweight, repeat)))

