import torch

import triton
import triton.language as tl

@triton.jit
def matmul_silu_kernel(
        # Pointers to matrices
        a_ptr, w1_ptr, w2_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        stride_am, stride_ak,    # input
        stride_w1k, stride_w1n,  # weight 1
        stride_w2k, stride_w2n,  # weight 2
        stride_cm, stride_cn,    # output
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """
       Fused kernel for computing F.silu(w1(x)) * w2(x)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to pid_m and pid_n
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w2_ptrs = w2_ptr + (offs_k[:, None] * stride_w2k + offs_bn[None, :] * stride_w2n)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc1 += tl.dot(a, w1)
        w2 = tl.load(w2_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc2 += tl.dot(a, w2)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w2_ptrs += BLOCK_SIZE_K * stride_w2k

    # -----------------------------------------------------------
    # Fuse silu activation function
    c = (acc1 * tl.sigmoid(acc1)) * acc2

    # -----------------------------------------------------------
    # Write back the block of the output matrix
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_silu(x, w1, w2):
    # Check constraints.
    assert x.shape[-1] == w1.shape[0], "Incompatible dimensions"
    assert x.shape[-1] == w2.shape[0], "Incompatible dimensions"
    assert w1.shape[1] == w2.shape[1], "Incompatible dimensions"

    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim
    N = w1.shape[1]
    x = x.view(M, K)

    # Allocates output.
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_silu_kernel[grid](
        x, w1, w2, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=2, num_warps=4
    )
    out = out.view(batch, seq_len, -1)
    return out

def torch_matmul_silu(x, w1, w2):
    y1 = torch.matmul(x, w1)
    y1 = torch.nn.functional.silu(y1)
    y2 = torch.matmul(x, w2)
    return y1 * y2

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((1, 64, 4096), device='cuda', dtype=torch.float16)
    w1 = torch.randn((4096, 11008), device='cuda', dtype=torch.float16) * 0.2
    w2 = torch.randn((4096, 11008), device='cuda', dtype=torch.float16) * 0.2
    triton_output = matmul_silu(a, w1, w2)
    torch_output = torch_matmul_silu(a, w1, w2)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(torch_output - triton_output))}')


