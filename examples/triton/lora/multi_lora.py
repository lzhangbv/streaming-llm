import torch

import triton
import triton.language as tl

@triton.jit
def _multi_lora_kernel(
    A, Bs, C,
    b_start_loc, b_seq_len, 
    M_max, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr,
):
    # pid_m, pid_n
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_max, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # group_id
    gid = tl.program_id(axis=1)
    B_ptr = tl.load(Bs + gid).to(tl.pointer_type(tl.float16))
    m_start_index = tl.load(b_start_loc + gid)
    m_seq_len = tl.load(b_seq_len + gid)

    # offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_max
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    need_store = tl.where(pid_m * BLOCK_SIZE_M >= m_seq_len, 0, 1)
    for _ in range(0, need_store, 1):
        a_ptrs = A + (m_start_index + offs_am[:, None]) * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float16)
        
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C +  (m_start_index + offs_cm[:, None]) * stride_cm +  offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < m_seq_len) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

def multi_lora_matmul(A, group_B, b_start_loc, b_seq_len):
    """
    C[si:si+mi] = A[si:si+mi] @ group_B[i]
        A: batched inputs, where i-th input is [mi, K]
        group_B: grouped weights, each B has same shape of [K, N] 
        b_start_loc, b_seq_len: start locations and sequence lengths for batched inputs
    """
    M, K = A.shape
    N = group_B[0].shape[1]
    group_size = len(group_B)
    B_ptrs = []
    for i in range(group_size):
        assert group_B[i].shape[0] == K
        assert group_B[i].shape[1] == N
        B_ptrs.append(group_B[i].data_ptr())

    assert b_start_loc.shape[0] == group_size
    assert b_seq_len.shape[0] == group_size
    M_max = b_seq_len.max().item()

    # allocate output
    Out = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    Bs = torch.tensor(B_ptrs, device=A.device)

    # 2D launch kernel
    grid = lambda META: (triton.cdiv(M_max, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), group_size)
    _multi_lora_kernel[grid](
        A, Bs, Out,
        b_start_loc, b_seq_len, 
        M_max, N, K, 
        A.stride(0), A.stride(1),
        group_B[0].stride(0), group_B[0].stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_stages=5,
        num_warps=2,
    )
    return Out

def torch_multi_lora_matmul(A, group_B, b_start_loc, b_seq_len):
    M = A.shape[0]
    N = group_B[0].shape[1]
    group_size = len(group_B)

    out = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    for i in range(group_size):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        out[start:end] = torch.matmul(A[start:end], group_B[i])
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    # LoRAs
    shrink = True
    r = 32
    dim = 4096
    group_size = 100
    print(f"group size: {group_size}")
    if shrink:
        # LoRA shrink
        K, N = dim, r
    else:
        # LoRA expand
        K, N = r, dim
    # Inputs
    b_seq_len = torch.randint(low=1, high=5, size=(group_size,), dtype=torch.int32, device="cuda")
    b_start_loc = torch.cumsum(b_seq_len, dim=0) - b_seq_len
    M = b_seq_len.sum().item()
    A = torch.randn((M, K), dtype=torch.float16, device="cuda")
    group_B = [torch.randn((K, N), dtype=torch.float16, device="cuda") for _ in range(group_size)]
    # Comparison
    torch_output = torch_multi_lora_matmul(A, group_B, b_start_loc, b_seq_len)
    triton_output = multi_lora_matmul(A, group_B, b_start_loc, b_seq_len)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')
    # Benchmark
    print('torch:', triton.testing.do_bench(lambda: torch_multi_lora_matmul(A, group_B, b_start_loc, b_seq_len)))
    print('triton:', triton.testing.do_bench(lambda: multi_lora_matmul(A, group_B, b_start_loc, b_seq_len)))
