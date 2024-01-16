import torch

import triton
import triton.language as tl

@triton.jit
def _flash_attention_kernel(
    Q, K, V, sm_scale, 
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    kv_group_num,
    BLOCK_M: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = (cur_batch_in_all_start_index + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = (cur_batch_in_all_start_index + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kbs,
                    mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        # mask
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vbs,
                    mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)

@torch.no_grad()
def flash_attention(q, k, v, b_start_loc, b_seq_len, max_input_len):
    """
    q, k, v: [num_token, num_head, head_dim], inputs with nopad
    b_start_loc, b_seq_len: start locations and sequence lengths in a batch 
    """
    BLOCK = 128
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    # allocate output
    o = torch.empty_like(q)

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1] # group query attention
    
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # (batch, head, block_q)

    num_warps = 4 if Lk <= 64 else 8
    _flash_attention_kernel[grid](
        q, k, v, sm_scale, 
        b_start_loc, b_seq_len,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o

def _naive_attention(q, k ,v):
    import math
    bs, seqlen, num_head, head_dim = q.shape
    device = q.device
    mask = 1.0 - torch.tril(torch.ones((seqlen, seqlen), device=device), diagonal=0).unsqueeze(0).unsqueeze(0)
    mask.masked_fill_(mask.to(torch.bool), -100000000.0)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float() + mask, dim=-1).to(q.dtype)
    output = torch.matmul(scores, v).transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output

def _sdpa(q, k, v):
    bs, seqlen, num_head, head_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output

def torch_attention(q, k, v, b_start_loc, b_seq_len, sdpa=True):
    out = torch.empty_like(q)
    Z = b_start_loc.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        qi = q[start:end].unsqueeze(0)
        ki = k[start:end].unsqueeze(0)
        vi = v[start:end].unsqueeze(0)
        if sdpa:
            oi = _sdpa(qi, ki, vi)
        else:
            oi = _naive_attention(qi, ki, vi)
        out[start:end] = oi.squeeze(0)
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    # inputs
    shape = (3 * 1024, 32, 128)
    dtype = torch.float16
    q = torch.randn(shape, device='cuda', dtype=dtype)
    k = torch.randn(shape, device='cuda', dtype=dtype)
    v = torch.randn(shape, device='cuda', dtype=dtype)
    # meta data
    max_input_len = 1024
    b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    # compute attention
    triton_output = flash_attention(q, k, v, b_start_loc, b_seq_len, max_input_len)
    torch_output = torch_attention(q, k, v, b_start_loc, b_seq_len, sdpa=False)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

