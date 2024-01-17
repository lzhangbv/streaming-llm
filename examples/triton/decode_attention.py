import torch

import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, sm_scale, 
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    kv_group_num,
    BLOCK: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.load(B_Start_Loc + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    off_k = (cur_batch_start_index + offs_n[:, None]) * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
    off_v = (cur_batch_start_index + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # load q
    q = tl.load(q_ptrs)

    # compute attention
    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK):
        start_n = tl.multiple_of(start_n, BLOCK)
        # -- compute qk ----
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale
        qk = tl.where(start_n + offs_n < cur_batch_seq_len, qk, float("-inf"))
        # -- online softmax
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        # -- update output
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max
        # -- update ptrs --
        k_ptrs += BLOCK * stride_kbs
        v_ptrs += BLOCK * stride_vbs
    acc = acc / e_sum
    
    # store output
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@torch.no_grad()
def attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len): 
    """
    q: [bsz, num_head, head_dim]
    k_cache, v_cache: [ntoken, kv_num_head, head_dim]
    b_start_loc, b_seq_len: start locations and sequence lengths in a batch
    """
    BLOCK = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k_cache.shape[-1], v_cache.shape[-1]
    assert Lq == Lk and Lk == Lv
    batch = q.shape[0]
    assert batch == b_start_loc.shape[0] and batch == b_seq_len.shape[0]

    # allocate output
    o = torch.empty_like(q)

    sm_scale = 1.0 / (Lq**0.5)
    head = q.shape[1]
    kv_group_num = q.shape[1] // k_cache.shape[1]

    grid = (batch, head)
    num_warps = 4 if Lk <= 64 else 8
    
    _attention_kernel[grid](
        q, k_cache, v_cache, sm_scale, 
        b_start_loc, b_seq_len,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK=BLOCK,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=1,
    )
    return o    

def _naive_attention(q, k, v):
    import math
    head_dim = q.shape[-1]
    q = q.transpose(0, 1)  #(nhead, 1, head_dim)
    k = k.transpose(0, 1)  #(nhead, seqlen, head_dim)
    v = v.transpose(0, 1)  #(nhead, seqlen, head_dim)
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(scores, v).transpose(0, 1).contiguous() #(1, nhead, head_dim)
    return output

def torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len):
    out = torch.empty_like(q)
    Z = q.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        qi = q[i:i+1]            #(1, nhead, head_dim)
        ki = k_cache[start:end]  #(seqlen, nhead, head_dim)
        vi = v_cache[start:end]  #(seqlen, nhead, head_dim)
        oi = _naive_attention(qi, ki, vi)
        out[i:i+1] = oi
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    # inputs
    batch, head, head_dim = 4, 32, 64
    max_input_len = 1024
    dtype = torch.float16
    q = torch.randn((batch, head, head_dim), device='cuda', dtype=dtype)
    k_cache = torch.randn((batch * max_input_len, head, head_dim), device='cuda', dtype=dtype)
    v_cache = torch.randn((batch * max_input_len, head, head_dim), device='cuda', dtype=dtype)
    # meta data
    b_start_loc = torch.tensor([0, 1024, 2048, 3072], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    # compute attention
    torch_output = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
    triton_output = attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')
    
