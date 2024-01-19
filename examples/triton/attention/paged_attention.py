import torch

import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, sm_scale, 
    Block_Tables, Seq_Lens,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbn, stride_kbs, stride_kh, stride_kd,
    stride_vbn, stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_tbs, stride_tn, # block tables
    kv_group_num,
    BLOCK: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(Seq_Lens + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    offs_k = offs_n[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
    offs_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    table_ptr = Block_Tables + cur_batch * stride_tbs

    # load q
    q_ptrs = Q + offs_q
    q = tl.load(q_ptrs)

    # compute attention
    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK):
        start_n = tl.multiple_of(start_n, BLOCK)
        # -- compute qk --
        block_id = tl.load(table_ptr)
        k_ptrs = K + block_id * stride_kbn + offs_k
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale
        qk = tl.where(start_n + offs_n < cur_batch_seq_len, qk, float("-inf"))
        # -- online softmax --
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        # -- update output --
        v_ptrs = V + block_id * stride_vbn + offs_v
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max
        # -- update ptr --
        table_ptr += stride_tn
    acc = acc / e_sum
    
    # store output
    offs_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    o_ptrs = Out + offs_o
    tl.store(o_ptrs, acc)


@torch.no_grad()
def paged_attention(q, k_cache, v_cache, block_tables, seq_lens): 
    """
    q: [bsz, num_head, head_dim]
    k_cache, v_cache: [block_num, block_size, kv_num_head, head_dim]
    block_tables: [bsz, max_seq_block_num]
    seq_lens: [bsz]
    """
    BLOCK = k_cache.shape[1]
    assert BLOCK % 32 == 0
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k_cache.shape[-1], v_cache.shape[-1]
    assert Lq == Lk and Lk == Lv
    batch = q.shape[0]
    assert batch == block_tables.shape[0] and batch == seq_lens.shape[0]

    # allocate output
    o = torch.empty_like(q)

    sm_scale = 1.0 / (Lq**0.5)
    head = q.shape[1]
    kv_group_num = q.shape[1] // k_cache.shape[2]

    grid = (batch, head)
    num_warps = 4 if Lk <= 64 else 8
    
    _attention_kernel[grid](
        q, k_cache, v_cache, sm_scale, 
        block_tables, seq_lens,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), 
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3), 
        o.stride(0), o.stride(1), o.stride(2),
        block_tables.stride(0), block_tables.stride(1), 
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

def torch_paged_attention(q, k_cache, v_cache, block_tables, seq_lens):
    out = torch.empty_like(q)
    nhead, head_dim = k_cache.shape[2], k_cache.shape[3]
    Z = q.shape[0]
    for i in range(Z):
        block_ids = block_tables[i]
        seq_len = seq_lens[i]
        qi = q[i:i+1]  #(1, nhead, head_dim)
        ki = k_cache[block_ids].view(-1, nhead, head_dim)[:seq_len]  #(seqlen, nhead, head_dim)
        vi = v_cache[block_ids].view(-1, nhead, head_dim)[:seq_len]  #(seqlen, nhead, head_dim)
        oi = _naive_attention(qi, ki, vi)
        out[i:i+1] = oi
    return out

if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    # inputs
    batch, head, head_dim = 4, 32, 64
    expand = 1
    max_input_len = 1024 * expand
    block_size = 32
    block_num = (batch * max_input_len + block_size - 1) // block_size
    dtype = torch.float16
    q = torch.randn((batch, head, head_dim), device='cuda', dtype=dtype)
    k_cache = torch.randn((block_num, block_size, head, head_dim), device='cuda', dtype=dtype)
    v_cache = torch.randn((block_num, block_size, head, head_dim), device='cuda', dtype=dtype)
    # meta data
    seq_lens = torch.tensor([512, 1024, 512, 1024], device='cuda', dtype=torch.int32) * expand
    arr = np.arange(block_num, dtype=np.int32)
    arr = np.random.permutation(arr)
    block_tables = torch.from_numpy(arr).cuda()
    block_tables = block_tables.reshape(batch, -1)
    # compute attention
    torch_output = torch_paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
    triton_output = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')
    # benchmark
    print('torch:', triton.testing.do_bench(lambda: torch_paged_attention(q, k_cache, v_cache, block_tables, seq_lens)))
    print('triton:', triton.testing.do_bench(lambda: paged_attention(q, k_cache, v_cache, block_tables, seq_lens)))
