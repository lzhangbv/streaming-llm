import torch

import triton
import triton.language as tl


@triton.jit
def _attention_kernel_stage1(
    Q, K, sm_scale, 
    B_K_Start_Loc, B_O_Start_Loc, B_Seqlen, 
    Att_Out,  #(nhead, total_seq_len)
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs, 
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_k_start_index = tl.load(B_K_Start_Loc + cur_batch)
    cur_batch_o_start_index = tl.load(B_O_Start_Loc + cur_batch)

    # initialize offsets
    block_start_index = start_n * BLOCK_N
    offs_n = block_start_index + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    off_k = (cur_batch_k_start_index + offs_n[:, None]) * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
    
    # compute qK
    block_mask = tl.where(block_start_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q)
        k = tl.load(K + off_k, mask=offs_n[:, None] < cur_batch_seq_len, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_o_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n < cur_batch_seq_len)

@torch.no_grad()
def attention_stage1(q, k_cache, att_out, b_k_start_loc, b_o_start_loc, b_seq_len, max_seq_len): 
    BLOCK = 32
    batch, head_num, Lq = q.shape
    sm_scale = 1.0 / (Lq ** 0.5)
    kv_group_num = q.shape[1] // k_cache.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    grid = (batch, head_num, triton.cdiv(max_seq_len, BLOCK))

    _attention_kernel_stage1[grid](
        q, k_cache, sm_scale, 
        b_k_start_loc, b_o_start_loc, b_seq_len, 
        att_out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        att_out.stride(0), att_out.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=Lq,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

@triton.jit
def _attention_kernel_stage2(
    Logics, V, Out,
    B_V_Start_Loc, B_Logit_Start_Loc, B_Seqlen, 
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_v_start_index = tl.load(B_V_Start_Loc + cur_batch)
    cur_batch_logit_start_index = tl.load(B_Logit_Start_Loc + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_qk = cur_head * stride_logic_h + (cur_batch_logit_start_index + offs_n) * stride_logic_bs
    qk_ptrs = Logics + off_qk
    off_v = (cur_batch_v_start_index + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load qk
        qk = tl.load(qk_ptrs, mask=(start_n + offs_n) < cur_batch_seq_len, other=float("-inf"))
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        # load v
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max
        # update ptrs
        qk_ptrs += BLOCK_N * stride_logic_bs
        v_ptrs +=  BLOCK_N * stride_vbs

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

@torch.no_grad()
def attention_stage2(logics, v, o, b_v_start_loc, b_logit_start_loc, b_seq_len):
    BLOCK = 64
    batch, head, dim = b_seq_len.shape[0], logics.shape[0], v.shape[-1]
    kv_group_num = logics.shape[0] // v.shape[1]
    grid = (batch, head)
    num_warps = 1
    _attention_kernel_stage2[grid](
        logics, v, o, 
        b_v_start_loc, b_logit_start_loc, b_seq_len,
        logics.stride(0), logics.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        kv_group_num,
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3
    )

@torch.no_grad()
def attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len, max_seq_len): 
    """
    q: [bsz, num_head, head_dim]
    k_cache, v_cache: [ntoken, kv_num_head, head_dim]
    b_start_loc, b_seq_len: start locations and sequence lengths for kv cache in a batch
    """
    Lq, Lk, Lv = q.shape[-1], k_cache.shape[-1], v_cache.shape[-1]
    assert Lq == Lk and Lk == Lv
    batch = q.shape[0]
    assert batch == b_start_loc.shape[0] and batch == b_seq_len.shape[0]
    head = q.shape[1]

    # stage 1: qk
    cu_seq_len = torch.nn.functional.pad(torch.cumsum(b_seq_len, dim=0, dtype=torch.torch.int32), (1, 0))
    b_k_start_loc = b_start_loc
    b_o_start_loc = cu_seq_len[0:-1].to(q.device)
    total_seq_len = cu_seq_len[-1].item()

    att_out = torch.empty((head, total_seq_len), dtype=q.dtype, device=q.device)
    attention_stage1(q, k_cache, att_out, b_k_start_loc, b_o_start_loc, b_seq_len, max_seq_len)

    # stage 2: softmax and reducev
    o = torch.empty_like(q)
    attention_stage2(att_out, v_cache, o, b_k_start_loc, b_o_start_loc, b_seq_len)
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
    triton_output = attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len, max_input_len)
    torch_output = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

