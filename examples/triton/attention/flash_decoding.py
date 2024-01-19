import torch

import triton
import triton.language as tl


@triton.jit
def _attention_kernel_stage1(
    Q, K, V, sm_scale, 
    B_Start_Loc, B_Seqlen, 
    Mid_O, Mid_O_LogExpSum, 
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, 
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    # initialize offsets
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    off_k = (cur_batch_start_loc + offs_n[:, None]) * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
    off_v = (cur_batch_start_loc + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # load q
    q = tl.load(q_ptrs) #[dim]

    # number of needed kv blocks
    num_block = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, (cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N)
    
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, num_block, 1): 
        # compute qk
        offs_n_new = start_n * BLOCK_N + offs_n
        k = tl.load(k_ptrs, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0) #[block_n, dim]
        att_value = tl.sum(q[None, :] * k, 1) #[block_n]
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        # softmax and reducev
        v = tl.load(v_ptrs, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0) #[block_n, dim]
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
        # update ptrs
        k_ptrs += BLOCK_N * stride_kbs
        v_ptrs += BLOCK_N * stride_vbs
    
    need_store = tl.where(num_block == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d * stride_mid_od
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))

@torch.no_grad()
def flash_decode_stage1(q, k_cache, v_cache, b_start_loc, b_seq_len, max_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE): 
    BLOCK = 32
    assert PARTITION_SIZE % BLOCK == 0

    batch, head_num, Lq = q.shape
    sm_scale = 1.0 / (Lq ** 0.5)
    kv_group_num = q.shape[1] // k_cache.shape[1]

    grid = (batch, head_num, triton.cdiv(max_seq_len, PARTITION_SIZE))

    _attention_kernel_stage1[grid](
        q, k_cache, v_cache, sm_scale, 
        b_start_loc, b_seq_len, 
        mid_o, mid_o_logexpsum,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_logexpsum.stride(0), mid_o_logexpsum.stride(1), mid_o_logexpsum.stride(2), 
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=Lq,
        BLOCK_N=BLOCK,
        BLOCK_SEQ=PARTITION_SIZE,
        num_warps=1,
        num_stages=2,
    )

@triton.jit
def _attention_kernel_stage2(
    Mid_O,           #[batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    O,               #[batch, head, head_dim]
    B_Seqlen, 
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    # initialize offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d * stride_mid_od
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    v_ptrs = Mid_O + offs_v
    logit_ptrs = Mid_O_LogExpSum + offs_logic

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for _ in range(0, num_partitions, 1):
        # load
        tv = tl.load(v_ptrs)
        tlogic = tl.load(logit_ptrs)
        # reduction
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
        # update ptrs
        v_ptrs += stride_mid_os
        logit_ptrs += stride_mid_o_es
    
    offs_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    tl.store(O + offs_o, acc / sum_exp)

@torch.no_grad()
def flash_decode_stage2(mid_o, mid_o_logexpsum, b_seq_len, o, PARTITION_SIZE):
    batch, head, dim = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]

    grid = (batch, head)
    _attention_kernel_stage2[grid](
        mid_o, mid_o_logexpsum, o, 
        b_seq_len,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_logexpsum.stride(0), mid_o_logexpsum.stride(1), mid_o_logexpsum.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SEQ=PARTITION_SIZE,
        BLOCK_DMODEL=dim,
        num_warps=4,
        num_stages=2
    )

@torch.no_grad()
def flash_decoding(q, k_cache, v_cache, b_start_loc, b_seq_len, max_seq_len): 
    """
    q: [bsz, num_head, head_dim]
    k_cache, v_cache: [ntoken, kv_num_head, head_dim]
    b_start_loc, b_seq_len: start locations and sequence lengths for kv cache in a batch
    """
    Lq, Lk, Lv = q.shape[-1], k_cache.shape[-1], v_cache.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    batch = q.shape[0]
    assert batch == b_start_loc.shape[0] and batch == b_seq_len.shape[0]
    head = q.shape[1]

    # middle results
    PARTITION_SIZE = 256
    max_num_partitions = (max_input_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    mid_o = torch.empty((batch, head, max_num_partitions, head_dim), dtype=torch.float32, device=q.device)
    mid_o_logexpsum = torch.empty((batch, head, max_num_partitions), dtype=torch.float32, device=q.device)

    # stage 1: attention in partitions
    flash_decode_stage1(q, k_cache, v_cache, b_start_loc, b_seq_len, max_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE)

    # stage 2: reduction among partitions
    o = torch.empty_like(q)
    flash_decode_stage2(mid_o, mid_o_logexpsum, b_seq_len, o, PARTITION_SIZE)
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
    expand = 1
    max_input_len = 1024 * expand
    dtype = torch.float16
    q = torch.randn((batch, head, head_dim), device='cuda', dtype=dtype)
    k_cache = torch.randn((batch * max_input_len, head, head_dim), device='cuda', dtype=dtype)
    v_cache = torch.randn((batch * max_input_len, head, head_dim), device='cuda', dtype=dtype)
    # meta data for kv cache
    b_start_loc = torch.tensor([0, 1024, 2048, 3072], dtype=torch.int32, device="cuda") * expand
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda") * expand
    # compute attention
    triton_output = flash_decoding(q, k_cache, v_cache, b_start_loc, b_seq_len, max_input_len)
    torch_output = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')
    # benchmark 
    print('torch:', triton.testing.do_bench(lambda: torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)))
    print('triton:', triton.testing.do_bench(lambda: flash_decoding(q, k_cache, v_cache, b_start_loc, b_seq_len, max_input_len)))
