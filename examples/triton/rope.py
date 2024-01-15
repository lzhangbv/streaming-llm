import torch

import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    Q, Cos, Sin,
    stride_qbs, stride_qh, stride_qd,
    stride_cosbs, stride_cosd,
    stride_sinbs, stride_sind,
    max_total_len,
    H,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)

    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)

    off_q0 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[None, :, None] * stride_qh + dim_range0[None, None, :] * stride_qd
    off_q1 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[None, :, None] * stride_qh + dim_range1[None, None, :] * stride_qd

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    q0 = tl.load(Q + off_q0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H), other=0.0)
    q1 = tl.load(Q + off_q1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H), other=0.0)

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H))
    tl.store(Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H))


def rotary_emb(q, cos, sin):
    total_len = q.shape[0]
    head_num = q.shape[1]
    head_dim = q.shape[2]
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    BLOCK_HEAD = 4
    BLOCK_SEQ = 32
    grid = (triton.cdiv(head_num, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    _rotary_kernel[grid](
        q, cos, sin,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        total_len, head_num,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return q

def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

if __name__ == "__main__":
    torch.manual_seed(0)
    x_shape = (128, 32, 64)
    dtype = torch.float16
    x = torch.randn(x_shape, dtype=dtype, device='cuda')

    cos_shape = (128, 64 // 2)
    y = torch.randn(cos_shape, dtype=dtype, device='cuda')
    cos = y.cos()
    sin = y.sin()

    output_torch = torch_rotary_emb(x, cos, sin)
    output_triton = rotary_emb(x, cos, sin)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')


