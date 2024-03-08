"""
Ring Attention for Sequence Parallel Causal LLM Training. 
    - Flash Attention is adopted from Tri Dao's Implementation
    - Ring Attention is adopted from Zhu Zilin's ZigZag Implementation

    Example: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ring_attention.py
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    from flash_attn import flash_attn_func
    is_flash = True
except:
    # Use Triton implementation if FlashAttention is not installed
    # Warning: the triton version could cause some numerical differences
    from flash_attention_triton import _flash_attn_forward, _flash_attn_backward, flash_attn_func
    is_flash = False


##### utilization #####
def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1) % self.world_size

        if self._process_group is not None:
            send_rank = dist.get_global_rank(self._process_group, send_rank)
            recv_rank = dist.get_global_rank(self._process_group, recv_rank)

        send_op = dist.P2POp(dist.isend, to_send, send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


##### ring attention #####

def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    causal=True,
):
    assert causal == True
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2  # zigzag trick for causal attention
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    def forward(q, k, v, causal):
        if is_flash:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q, 
                k, 
                v, 
                dropout_p=0,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
            )
        else:
            block_out, block_lse, _ = _flash_attn_forward(
                q,
                k,
                v,
                causal=causal,
                softmax_scale=softmax_scale,
                bias=None,
            )
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        # three cases: 
        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    causal=True,
):
    assert causal == True
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        if is_flash:
            _flash_attn_backward(
                dout,
                q, 
                k, 
                v, 
                out,
                softmax_lse,
                dq_buffer[:, :seqlen_q],
                dk_buffer[:, :seqlen_kv],
                dv_buffer[:, :seqlen_kv],
                dropout_p=0,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                rng_state=None,
            )
        else:
            _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq_buffer[:, :seqlen_q],
                dk_buffer[:, :seqlen_kv],
                dv_buffer[:, :seqlen_kv],
                causal=causal,
                softmax_scale=softmax_scale,
                bias=None,
            )

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def ring_flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=True,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        group,
    )


def _log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    import os
    dist.init_process_group("nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.float16

    batch_size = 1
    seqlen = 4096
    nheads = 32
    d = 128
    assert seqlen % (2 * world_size) == 0

    # inputs
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, dtype=dtype).cuda()
    dist.broadcast(qkv, src=0)
    qkv.requires_grad = True

    # output gradients
    dout = torch.randn(batch_size, seqlen, nheads, d, dtype=dtype).cuda()
    dist.broadcast(dout, src=0)

    # flash attention
    out_ref = flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], causal=True)
    out_ref.backward(dout)
    dqkv_ref = qkv.grad
    local_out_ref = extract_local(out_ref, rank, world_size, dim=1)
    local_dqkv_ref = extract_local(dqkv_ref, rank, world_size, dim=1)

    # zigzag-style sequence parallel inputs
    local_qkv = extract_local(qkv, rank, world_size, dim=1).detach().clone()
    local_qkv.requires_grad = True
    local_dout = extract_local(dout, rank, world_size, dim=1).detach().clone()

    # ring attention
    local_out = ring_flash_attn_func(local_qkv[:, :, 0], local_qkv[:, :, 1], local_qkv[:, :, 2], causal=True)
    local_out.backward(local_dout)
    local_dqkv = local_qkv.grad

    # compare results
    _log("out diff", local_out - local_out_ref)
    _log("dq diff", local_dqkv[:, :, 0, :] - local_dqkv_ref[:, :, 0, :])
    _log("dk diff", local_dqkv[:, :, 1, :] - local_dqkv_ref[:, :, 1, :])
    _log("dv diff", local_dqkv[:, :, 2, :] - local_dqkv_ref[:, :, 2, :])
