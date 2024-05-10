import torch
from flash_attention_triton import _flash_attn_forward, _flash_attn_backward, flash_attn_func


"""
No flash-attention-backward-kvcache operation
1) We can implement an efficient fused operation with Triton (todo)
2) We can run multiple flash-attention, and merge them like ring-attention: 
   - merge[attn(qn, k1, v1, causal=False), ..., attn(qn, kn, vn, causal=True)]
   - warning: current Triton version flash-attention is not very numerical stable
"""


@torch.jit.script
def _update_out_and_lse(out, lse, block_out, block_lse):
    "code from ring attention"
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    return out, new_lse


def _flash_attn_forward_kvcache(q, k_cache, v_cache, sm_scale): 
    for i in range(len(k_cache)):
        k = k_cache[i]
        v = v_cache[i]

        block_out, block_lse, _ = _flash_attn_forward(
            q,
            k,
            v,
            causal=(i == len(k_cache) - 1),
            softmax_scale=sm_scale,
            bias=None,
        )

        if len(k_cache) == 1:
            return block_out, block_lse
        
        else:
            if i == 0:
                out = block_out.to(torch.float32)
                lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
            else:
                out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def _flash_attn_backward_kvcache(do, q, k_cache, v_cache, o, lse, dq, dk_cache, dv_cache, sm_scale):
    for i in range(len(k_cache)):
        k = k_cache[i]
        v = v_cache[i]
        dk = dk_cache[i]
        dv = dv_cache[i]
        _flash_attn_backward(
            do, 
            q, 
            k, 
            v, 
            o, 
            lse,
            dq, 
            dk, 
            dv,
            causal=(i == len(k_cache) - 1),
            softmax_scale=sm_scale,
            bias=None,
        )


class PipelineFlashAttnFunc(torch.autograd.Function):
    """
    Split sequence into chunks for pipeline parallelism. 
    - Forward(chunk-1), Forward(chunk-2), ..., Forward(chunk-n)
    - Backward(chunk-n), ..., Backward(chunk-2), Backward(chunk-1) 
    """
    @staticmethod
    def forward(ctx, q, k, v, k_cache, v_cache, k_cache_grad, v_cache_grad, sm_scale): 
        # kv cache
        k_cache.append(k)
        v_cache.append(v)

        # flash attn forward with kvcache
        o, lse = _flash_attn_forward_kvcache(q, k_cache, v_cache, sm_scale)

        # save for backward
        ctx.save_for_backward(q, o, lse)
        ctx.sm_scale = sm_scale or q.shape[-1] ** (-0.5)
        ctx.k_cache = k_cache
        ctx.v_cache = v_cache
        ctx.k_cache_grad = k_cache_grad
        ctx.v_cache_grad = v_cache_grad

        return o

    @staticmethod
    def backward(ctx, do): 
        q, o, lse = ctx.saved_tensors
        k_cache = ctx.k_cache
        v_cache = ctx.v_cache
        k_cache_grad = ctx.k_cache_grad
        v_cache_grad = ctx.v_cache_grad

        # flash attn backward with kvcache
        dq = torch.empty_like(q)
        dk_cache = [torch.empty_like(k) for k in k_cache]
        dv_cache = [torch.empty_like(v) for v in v_cache]
        
        _flash_attn_backward_kvcache(do, q, k_cache, v_cache, o, lse, dq, dk_cache, dv_cache, ctx.sm_scale)

        # kv cache
        k_cache.pop()
        v_cache.pop()

        # kv cache grad
        if len(dk_cache) > 1:
            if len(k_cache_grad) == 0:
                # store past kv cache grads
                k_cache_grad.extend(dk_cache[:-1])
                v_cache_grad.extend(dv_cache[:-1])
                # return current kv grads
                dk, dv = dk_cache[-1], dv_cache[-1]
            else:
                assert len(k_cache_grad) == len(dk_cache)
                assert len(v_cache_grad) == len(dv_cache)
                # grad accumulation
                for i in range(len(k_cache_grad)):
                    k_cache_grad[i].add_(dk_cache[i])
                    v_cache_grad[i].add_(dv_cache[i])
                # return current kv grads
                dk = k_cache_grad.pop()
                dv = v_cache_grad.pop()
        
        else:
            # last backward iteration
            if len(k_cache_grad) == 1:
                k_cache_grad[0].add_(dk_cache[0])
                v_cache_grad[0].add_(dv_cache[0])

                # return current kv grads
                dk = k_cache_grad.pop()
                dv = v_cache_grad.pop()

            else:
                dk = dk_cache[0]
                dv = dv_cache[0]

        return dq, dk, dv, None, None, None, None, None


def pipeline_flash_attention(q, k, v, k_cache, v_cache, k_cache_grad, v_cache_grad, sm_scale=None):
    return PipelineFlashAttnFunc.apply(q, k, v, k_cache, v_cache, k_cache_grad, v_cache_grad, sm_scale)


if __name__ == "__main__":
    # shapes
    batch_size = 1
    seqlen = 4096
    nheads = 32
    headdim = 128
    dtype = torch.float16

    num_chunks = 4
    assert seqlen % num_chunks == 0

    # inputs
    q_ref = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype).cuda().div_(100)
    k_ref = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype).cuda().div_(100)
    v_ref = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype).cuda().div_(100)
    q_ref.requires_grad = True
    k_ref.requires_grad = True
    v_ref.requires_grad = True

    # output gradients
    dout = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype).cuda().div_(100)

    # flash attention
    out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
    out_ref.backward(dout)

    # clone inputs
    q = q_ref.detach().clone()
    k = k_ref.detach().clone()
    v = v_ref.detach().clone()
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    # pipeline flash attention
    k_cache = []
    v_cache = []
    k_cache_grad = []
    v_cache_grad = []

    q_chunks = torch.chunk(q, chunks=num_chunks, dim=1)
    k_chunks = torch.chunk(k, chunks=num_chunks, dim=1)
    v_chunks = torch.chunk(v, chunks=num_chunks, dim=1)
    dout_chunks = torch.chunk(dout, chunks=num_chunks, dim=1)

    outs = []
    for i in range(num_chunks): 
        out = pipeline_flash_attention(
            q_chunks[i], 
            k_chunks[i], 
            v_chunks[i], 
            k_cache, 
            v_cache, 
            k_cache_grad, 
            v_cache_grad,
        )
        outs.append(out)
    
    # backward is in reversed order
    for i in reversed(range(num_chunks)):
        outs[i].backward(dout_chunks[i])

    print(torch.max(torch.abs(q.grad - q_ref.grad)))
    print(torch.max(torch.abs(k.grad - k_ref.grad)))
    print(torch.max(torch.abs(v.grad - v_ref.grad)))

    # profile if needed
    no_profile = True
    if no_profile:
        exit()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # no split
        out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
        out_ref.backward(dout)

        # split
        outs = []
        for i in range(num_chunks):
            out = pipeline_flash_attention(
                q_chunks[i],
                k_chunks[i],
                v_chunks[i],
                k_cache,
                v_cache,
                k_cache_grad,
                v_cache_grad,
            )
            outs.append(out)

        for i in reversed(range(num_chunks)):
            outs[i].backward(dout_chunks[i])

    prof.export_chrome_trace("trace.json")

