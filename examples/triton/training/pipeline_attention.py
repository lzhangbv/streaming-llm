import torch

"""
WIP: no flash-attention-with-kv-cache for backward 
1) We can implement it with Triton (todo)
2) We can run multiple flash-attention ops, like ring-attention: 
   - merge[attn(qn, k1, v1, causal=False), ..., attn(qn, kn, vn, causal=True)]
"""

seq_dim = 1 # (batch_size, seqlen, nheads, headdim)
# seq_dim = 2 # (batch_size, nheads, seqlen, headdim)

class PipelineFlashAttnFunc(torch.autograd.Function):
    """
    Split sequence into chunks for pipeline parallelism. 
    - Forward(chunk-1), Forward(chunk-2), ..., Forward(chunk-n)
    - Backward(chunk-n), ..., Backward(chunk-2), Backward(chunk-1) 
    """
    @staticmethod
    def forward(ctx, q, k, v, k_cache, v_cache, k_cache_grad, v_cache_grad, sm_scale): 
        # kv cache
        k_cat = torch.cat(k_cache + [k], dim=seq_dim) if len(k_cache) > 0 else k
        v_cat = torch.cat(v_cache + [v], dim=seq_dim) if len(v_cache) > 0 else v
        k_cache.append(k)
        v_cache.append(v)

        # flash attn forward
        o, lse = _flash_attn_forward_kvcache(q, k_cat, v_cat, sm_scale)

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

        # kv cache
        k_cat = torch.cat(k_cache, dim=seq_dim) if len(k_cache) > 1 else k_cache[0]
        v_cat = torch.cat(v_cache, dim=seq_dim) if len(v_cache) > 1 else v_cache[0]
        k_cache.pop()
        v_cache.pop()

        # # flash attn backward
        dq = torch.empty_like(q)
        dk_cat = torch.empty_like(k_cat)
        dv_cat = torch.empty_like(v_cat)
        _flash_attn_backward_kvcache(do, q, k_cat, v_cat, o, lse, dq, dk_cat, dv_cat, ctx.sm_scale)

        # kv cache grad
        num_chunks = dk_cat.shape[seq_dim] // dq.shape[seq_dim]
        if num_chunks > 1:
            dk_chunks = torch.chunk(dk_cat, chunks=num_chunks, dim=seq_dim)
            dv_chunks = torch.chunk(dv_cat, chunks=num_chunks, dim=seq_dim)
            
            if len(k_cache_grad) == 0:
                # store past kv cache grads
                k_cache_grad.extend(dk_chunks[:-1])
                v_cache_grad.extend(dv_chunks[:-1])
                # return current kv grads
                dk, dv = dk_chunks[-1], dv_chunks[-1]
            else:
                assert len(k_cache_grad) == len(dk_chunks)
                assert len(v_cache_grad) == len(dv_chunks)
                # grad accumulation
                for i in range(len(k_cache_grad)):
                    k_cache_grad[i].add_(dk_chunks[i])
                    v_cache_grad[i].add_(dv_chunks[i])
                # return current kv grads
                dk = k_cache_grad.pop()
                dv = v_cache_grad.pop()
        
        else:
            # last backward iteration
            assert len(k_cache_grad) == 1
            assert len(v_cache_grad) == 1
            k_cache_grad[0].add_(dk_cat)
            v_cache_grad[0].add_(dv_cat)

            # return current kv grads
            dk = k_cache_grad.pop()
            dv = v_cache_grad.pop()

        return dq, dk, dv, None, None, None, None, None

