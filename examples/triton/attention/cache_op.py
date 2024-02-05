import torch

import triton
import triton.language as tl

@triton.jit
def _kv_cache_kernel(
    K, V, 
    K_Cache, V_Cache,
    slots,
    stride_kn, stride_kd, 
    stride_kcn, stride_kcd,
    BLOCK: tl.constexpr, 
):
    cur_token = tl.program_id(0)
    cur_block = tl.program_id(1)
    cur_slot = tl.load(slots + cur_token)
    # initialize offsets
    offs_n = cur_block * BLOCK + tl.arange(0, BLOCK)
    offs_kv = cur_token * stride_kn + offs_n * stride_kd
    offs_kv_cache = cur_slot * stride_kcn + offs_n * stride_kcd
    # k cache
    k = tl.load(K + offs_kv)
    tl.store(K_Cache + offs_kv_cache, k)
    # v cache
    v = tl.load(V + offs_kv)
    tl.store(V_Cache + offs_kv_cache, v)

@torch.no_grad()
def reshape_and_cache(k, v, k_cache, v_cache, slots):
    """
    k, v: [num_token, num_head, head_size]
    k_cache, v_cache: [block_num, block_size, num_head, head_size]
    slots: slot ids of tokens
    """
    N, head_num, head_size = k.shape
    k = k.view(-1, head_num * head_size)
    v = v.view(-1, head_num * head_size)
    k_cache = k_cache.view(-1, head_num * head_size)
    v_cache = v_cache.view(-1, head_num * head_size)
    assert N == slots.shape[0]

    BLOCK = min(head_num * head_size, 512)
    assert (head_num * head_size) % BLOCK == 0

    grid = (N, triton.cdiv(head_num * head_size, BLOCK))

    _kv_cache_kernel[grid](
        k, v, 
        k_cache, v_cache,
        slots,
        k.stride(0), k.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        BLOCK,
        num_warps=4,
    )

def torch_reshape_and_cache(k, v, k_cache, v_cache, slots):
    N, head_num, head_size = k.shape
    k_cache = k_cache.view(-1, head_num, head_size)
    v_cache = v_cache.view(-1, head_num, head_size)

    for i in range(N):
        slot = slots[i]
        k_cache[slot] = k[i]
        v_cache[slot] = v[i]

if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    torch.manual_seed(0)
    # kv
    ntoken, nhead, head_dim = 1024, 32, 128
    shape = (ntoken, nhead, head_dim)
    dtype = torch.float16
    k = torch.randn(shape, device='cuda', dtype=dtype)
    v = torch.randn(shape, device='cuda', dtype=dtype)
    # meta data
    block_size = 32
    assert ntoken % block_size == 0
    block_num = ntoken // block_size
    arr = np.arange(ntoken, dtype=np.int32)
    arr = np.random.permutation(arr)
    slots = torch.from_numpy(arr).cuda()
    # kv cache
    shape = (block_num, block_size, nhead, head_dim)
    k_cache_torch = torch.randn(shape, device='cuda', dtype=dtype)
    v_cache_torch = torch.randn(shape, device='cuda', dtype=dtype)
    k_cache_triton = torch.randn(shape, device='cuda', dtype=dtype)
    v_cache_triton = torch.randn(shape, device='cuda', dtype=dtype)
    # reshape and cache
    torch_reshape_and_cache(k, v, k_cache_torch, v_cache_torch, slots)
    reshape_and_cache(k, v, k_cache_triton, v_cache_triton, slots)
    print(f'The maximum difference of kcache is {torch.max(torch.abs(k_cache_torch - k_cache_triton))}')
    print(f'The maximum difference of vcache is {torch.max(torch.abs(v_cache_torch - v_cache_triton))}')
    # benchmark
    print("torch:", triton.testing.do_bench(lambda: torch_reshape_and_cache(k, v, k_cache_torch, v_cache_torch, slots)))
    print("triton:", triton.testing.do_bench(lambda: reshape_and_cache(k, v, k_cache_triton, v_cache_triton, slots)))
