import torch

import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    X,  # pointer to the input
    W,  # pointer to the weight
    Y,  # pointer to the output
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

def rmsnorm(x, rms_w, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    
    # block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    rmsnorm_kernel[(M,)](x, rms_w, out,
                         x.stride(0),
                         N, 
                         eps,
                         BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out

def torch_rmsnorm(x, rms_w, eps=1e-6):
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return rms_w * x.to(torch.float16)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1151, 8192, device='cuda', dtype=torch.float16)
    w = torch.rand(8192, device='cuda', dtype=torch.float16)

    y_triton = rmsnorm(x, w)
    y_torch = torch_rmsnorm(x, w)
    print(y_torch)
    print(y_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(y_torch - y_triton))}')
