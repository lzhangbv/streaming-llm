import torch
from torch.profiler import profile, ProfilerActivity


def non_blocking_io_and_comp(x, weights_cpu, non_blocking=True):
    for i in range(len(weights_cpu)):
        weight = weights_cpu[i].to('cuda', non_blocking=non_blocking)
        torch.matmul(x, weight)
    torch.cuda.synchronize()


def overlap_io_and_comp(x, weights_cpu, non_blocking, transfer_stream):
    for i in range(len(weights_cpu)):
        with torch.cuda.stream(transfer_stream):
            weight = weights_cpu[i].to('cuda', non_blocking=non_blocking)
        torch.cuda.current_stream().wait_stream(transfer_stream)
        torch.matmul(x, weight)
    torch.cuda.synchronize()


def non_blocking_comp_and_io(x, weights_gpu, non_blocking=True):
    for i in range(len(weights_gpu)):
        torch.matmul(x, weights_gpu[i])
        weight = weights_gpu[i].to('cpu', non_blocking=non_blocking) # cpu tensor is pinned if non_blocking=True
    torch.cuda.synchronize()


def overlap_comp_and_io(x, weights_gpu, non_blocking, transfer_stream):
    for i in range(len(weights_gpu)):
        torch.matmul(x, weights_gpu[i])
        transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(transfer_stream):
            weight = weights_gpu[i].to('cpu', non_blocking=non_blocking)
    torch.cuda.synchronize()


if __name__ == "__main__":
    N = 8
    batch = 4096  # it controls the computation-to-io ratio
    dim = 8192

    x = torch.randn((batch, dim), dtype=torch.float, device='cuda')
    
    # pinned cpu memory is required
    #weights_cpu = [torch.randn((dim, dim), dtype=torch.float, device='cpu') for _ in range(N)]
    weights_cpu = [torch.randn((dim, dim), dtype=torch.float, device='cpu').pin_memory() for _ in range(N)]
    
    weights_gpu = [torch.randn((dim, dim), dtype=torch.float, device='cuda') for _ in range(N)]
    transfer_stream = torch.cuda.Stream()

    torch.cuda.synchronize()

    # warmup
    #non_blocking_io_and_comp(x, weights_cpu, non_blocking=False)
    non_blocking_io_and_comp(x, weights_cpu, non_blocking=True)
    overlap_io_and_comp(x, weights_cpu, non_blocking=True, transfer_stream=transfer_stream)
    #non_blocking_comp_and_io(x, weights_gpu, non_blocking=False)
    #non_blocking_comp_and_io(x, weights_gpu, non_blocking=True)
    #overlap_comp_and_io(x, weights_gpu, non_blocking=True, transfer_stream=transfer_stream)

    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #non_blocking_io_and_comp(x, weights_cpu, non_blocking=False)
        non_blocking_io_and_comp(x, weights_cpu, non_blocking=True)
        overlap_io_and_comp(x, weights_cpu, non_blocking=True, transfer_stream=transfer_stream)
        #non_blocking_comp_and_io(x, weights_gpu, non_blocking=False)
        #non_blocking_comp_and_io(x, weights_gpu, non_blocking=True)
        #overlap_comp_and_io(x, weights_gpu, non_blocking=True, transfer_stream=transfer_stream)
    
    prof.export_chrome_trace("trace.json")
