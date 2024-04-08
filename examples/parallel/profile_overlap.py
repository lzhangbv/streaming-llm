import os

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity


def init_dist():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size


def naive_comp_and_comm(x, weights):
    for i in range(len(weights)):
        x = torch.matmul(x, weights[i])
        dist.all_reduce(weights[i])
    torch.cuda.synchronize()


def overlap_comp_and_comm(x, weights, comm_stream):
    for i in range(len(weights)):
        x = torch.matmul(x, weights[i])        
        comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(weights[i])
    torch.cuda.synchronize()


def overlap_comm_and_comp(x, weights, comm_stream):
    for i in range(len(weights)):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(weights[i])
        torch.cuda.current_stream().wait_stream(comm_stream)
        x = torch.matmul(x, weights[i])
    torch.cuda.synchronize()


if __name__ == "__main__":
    rank, world_size = init_dist()
    assert world_size > 1

    N = 8
    batch = 4096 # it controls the computation-to-communication ratio
    dim = 8192

    x = torch.randn((batch, dim), dtype=torch.float, device='cuda')
    weights = [torch.randn((dim, dim), dtype=torch.float, device='cuda') for _ in range(N)]
    comm_stream = torch.cuda.Stream()

    # warmup
    naive_comp_and_comm(x, weights)
    overlap_comp_and_comm(x, weights, comm_stream)
    overlap_comm_and_comp(x, weights, comm_stream)
    
    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        naive_comp_and_comm(x, weights)
        overlap_comp_and_comm(x, weights, comm_stream)
        overlap_comm_and_comp(x, weights, comm_stream)

    prof.export_chrome_trace("trace.json")

