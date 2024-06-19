import os

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

"""
Implement All2All with torch's isend/irecv ops

todo: more efficient all2all implementations based on mscclang
https://github.com/microsoft/msccl-tools/tree/main/examples/mscclang
"""

def init_dist():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size


def profile_linear_all2all(inputs, outputs, rank, world_size):
    assert inputs.shape[0] == world_size
    assert outputs.shape[0] == world_size

    for i in range(world_size):
        if i == 0:
            outputs[rank] = inputs[rank]
        else:
            ops = []
            # send: rank to tgt
            tgt = (rank + i) % world_size
            ops.append(dist.P2POp(dist.isend, inputs[tgt], tgt))
            # recv: src to rank
            src = (rank + world_size - i) % world_size
            ops.append(dist.P2POp(dist.irecv, outputs[src], src))
            
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    #torch.cuda.synchronize()


def profile_batch_all2all(inputs, outputs, rank, world_size):
    assert inputs.shape[0] == world_size
    assert outputs.shape[0] == world_size

    ops = []
    for i in range(world_size):
        if i == 0:
            outputs[rank] = inputs[rank]
        else:
            # send: rank to tgt
            tgt = (rank + i) % world_size
            ops.append(dist.P2POp(dist.isend, inputs[tgt], tgt))
            # recv: src to rank
            src = (rank + world_size - i) % world_size
            ops.append(dist.P2POp(dist.irecv, outputs[src], src))
            
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    #torch.cuda.synchronize()


if __name__ == "__main__":
    rank, world_size = init_dist()

    dim = 2048
    inputs = torch.randn((world_size, dim), dtype=torch.float, device='cuda')
    outputs = torch.randn((world_size, dim), dtype=torch.float, device='cuda')
    outputs_ref = torch.randn((world_size, dim), dtype=torch.float, device='cuda')

    dist.all_to_all_single(outputs_ref, inputs)
    profile_linear_all2all(inputs, outputs, rank, world_size)
    profile_batch_all2all(inputs, outputs, rank, world_size)
    #print(f'diff: {torch.max(torch.abs(outputs_ref - outputs))}')

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        dist.all_to_all_single(outputs_ref, inputs)
        profile_linear_all2all(inputs, outputs, rank, world_size)
        profile_batch_all2all(inputs, outputs, rank, world_size)

    if rank == 0:
        prof.export_chrome_trace("all2all.json")



