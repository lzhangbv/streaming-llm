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

def profile_send_recv(weights, rank, world_size):
    """
    Send and recv use two cuda streams, and they are different from nccl stream, which is used for collective communications
    """
    recv_prev, send_next, recv_next, send_prev = weights[0], weights[1], weights[2], weights[3]
    next_rank = (rank + 1) % world_size
    prev_rank = (rank + world_size - 1) % world_size

    # computation and collective communication
    torch.matmul(recv_prev, recv_prev.T)
    dist.all_reduce(recv_prev)
    torch.cuda.synchronize()

    # recv_prev, send_next
    if rank % 2 == 0:
        dist.send(send_next, dst=next_rank)
        dist.recv(recv_prev, src=prev_rank)
    else:
        dist.recv(recv_prev, src=prev_rank)
        dist.send(send_next, dst=next_rank)
    torch.cuda.synchronize()

    # recv_next, send_prev
    if rank % 2 == 0:
        dist.send(send_prev, dst=prev_rank)
        dist.recv(recv_next, src=next_rank)
    else:
        dist.recv(recv_next, src=next_rank)
        dist.send(send_prev, dst=prev_rank)
    torch.cuda.synchronize()


def profile_isend_irecv(weights, rank, world_size):
    recv_prev, send_next, recv_next, send_prev = weights[0], weights[1], weights[2], weights[3]
    next_rank = (rank + 1) % world_size
    prev_rank = (rank + world_size - 1) % world_size

    handles = []
    if rank % 2 == 0:
        handles.append(dist.isend(send_prev, dst=prev_rank))
        handles.append(dist.irecv(recv_next, src=next_rank))
    else:
        handles.append(dist.irecv(recv_next, src=next_rank))
        handles.append(dist.isend(send_prev, dst=prev_rank))

    if rank % 2 == 0:
        handles.append(dist.isend(send_next, dst=next_rank))
        handles.append(dist.irecv(recv_prev, src=prev_rank))
    else:
        handles.append(dist.irecv(recv_prev, src=prev_rank))
        handles.append(dist.isend(send_next, dst=next_rank))

    for handle in handles:
        handle.wait()

    torch.cuda.synchronize()


def profile_batch_isend_irecv(weights, rank, world_size):
    """
    Batch async p2p ops: operations within a group will be a single kernel, and sent to nccl stream
    """
    recv_prev, send_next, recv_next, send_prev = weights[0], weights[1], weights[2], weights[3]
    next_rank = (rank + 1) % world_size
    prev_rank = (rank + world_size - 1) % world_size

    ops = []
    ops.append(dist.P2POp(dist.isend, send_prev, prev_rank))
    ops.append(dist.P2POp(dist.irecv, recv_next, next_rank))

    ops.append(dist.P2POp(dist.isend, send_next, next_rank))
    ops.append(dist.P2POp(dist.irecv, recv_prev, prev_rank))

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    torch.cuda.synchronize()


if __name__ == "__main__":
    rank, world_size = init_dist()
    assert world_size > 1

    dim = 2048
    weights = [torch.randn((dim, dim), dtype=torch.float, device='cuda') for _ in range(4)]
    torch.cuda.synchronize()

    # warmup
    profile_send_recv(weights, rank, world_size)
    profile_isend_irecv(weights, rank, world_size)
    profile_batch_isend_irecv(weights, rank, world_size)

    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        profile_send_recv(weights, rank, world_size)
        profile_isend_irecv(weights, rank, world_size)
        profile_batch_isend_irecv(weights, rank, world_size)

    if rank == 0:
        prof.export_chrome_trace("p2p.json")
        
