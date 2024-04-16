import os
import collections

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


def overlap_comp_and_comm_with_async_op(x, weights):
    handles = []
    for i in range(len(weights)):
        x = torch.matmul(x, weights[i])
        handles.append(dist.all_reduce(weights[i], async_op=True))
    for handle in handles:
        handle.wait()
    torch.cuda.synchronize()


def overlap_comm_and_comp(x, weights, comm_stream):
    for i in range(len(weights)):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(weights[i])
        torch.cuda.current_stream().wait_stream(comm_stream)
        x = torch.matmul(x, weights[i])
    torch.cuda.synchronize()


def overlap_comm_and_comp_with_async_op(x, weights):
    handles = []
    for i in range(len(weights)):
        handles.append(dist.all_reduce(weights[i], async_op=True))
    for i in range(len(weights)):
        handles[i].wait()
        x = torch.matmul(x, weights[i])


def overlap_two_comms(weights1, weights2, stream1, stream2):
    """collective communications are using the same cuda stream (i.e., no overlap)"""
    for i in range(len(weights1)):
        with torch.cuda.stream(stream1):
            dist.all_reduce(weights1[i])
        with torch.cuda.stream(stream2):
            dist.all_reduce(weights2[i][0:1024, :])
    torch.cuda.synchronize()


def overlap_two_comps(x, weights1, weights2, stream1, stream2):
    """large matmul and small matmul operations are on two cuda streams"""
    for i in range(len(weights1)):
        with torch.cuda.stream(stream1):
            torch.matmul(x, weights1[i])
        with torch.cuda.stream(stream2):
            torch.matmul(x[0:128, :], weights2[i])
    torch.cuda.synchronize()


class EventQueue:
    def __init__(self, limit_num=2):
        self.queue = collections.deque()
        self.limit_num = limit_num
    
    def enqueue(self, event):
        self.queue.append(event)
    
    def dequent(self):
        if len(self.queue) >= self.limit_num:
            return self.queue.popleft()
        return None

    def clear(self):
        self.queue.clear()


def limit_overlap_comm_and_comp(x, weights, comm_stream, event_queue):
    """
    Limit the maximum number of communication operations, e.g., no overlap if limit_num=1
    """
    for i in range(len(weights)):
        event = event_queue.dequent()
        if event is not None:
            # option 1: cpu waits event
            #event.synchronize()

            # option 2: comm_stream waits event (suggest)
            comm_stream.wait_event(event)
        
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(weights[i])

        torch.cuda.current_stream().wait_stream(comm_stream)
        x = torch.matmul(x, weights[i])
        
        new_event = torch.cuda.Event()
        new_event.record()
        event_queue.enqueue(new_event)

    event_queue.clear()
    torch.cuda.synchronize()


def profile_collective_ops(world_size):
    dim = 2048
    x = torch.randn((dim, dim), dtype=torch.float, device='cuda')
    y = torch.randn((dim, dim), dtype=torch.float, device='cuda')
    shard_x = torch.randn((dim // world_size, dim), dtype=torch.float, device='cuda')

    # all collective communications use the same nccl stream
    dist.all_reduce(x)
    #dist._reduce_scatter_base(shard_x, x)
    #dist._all_gather_base(x, shard_x)
    dist.reduce_scatter_tensor(shard_x, x)
    dist.all_gather_into_tensor(x, shard_x)
    dist.reduce(x, dst=0)
    dist.broadcast(x, src=0)
    torch.distributed.all_to_all_single(y, x)

    torch.cuda.synchronize()

    # Even with explicit cuda stream, collective operations are sent to nccl stream
    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        dist.all_reduce(x)
    comm_stream.synchronize()

    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        dist.reduce_scatter_tensor(shard_x, x)
    comm_stream.synchronize()

    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        dist.all_gather_into_tensor(x, shard_x)
    comm_stream.synchronize()

    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        dist.reduce(x, dst=0)
    comm_stream.synchronize()

    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        dist.broadcast(x, src=0)
    comm_stream.synchronize()

    comm_stream = torch.cuda.Stream()
    with torch.cuda.stream(comm_stream):
        torch.matmul(x, x)
        torch.distributed.all_to_all_single(y, x)
    comm_stream.synchronize()


if __name__ == "__main__":
    rank, world_size = init_dist()
    assert world_size > 1

    N = 16
    batch = 4096 * 2 # it controls the computation-to-communication ratio
    dim = 8192

    x = torch.randn((batch, dim), dtype=torch.float, device='cuda')
    weights = [torch.randn((dim, dim), dtype=torch.float, device='cuda') for _ in range(N)]
    weights2 = [torch.randn((dim, dim), dtype=torch.float, device='cuda') for _ in range(N)]
    comm_stream = torch.cuda.Stream()
    comm_stream2 = torch.cuda.Stream()
    event_queue = EventQueue(limit_num=3)

    # warmup
    #profile_collective_ops(world_size)
    #naive_comp_and_comm(x, weights)
    overlap_comp_and_comm(x, weights, comm_stream)
    overlap_comp_and_comm_with_async_op(x, weights)
    #overlap_comm_and_comp(x, weights, comm_stream)
    #overlap_comm_and_comp_with_async_op(x, weights)
    #limit_overlap_comm_and_comp(x, weights, comm_stream, event_queue)
    #overlap_two_comms(weights, weights2, comm_stream, comm_stream2)
    #overlap_two_comps(x, weights, weights2, comm_stream, comm_stream2)
    
    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #profile_collective_ops(world_size)
        #naive_comp_and_comm(x, weights)
        overlap_comp_and_comm(x, weights, comm_stream)
        overlap_comp_and_comm_with_async_op(x, weights)
        #overlap_comm_and_comp(x, weights, comm_stream)
        #overlap_comm_and_comp_with_async_op(x, weights)
        #limit_overlap_comm_and_comp(x, weights, comm_stream, event_queue)
        #overlap_two_comms(weights, weights2, comm_stream, comm_stream2)
        #overlap_two_comps(x, weights, weights2, comm_stream, comm_stream2)

    prof.export_chrome_trace("trace.json")

