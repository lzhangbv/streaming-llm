import os
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

"""
LOGLEVEL=INFO USE_LIBUV=1 python profile_launch.py
"""

def worker_fn(launch_time):
    stime = time.time()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"rank: {rank}, launch time: {stime - launch_time:.2f}")

    dist.init_process_group("nccl")
    etime = time.time()
    print(f"rank: {rank}, dist init time: {etime - stime:.2f}")


if __name__ == "__main__":
    launch_time = time.time()

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=4, 
        rdzv_endpoint="localhost:1234", 
        rdzv_backend="c10d",
        run_id="1234"
    )

    elastic_launch(config, worker_fn)(launch_time)

