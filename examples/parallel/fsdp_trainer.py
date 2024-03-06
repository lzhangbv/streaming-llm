"""
Fully Sharded Data Parallelism Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 fsdp_trainer.py --model_id xxx/llama7b --dtype fp16
"""

import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import timeit
import numpy as np
import os
import time
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_id', type=str, 
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=4,
                    help='input batch size')
parser.add_argument('--seq-len', type=int, default=1024,
                    help='input sequence length')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=5,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument("--memory_snapshot", action="store_true")

parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')


# fsdp parameters
parser.add_argument('--dtype', default="fp32", type=str, choices=["bf16", "fp16", "fp32"])
parser.add_argument("--cpu_offload", action="store_true")
parser.add_argument("--gradient_checkpoint", action="store_true")
parser.add_argument("--fused_adam", action="store_true")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dist.init_process_group(backend='nccl', init_method='env://')
args.local_rank = int(os.environ['LOCAL_RANK'])

if args.cuda:
    # pin GPU to local rank.
    torch.cuda.set_device(args.local_rank)

# Set up standard model.
config = AutoConfig.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_config(config)

if args.gradient_checkpoint:
    model.gradient_checkpointing_enable()

llama_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

if args.dtype == 'fp32':
    mp_policy = None
elif args.dtype == 'fp16':
    mp_policy = MixedPrecision(param_dtype=torch.float16)
else:
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16)

# Memory Snapshot
if args.memory_snapshot and dist.get_rank() == 0:
    torch.cuda.memory._record_memory_history(max_entries=100000)

model = FSDP(
    model, 
    auto_wrap_policy=llama_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=args.cpu_offload),
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)

optimizer = optim.AdamW(model.parameters(), lr=0.0001, fused=args.fused_adam)

# Set up fixed fake data
data = torch.randint(low=0, high=1000, size=(args.batch_size, args.seq_len + 1))
target = data[:, 1:]
data = data[:, 0:args.seq_len]
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    optimizer.zero_grad()
    outputs = model(data)
    logits = outputs.logits.view(-1, outputs.logits.shape[-1])
    loss = F.cross_entropy(logits, target.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

def log(s, nl=True):
    if dist.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')

log('Model: %s' % args.model_id.split('/')[-1])
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, dist.get_world_size()))

# Memory Snapshot
if args.memory_snapshot and dist.get_rank() == 0:
    timeit.timeit(benchmark_step, number=3)
    torch.cuda.memory._dump_snapshot("fsdp_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (dist.get_world_size(), device, dist.get_world_size() * img_sec_mean, dist.get_world_size() * img_sec_conf))



