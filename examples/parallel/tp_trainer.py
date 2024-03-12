"""
Torch Native Tensor Parallelism Example: 
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tp_trainer.py --model_id xxx/llama7b --dtype fp16

Based on: https://github.com/pytorch/pytorch/blob/main/test/distributed/tensor/parallel/test_tp_examples.py
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

from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import DeviceMesh, Replicate

from transformers import AutoModelForCausalLM, AutoConfig

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


# tp parameters
parser.add_argument('--dtype', default="fp32", type=str, choices=["bf16", "fp16", "fp32"])
parser.add_argument("--shard_embed", action="store_true")

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

device = "cuda" if args.cuda else "cpu"
model.to(device)

# dtype
dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.dtype]
ctx = torch.amp.autocast(device, dtype=dtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'fp16'))

device_mesh = DeviceMesh(device, torch.arange(0, dist.get_world_size()))

# Parallelize the attention and mlp
if dist.get_world_size() > 1:
    for layer in model.model.layers:
        layer_parallelize_plan = {} 
        layer_parallelize_plan["self_attn.q_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.k_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.v_proj"] = ColwiseParallel()
        layer_parallelize_plan["self_attn.o_proj"] = RowwiseParallel()
        layer_parallelize_plan["mlp.gate_proj"] = ColwiseParallel()
        layer_parallelize_plan["mlp.up_proj"] = ColwiseParallel()
        layer_parallelize_plan["mlp.down_proj"] = RowwiseParallel()
        parallelize_module(layer, device_mesh, layer_parallelize_plan)
    
        # adjust the number of heads locally
        assert model.model.config.num_attention_heads % dist.get_world_size() == 0
        layer.self_attn.num_heads = model.model.config.num_attention_heads // dist.get_world_size()
        layer.self_attn.num_key_value_heads = model.model.config.num_key_value_heads // dist.get_world_size()
        layer.self_attn.hidden_size = model.model.config.hidden_size // dist.get_world_size()

    if args.shard_embed:
        parallelize_module(model.model.embed_tokens, device_mesh, ColwiseParallel(output_layouts=Replicate()))
        parallelize_module(model.lm_head, device_mesh, RowwiseParallel(input_layouts=Replicate()))

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Set up fixed fake data
data = torch.randint(low=0, high=1000, size=(args.batch_size, args.seq_len + 1))
if args.cuda:
    data = data.cuda()

dist.broadcast(data, src=0)

target = data[:, 1:].contiguous()
data = data[:, 0:args.seq_len]

def benchmark_step():
    optimizer.zero_grad()
    with ctx:
        outputs = model(data)
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        loss = F.cross_entropy(logits, target.view(-1))
    loss.backward()
    optimizer.step()
    #tofix: grad scaler is imcompatible to dtensor
    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()
    torch.cuda.synchronize()

def log(s, nl=True):
    if dist.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')

log('Model: %s' % args.model_id.split('/')[-1])
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, dist.get_world_size()))

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
#log('Total img/sec on %d %s(s): %.1f +-%.1f' %
#    (dist.get_world_size(), device, dist.get_world_size() * img_sec_mean, dist.get_world_size() * img_sec_conf))
