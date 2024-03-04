"""
DeepSpeed Training Examples:
    deepspeed --include localhost:0,1,2,3 ds_trainer.py --deepspeed --model_id xxx/llama7b --stage 3 
"""

import argparse
import deepspeed

import torch.nn.functional as F
import torch.utils.data.distributed
import torch.distributed as dist
import timeit
import numpy as np
import time

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

parser.add_argument('--stage', default=3, type=int, choices=[0, 1, 2, 3])
parser.add_argument('--dtype', default="fp32", type=str, choices=["bf16", "fp16", "fp32"])
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument("--memory_snapshot", action="store_true")
parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

deepspeed.init_distributed()

if args.cuda:
    torch.cuda.set_device(args.local_rank)

# Set up standard model.
config = AutoConfig.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_config(config)

if args.cuda:
    model.cuda()

def get_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.0001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
                "torch_adam": True,   #todo: use fused adam
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config

# Memory Snapshot
if args.memory_snapshot and dist.get_rank() == 0:
    torch.cuda.memory._record_memory_history(max_entries=100000)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=get_ds_config(args),
)

# Set up fixed fake data
data = torch.randint(low=0, high=1000, size=(args.batch_size, args.seq_len + 1))
target = data[:, 1:]
data = data[:, 0:args.seq_len]
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    outputs = model_engine(data)
    logits = outputs.logits.view(-1, outputs.logits.shape[-1])
    loss = F.cross_entropy(logits, target.view(-1))
    model_engine.backward(loss)
    model_engine.step()
    torch.cuda.synchronize()

def log(s, nl=True):
    if dist.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')

log('Model: %s' % args.model_id.split('/')[-1])
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, dist.get_world_size()))

if args.memory_snapshot and dist.get_rank() == 0:
    timeit.timeit(benchmark_step, number=3)
    torch.cuda.memory._dump_snapshot("snapshot.pickle")
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


