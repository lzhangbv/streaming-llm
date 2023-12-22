import torch
import time
import argparse
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--input", type=int, default=2048)
    parser.add_argument("--output", type=int, default=32)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=5)
    # kmeans params
    parser.add_argument("--enable_kmeans", action="store_true")
    parser.add_argument("--enable_kmeans_v2", action="store_true")
    parser.add_argument("--start_size", type=int, default=256)
    parser.add_argument("--recent_size", type=int, default=1024)
    parser.add_argument("--cache_size", type=int, default=2048)
    # chunk inference (to avoid oom)
    parser.add_argument("--chunk_infer", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--no_cache", action="store_true")
    # xformers and flashattention
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--enable_flash", action="store_true")
    # bnb quantization
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()
    return args

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if "llama-32k" in model_name_or_path: 
        # disable modeling_flash_llama 
        config.auto_map = {}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_flash_attention_2=args.enable_flash,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    model.eval()
    return config, model

@torch.no_grad()
def benchmark_step(model, input_ids, max_gen_len):
    prompt_length = input_ids.size()[-1]
    input_ids = input_ids.to(model.device)
    use_cache = not args.no_cache
    # chunk infer
    if args.chunk_infer:
        chunk_size = args.chunk_size
        iter_num = (prompt_length+chunk_size-1) // chunk_size
        input_chunks = [input_ids[:,i * chunk_size: (i+1) * chunk_size] for i in range(iter_num)]
    else:
        input_chunks = [input_ids]
    
    past_key_values = None
    for input_chunk in input_chunks:
        outputs = model(
            input_ids=input_chunk,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        past_key_values = outputs.past_key_values

    # generate
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        past_key_values = outputs.past_key_values
            
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    torch.cuda.synchronize()

device = "cuda"
args = parse_args()
print(args)

config, model = load(args.model_name_or_path)
vocab_size=config.vocab_size
assert "llama" in model.config.model_type
assert not (args.no_cache and args.chunk_infer) 

if args.enable_kmeans:
    from streaming_llm.pos_shift.kmeans_llama import enable_llama_kmeans_attention
    enable_llama_kmeans_attention(model, args.start_size, args.recent_size, args.cache_size)
elif args.enable_kmeans_v2:
    from streaming_llm.pos_shift.kmeans_llama import enable_llama_kmeans_attention_v2
    enable_llama_kmeans_attention_v2(model, args.start_size, args.recent_size, args.cache_size)
elif args.enable_xformers:
    from streaming_llm.pos_shift.modify_llama import enable_llama_xops_attention
    enable_llama_xops_attention(model)

input_ids = torch.randint(0, vocab_size-1, (args.bs, args.input), dtype=torch.long)
max_gen_len = args.output

iter_times = []
tokens_per_sec = []
for x in range(args.num_iters + 1):
    stime = time.time()
    benchmark_step(model, input_ids, max_gen_len)
    t = time.time() - stime
    iter_times.append(t)
    tokens_per_sec.append(max_gen_len/t)
    print('Iter #%d: %.3f seconds, %.3f tokens/second' % (x, iter_times[-1], tokens_per_sec[-1]))

iter_times = iter_times[1:]
tokens_per_sec = tokens_per_sec[1:]

# Results
print('Iteraction time: %.3f +-%.3f seconds' % (np.mean(iter_times), 1.96*np.std(iter_times)))
print('Throughput: %.3f +-%.3f tokens/second' % (np.mean(tokens_per_sec), 1.96*np.std(tokens_per_sec)))

