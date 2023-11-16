import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

args = parse_args()
print(args)

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path, factor=args.scaling_factor)
rwkv = "rwkv" in args.model_name_or_path

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        layer_id=args.no_sliding_layers,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    assert args.enable_start_recent_kv_cache
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")
elif args.enable_pos_abs:
    assert args.enable_start_recent_kv_cache
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_abs_attention

        enable_llama_pos_abs_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
elif args.enable_pos_inf:
    assert not args.enable_start_recent_kv_cache
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_inf_attention

        enable_llama_pos_inf_attention(model, args.start_size, args.recent_size)
    else:
        raise ValueError(f"got {model.config.model_type}")
elif args.enable_kmeans_attention:
    assert not args.enable_start_recent_kv_cache
    if "llama" in model.config.model_type: 
        #from streaming_llm.pos_shift.modify_llama import enable_llama_kmeans_attention
        from streaming_llm.pos_shift.kmeans_llama import enable_llama_kmeans_attention

        enable_llama_kmeans_attention(model, args.start_size, args.recent_size)
    else:
        raise ValueError(f"got {model.config.model_type}")

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for it, text in enumerate(data["text"][: args.num_samples]):
    try:
        encodings = tokenizer(text, return_tensors="pt")
    except:
        print("Tokenization error for the text: ", text)
        continue

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    if seq_len < 4:
        continue
    print(f"iter: {it}, seq_len: {seq_len}, eval_seq_len: {num_eval_tokens}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            if rwkv:
                outputs = model(input_ids, state=past_key_values, use_cache=True)
                past_key_values = outputs.state
            else:
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)

            logits = outputs.logits.view(-1, model.config.vocab_size)
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

log_lens = [1, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
log_lens = [item for item in log_lens if item <= args.num_eval_tokens]
assert len(nlls) >= log_lens[-1]

log_ppl = []
for log_len in log_lens:    
    ppl = torch.exp(torch.stack(nlls[:log_len]).mean())
    log_ppl.append(ppl.item())

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())

logfile = os.path.join(args.output_dir, "ppl_sliding{}_posShift{}_posAbs{}_posInf{}_Kmeans{}.log".format( 
    args.enable_start_recent_kv_cache, args.enable_pos_shift, args.enable_pos_abs, args.enable_pos_inf, args.enable_kmeans_attention))


with open(logfile, "w") as f:
    for i in range(len(log_lens)):
        f.write(f"input length: {log_lens[i]:.0f}, ppl: {log_ppl[i]:.2f}\n")
