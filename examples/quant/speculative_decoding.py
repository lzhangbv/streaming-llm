import torch
import os
import json
from typing import List
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--top_k', type=float, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    # speculative decoding
    parser.add_argument('--draft_name_or_path', type=str)
    parser.add_argument('--speculate_k', type=int, default=5)
    # hf assisted decoding
    parser.add_argument('--hf_assisted', action="store_true") 
    # support multi-turn chat
    parser.add_argument('--multi_turn', action="store_true") 
    
    args = parser.parse_args()
    return args

def load_model(model, draft_model): 
    print(f"Loading model from {model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    if "gptq" in model: 
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(model, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map='auto',
            trust_remote_code=True,
        )

    if "gptq" in draft_model: 
        from auto_gptq import AutoGPTQForCausalLM
        draft_model = AutoGPTQForCausalLM.from_quantized(draft_model, device_map='auto')
    else:
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model,
            device_map='auto',
            trust_remote_code=True,
            #load_in_8bit=True,
            #load_in_4bit=True,
        )

    model.eval()    
    draft_model.eval()
    return tokenizer, model, draft_model

def load_jsonl(file_path):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

def load_mt_bench(args):
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    
    list_data = load_jsonl(test_filepath)
    if args.num_samples is None:
        args.num_samples = len(list_data)
    
    prompts = []
    for idx, example in enumerate(list_data):
        if args.multi_turn: 
            prompts += example["turns"]  # add two turns
        else:
            prompts.append(example["turns"][0])  # add first turn
        if (idx + 1) == args.num_samples:
            break
    return  prompts

def multinomial_sample_one_no_sync(probs_sort): 
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature, top_k):
    #logits: (1, seq, nclass), probs: (nclass), idx_next: (1)
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def decode_n_tokens(args, model, cur_token, past_key_values, num_new_tokens):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        outputs = model(cur_token, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        next_token, next_prob = sample(outputs.logits, args.temperature, args.top_k)
        new_tokens.append(next_token.clone())
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(1, -1)
    return new_tokens, new_probs, past_key_values

def speculative_decode(args, model, draft_model, cur_token, input_pos, past_key_values, draft_past_key_values):
    # draft model inference sequentially
    device = cur_token.device
    draft_tokens, draft_probs, draft_past_key_values = decode_n_tokens(args, draft_model, cur_token.view(1, -1), draft_past_key_values, args.speculate_k)
    draft_tokens = torch.cat(draft_tokens)
    draft_probs = torch.stack(draft_probs) #[k, nclass]

    # parallel inference on target model using draft tokens
    outputs = model(
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1), 
        use_cache=True, 
        past_key_values=past_key_values
        )
    past_key_values = outputs.past_key_values
    target_probs = logits_to_probs(outputs.logits[0], args.temperature, args.top_k) #[k, nclass]

    # q >= p: always accept draft token; q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, args.speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, args.speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:args.speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = args.speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        outputs = draft_model(
            draft_tokens[-1].view(1, -1), 
            use_cache=True, 
            past_key_values=draft_past_key_values
            )
        draft_past_key_values = outputs.past_key_values
        return torch.cat([draft_tokens, last_token]), past_key_values, draft_past_key_values
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token]), past_key_values, draft_past_key_values 

def rollback_kv_cache(past_key_values, n):
    # kv cache shape: [bzs, head_num, seq_len, head_dim]
    if past_key_values is None:
        return None
    else:
        return [
            [k[:, :, :n, :], v[:, :, :n, :]] for k, v in past_key_values
        ] 

@torch.no_grad()
def generate(args, model, draft_model, encoded, eos_token_id):
    T = encoded.shape[1] #(bsz, seq_len)
    assert encoded.shape[0] == 1

    T_new = T + args.max_new_tokens
    seq = torch.empty(T_new, dtype=encoded.dtype, device=encoded.device)
    seq[:T] = encoded[0]

    # prefill
    outputs = model(encoded, use_cache=True, past_key_values=None)
    past_key_values = outputs.past_key_values
    next_token = sample(outputs.logits, args.temperature, args.top_k)[0] #tensor
    seq[T] = next_token
    outputs = draft_model(encoded, use_cache=True, past_key_values=None)
    draft_past_key_values = outputs.past_key_values

    # decode
    accept_counts = [0] * (args.speculate_k + 1)
    input_pos = T
    while input_pos < T_new - 1:
        cur_token = next_token
        next_tokens, past_key_values, draft_past_key_values = speculative_decode(args, model, draft_model, cur_token, input_pos, past_key_values, draft_past_key_values)
        # eos 
        eos_location = (next_tokens == eos_token_id).nonzero()
        if eos_location.shape[0]: 
            next_tokens = next_tokens[:eos_location[0].item() + 1]

        accept_counts[len(next_tokens) - 1] += 1
        num_added = min(T_new - input_pos - 1, len(next_tokens))
        seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
        input_pos = input_pos + num_added
        next_token = next_tokens[-1]
        # rollback kv cache
        past_key_values = rollback_kv_cache(past_key_values, input_pos)
        draft_past_key_values = rollback_kv_cache(draft_past_key_values, input_pos)
        # eos
        if eos_location.shape[0]:
            return seq[:input_pos+1], accept_counts
    return seq, accept_counts


def main(args):
    user_prompts = load_mt_bench(args)
    tokenizer, model, draft_model = load_model(args.model_name_or_path, args.draft_name_or_path)

    accept_counts = []
    prompt = None
    for idx, user_prompt in enumerate(user_prompts):
        print("\n" + user_prompt)
        if prompt is None: # first-turn
            if "llama2" in args.model_name_or_path: 
                system = "You are a helpful, respectful and honest assistant. "
                prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_prompt}[/INST]"
            else:
                prompt = f"User: {user_prompt}\nAssistant: "
        else: # second-turn
            if "llama2" in args.model_name_or_path:
                prompt += f"[INST] {user_prompt} [/INST]"
            else:
                prompt += f"User: {user_prompt}\nAssistant: "
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        outputs, accept_count = generate(args, model, draft_model, encoded=input_ids, eos_token_id=tokenizer.eos_token_id)
        accept_counts.append(accept_count)
        
        T = input_ids.shape[1]
        answer = tokenizer.decode(outputs[T:].tolist(), skip_special_tokens=True)
        print(answer)

        if args.multi_turn and idx % 2 == 0:
            prompt += answer
        else:
            prompt = None 

    counts_aggregated = [sum(i) for i in zip(*accept_counts)]
    acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
    print(f"Acceptance probs: {acceptance_probs}")
    print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

def hf_assisted(args):
    prompts = load_mt_bench(args)
    tokenizer, model, draft_model = load_model(args.model_name_or_path, args.draft_name_or_path)

    model.config.use_cache = True
    draft_model.config.use_cache = True

    for idx, prompt in enumerate(prompts):
        #print("\n" + prompt, end="")
        if "llama2" in args.model_name_or_path: 
            system = "You are a helpful, respectful and honest assistant. "
            prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt}[/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, assistant_model=draft_model, max_new_tokens=args.max_new_tokens)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.hf_assisted:
        hf_assisted(args)
    else:
        main(args)

