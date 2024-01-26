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
    # ngram speculation
    parser.add_argument('--max_ngram_size', type=int, default=3)
    parser.add_argument('--speculate_k', type=int, default=10)
    # support multi-turn chat
    parser.add_argument('--multi_turn', action="store_true") 
    
    args = parser.parse_args()
    return args

def load_model(model): 
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

    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map='auto',
        trust_remote_code=True,
    )

    model.eval()    
    return tokenizer, model

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

def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=5):
    # based on: https://github.com/apoorvumang/prompt-lookup-decoding
    input_length = input_ids.size(1)
    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)
        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]
        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]
    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)

def speculative_decode(args, input_tokens, model, cur_token, past_key_values):
    # ngram speculation
    draft_tokens = find_candidate_pred_tokens(input_tokens.unsqueeze(0), args.max_ngram_size, args.speculate_k)
    if len(draft_tokens) == 0:
        draft_tokens = torch.tensor([100], dtype=cur_token.dtype, device=cur_token.device)
    
    candidate_input_ids = torch.cat([cur_token.view(1), draft_tokens]).unsqueeze(0)
    candidate_length = candidate_input_ids.shape[1] - 1

    # parallel inference
    outputs = model(
        candidate_input_ids, 
        use_cache=True, 
        past_key_values=past_key_values
        )
    past_key_values = outputs.past_key_values
    selected_tokens = outputs.logits[:, -candidate_length-1:].argmax(dim=-1)

    # validation
    candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
    n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
    valid_tokens = selected_tokens[0, :n_matches + 1]
    return valid_tokens, past_key_values

def rollback_kv_cache(past_key_values, n):
    # kv cache shape: [bzs, head_num, seq_len, head_dim]
    if past_key_values is None:
        return None
    else:
        return [
            [k[:, :, :n, :], v[:, :, :n, :]] for k, v in past_key_values
        ] 

@torch.no_grad()
def generate(args, model, encoded, eos_token_id):
    T = encoded.shape[1] #(bsz, seq_len)
    assert encoded.shape[0] == 1

    T_new = T + args.max_new_tokens
    seq = torch.empty(T_new, dtype=encoded.dtype, device=encoded.device)
    seq[:T] = encoded[0]

    # prefill
    outputs = model(encoded, use_cache=True, past_key_values=None)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[0, -1].argmax(dim=-1)
    seq[T] = next_token

    # decode
    accept_counts = [0] * (args.speculate_k + 1)
    input_pos = T
    while input_pos < T_new - 1:
        cur_token = next_token
        input_tokens = seq[:input_pos+1]
        next_tokens, past_key_values = speculative_decode(args, input_tokens, model, cur_token, past_key_values)
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
        # eos
        if eos_location.shape[0]:
            return seq[:input_pos+1], accept_counts
    return seq, accept_counts

def main(args):
    user_prompts = load_mt_bench(args)
    tokenizer, model = load_model(args.model_name_or_path)

    accept_counts = []
    accept_counts2 = []
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
        outputs, accept_count = generate(args, model, encoded=input_ids, eos_token_id=tokenizer.eos_token_id)

        if args.multi_turn and idx % 2 == 1: 
            accept_counts2.append(accept_count)
        else:
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

    if args.multi_turn:
        counts_aggregated = [sum(i) for i in zip(*accept_counts2)]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Second-turn acceptance probs: {acceptance_probs}")
        print(f"Second-turn mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
