import torch
from tqdm import tqdm
import os
import json
import re
import string
from typing import List
import regex
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--task", type=str, default="lines", choices=["lines", 'topics'])
    # kmeans params
    parser.add_argument("--enable_kmeans_attention", action="store_true")
    parser.add_argument("--start_size", type=int, default=256)
    parser.add_argument("--recent_size", type=int, default=1024)
    parser.add_argument("--cache_size", type=int, default=2048)
    # chunk inference (to avoid oom)
    parser.add_argument("--chunk_infer", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1024)

    args = parser.parse_args()
    return args    

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
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
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()
    return model, tokenizer    

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)
    return test_cases

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_gen_len):
    # prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = input_ids.size()[-1]
    input_ids = input_ids.to(model.device)

    # chunk infer
    if args.chunk_infer:
        chunk_size = args.chunk_size
        iter_num = (prompt_length+chunk_size-1) // chunk_size
        input_chunks = [input_ids[:,i * chunk_size: (i+1) * chunk_size] for i in range(iter_num)]
    else:
        input_chunks = [input_ids]
    
    past_key_values = None
    for input_chunk in input_chunks:
        if rwkv:
            outputs = model(input_chunk, state=past_key_values, use_cache=True)
            past_key_values = outputs.state
        else:
            outputs = model(
                input_ids=input_chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

    # generate
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for _ in range(max_gen_len - 1):
        if rwkv: 
            outputs = model(pred_token_idx, state=past_key_values, use_cache=True)
            past_key_values = outputs.state
        else:
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt_length, generated_text


device = "cuda"
args = parse_args()
print(args)

model, tokenizer = load(args.model_name_or_path)
rwkv = "rwkv" in args.model_name_or_path

if args.enable_kmeans_attention:
    assert "llama" in model.config.model_type
    from streaming_llm.pos_shift.kmeans_llama import enable_llama_kmeans_attention
    enable_llama_kmeans_attention(model, args.start_size, args.recent_size, args.cache_size)

if args.task == "topics":
    total_num_topics = [5] #[5, 10, 15, 20, 25]
    for num_topics in total_num_topics: 
        print(f"************ Start testing {num_topics} topics per prompt ***********")
        num_correct = 0
        avg_length = 0

        test_file = os.path.join(args.dataset_path, f"topics/testcases/{num_topics}_topics.jsonl")

        test_cases = load_testcases(test_file)
        for idx, test_case in tqdm(enumerate(test_cases)):
            prompt = test_case["prompt"]
            topics = test_case["topics"]

            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path: 
                prompt = prompt + "\n ASSISTANT: "

            # inference
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=50)

            avg_length += prompt_length / len(test_cases)
            
            correct = best_subspan_em(prediction=output, ground_truths=[topics[0]])
            num_correct += correct
            
            summary = f"Label: {topics[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)
        
        accuracy = num_correct / len(test_cases)
        print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
elif args.task == "lines":
    total_num_lines = [200]  #[200, 300, 400, 500, 600, 700]
    for num_lines in total_num_lines:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")
        num_correct = 0
        avg_length = 0

        test_file = os.path.join(args.dataset_path, f"lines/testcases/{num_lines}_lines.jsonl")

        test_cases = load_testcases(test_file)
        for idx, test_case in tqdm(enumerate(test_cases)):
            prompt = test_case["prompt"]
            correct_line = test_case["correct_line"]
            expected_number = test_case["expected_number"]
            
            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                prompt = prompt + "\n ASSISTANT: "

            # inference
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=15)

            # Matching the last digit of the model output
            response_number = re.findall("\d+", output)
            if response_number is not None and len(response_number) > 0:
                #response_number = int(response_number[-1])
                response_number = int(response_number[0])
            else:
                print(f"Got unparsable result")
                response_number = -1

            avg_length += prompt_length / len(test_cases)
            correct = (expected_number == response_number)
            num_correct += correct
            
            summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)
            
        accuracy = num_correct / len(test_cases)
        print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
