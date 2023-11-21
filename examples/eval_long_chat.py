import torch
from tqdm import tqdm
import os
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import numpy as np
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load, best_subspan_em, generate_prompt_landmark, generate_pos_id

device = "cuda"

args = parse_args()
print(args)

model, tokenizer = load(args.model_name_or_path, factor=args.scaling_factor)
rwkv = "rwkv" in args.model_name_or_path

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

        enable_llama_kmeans_attention(model, args.start_size, args.recent_size, args.cache_size)
    else:
        raise ValueError(f"got {model.config.model_type}")

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)
    return test_cases


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_gen_len, kv_cache_evict=None):
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
            if kv_cache_evict is not None:
                past_key_values = kv_cache_evict(past_key_values)

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
            if kv_cache_evict is not None:
                past_key_values = kv_cache_evict(past_key_values)
            
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt_length, generated_text

if args.task == "topics":
    total_num_topics = [5] #[5, 10, 15, 20, 25]
    for num_topics in total_num_topics: 
        print(f"************ Start testing {num_topics} topics per prompt ***********")
        num_correct = 0
        num_correct_list = [0] * num_topics
        avg_length = 0

        test_file = os.path.join(args.dataset_name, f"topics/testcases/{num_topics}_topics.jsonl")

        test_cases = load_testcases(test_file)
        for idx, test_case in tqdm(enumerate(test_cases)):
            prompt = test_case["prompt"]
            topics = test_case["topics"]

            # retrieval all topics
            if args.retrieval_all:
                prompt = prompt.replace("What is the first topic", "What are all topics")
                prompt = prompt.replace("the first topic", "all topics")

            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path: 
                prompt = prompt + "\n ASSISTANT: "

            # streaming inference
            max_gen_len = num_topics * 20 if args.retrieval_all else 20
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=max_gen_len, kv_cache_evict=kv_cache)

            avg_length += prompt_length / len(test_cases)
            
            if args.retrieval_all:
                correct = 1
                for i in range(num_topics):
                    match = best_subspan_em(prediction=output, ground_truths=[topics[i]])
                    num_correct_list[i] += match
                    if not match:
                        correct = 0
            else:
                correct = best_subspan_em(prediction=output, ground_truths=[topics[0]])
            num_correct += correct
            
            summary = f"The first topic: {topics[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)
        
        accuracy = num_correct / len(test_cases)
        print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
        if args.retrieval_all:
            num_correct_list = [i/len(test_cases) for i in num_correct_list]
            print("Accuracy per topic:", num_correct_list)
            print("Average accuracy:", np.mean(num_correct_list))
elif args.task == "lines":
    total_num_lines = [200]  #[200, 300, 400, 500, 600, 700]
    for num_lines in total_num_lines:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")
        num_correct = 0
        avg_length = 0

        test_file = os.path.join(args.dataset_name, f"lines/testcases/{num_lines}_lines.jsonl")

        test_cases = load_testcases(test_file)
        for idx, test_case in tqdm(enumerate(test_cases)):
            prompt = test_case["prompt"]
            correct_line = test_case["correct_line"]
            expected_number = test_case["expected_number"]
            
            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                prompt = prompt + "\n ASSISTANT: "

            # streaming inference
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=15, kv_cache_evict=kv_cache)

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
elif args.task == "passkey":
    n_garbages = [10000] #[10000, 20000, 30000, 40000, 50000, 60000]
    for n_garbage in n_garbages:
        print(f"************ Start testing passkey retrieval with {n_garbage} garbage texts ************")
        num_correct = 0
        avg_length = 0
        seed = 42
        num_iter = 50
        
        for idx in range(num_iter):
            prompt, answer = generate_prompt_landmark(n_garbage, seed+idx)

            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                prompt += "\n ASSISTANT: The pass key is"
            else:
                prompt += "The pass key is"

            # streaming inference
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=15, kv_cache_evict=kv_cache)

            # Matching the last digit of the model output
            response_number = re.findall("\d+", output)
            if response_number is not None and len(response_number) > 0:
                #response_number = int(response_number[-1])
                response_number = int(response_number[0])
            else:
                print(f"Got unparsable result")
                response_number = -1

            avg_length += prompt_length / num_iter
            correct = (answer == response_number)
            num_correct += correct
            
            summary = f"Label: {answer}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)
            
        accuracy = num_correct / num_iter
        print(f"************ Finish testing passkey retrieval per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
elif args.task == "qa":
    total_num_docs = [10] #[10, 20, 30]
    for num_docs in total_num_docs: 
        print(f"************ Start testing {num_docs} documents QA ***********")
        num_correct = 0
        num_correct_list = [0] * num_docs
        avg_length = 0
        closedbook = False

        test_file = os.path.join(args.dataset_name, f"nq-open-{num_docs}_total_documents_gold_at_{args.topic_id}.jsonl")
        test_cases = load_testcases(test_file)
        
        eval_num = 50
        for idx, test_case in enumerate(test_cases):
            question = test_case["question"]
            answers = test_case["answers"]
            
            # document title retrieval
            if args.retrieval_all:
                question = "Please list all document titles."

            # prompt
            if closedbook:
                prompt = 'Question: ' + question
            else:
                prompt = 'Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n'
                
                # query aware contextualization
                #prompt += "Question: " + question + "\n\n"

                formatted_documents = []
                titles = []
                for document_id, ctx in enumerate(test_case["ctxs"]):
                    title = ctx["title"]
                    text = ctx["text"]
                    formatted_documents.append(f"Document [{document_id+1}](Title: {title}) {text}")
                    titles.append(title)
                    # break # add support document only

                prompt += "\n".join(formatted_documents)
                # qa
                prompt += '\n\nQuestion: ' + question
                # prompt: give me source
            
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                prompt += '\n Assistant: '
            else:
                prompt += '\nAnswer:'

            #print(prompt)
            
            # streaming inference
            max_gen_len = num_docs * 20 if args.retrieval_all else 50
            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=max_gen_len, kv_cache_evict=kv_cache)

            avg_length += prompt_length / eval_num
            if args.retrieval_all:
                correct = 1
                for i in range(num_docs):
                    match = best_subspan_em(prediction=output, ground_truths=[titles[i]])
                    num_correct_list[i] += match
                    if not match:
                        correct = 0
            else:
                correct = best_subspan_em(prediction=output, ground_truths=answers)

            num_correct += correct
            
            if args.retrieval_all: 
                summary = f"The first title: {titles[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            else:
                summary = f"Label: {answers[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)

            if idx + 1 > eval_num:
                break

        accuracy = num_correct / eval_num
        print(f"************ Finish testing {num_docs} documents QA per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
        if args.retrieval_all: 
            num_correct_list = [i/eval_num for i in num_correct_list]
            print("Accuracy per document:", num_correct_list)
            print("Average accuracy:", np.mean(num_correct_list))


