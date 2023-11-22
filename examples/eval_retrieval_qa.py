import torch
from tqdm import tqdm
import os
import json
import argparse
import re
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
from streaming_llm.utils import load, best_subspan_em
from streaming_llm.normalize_text import normalize
from streaming_llm.splitter import split_long_sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/llama/llama-7b")
    parser.add_argument("--retrieval_name_or_path", type=str, default="models/contriever")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--method", type=str, default="retrieval", choices=["retrieval", "nbce"])
    parser.add_argument("--eval_num", type=int, default=50)

    parser.add_argument("--split_text", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--normalize_embed", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--closedbook", action="store_true")

    args = parser.parse_args()
    return args

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)
    return test_cases

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

@torch.no_grad()
def embed_passages(model, tokenizer, question, passages, args):
    if args.lowercase:
        question = question.lower()
    if args.normalize_text:
        question = normalize(question)

    batch = [question]
    for p in passages:
        if args.no_title or not "title" in p:
            text = p["text"]
        else:
            title = p["title"]
            text = p["text"]
            text = title + ": " + text
        
        if args.lowercase:
            text.lower()
        
        if args.normalize_text:
            text = normalize(text)
        
        if args.split_text:
            chunks = split_long_sentence(text, args.chunk_size)
        else:
            chunks = [text]

        batch.extend(chunks)
    
    print("Total document number: ", len(batch)-1)
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embs = mean_pooling(outputs[0], inputs['attention_mask'])

    if args.normalize_embed:
        embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs, batch[1:]

def select_topk(q, embs, topk=5):
    scores = q @ embs.T
    _, index = torch.topk(scores, k=topk)
    return index.tolist()

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_gen_len):
    # prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = input_ids.size()[-1]
    input_ids = input_ids.to(model.device)

    outputs = model(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values

    # generate
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for _ in range(max_gen_len - 1):
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

@torch.no_grad()
def parallel_generate(model, tokenizer, batch, max_gen_len):
    """Naive Bayes-based Context Extension"""
    inputs = tokenizer(batch, padding='longest', return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    past_key_values = None
    n = input_ids.shape[0]
    prompt_length = input_ids.size()[-1]
    
    # generate
    generated_ids = []
    for i in range(max_gen_len):
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values
                       )
        past_key_values = outputs.past_key_values
        
        # ===== nbce =====
        beta, eta = 0.25, 0.1
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
        if i > 0:
            entropy[k] -= eta
        k = entropy[1:].argmin() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
        # ===== nbce =====
        
        # sampling
        #tau = 0.01
        #probas = torch.nn.functional.softmax(logits[None] / tau , dim=-1)
        #next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)

        # greedy
        next_tokens = torch.argmax(logits, keepdims=True)
        
        generated_ids.append(next_tokens[0].item())
        if next_tokens[0] == tokenizer.eos_token_id:
            break
            
        #ret = tokenizer.batch_decode(next_tokens)
        #print(ret[0], flush=True, end='')
        
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1) 
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt_length, generated_text

device = "cuda"
args = parse_args()
print(args)

total_num_docs = [10] #[10, 20, 30]
model, tokenizer = load(args.model_name_or_path)

if args.method == "retrieval":
    # load retrieval model
    retrieval_model = AutoModel.from_pretrained(args.retrieval_name_or_path)
    retrieval_model.eval()
    retrieval_tokenizer = AutoTokenizer.from_pretrained(args.retrieval_name_or_path)
    
    # multiple docuemnt QA
    for num_docs in total_num_docs: 
        print(f"************ Start testing {num_docs} documents QA ***********")
        num_correct = 0
        avg_length = 0

        test_file = os.path.join(args.dataset_name, f"nq-open-{num_docs}_total_documents_gold_at_0.jsonl")
        test_cases = load_testcases(test_file)
        
        eval_num = min(args.eval_num, len(test_cases))
        for idx, test_case in enumerate(test_cases):
            question = test_case["question"]
            answers = test_case["answers"]
            passages = test_case["ctxs"]
            
            # prompt
            if args.closedbook:
                prompt = 'Question: ' + question
            else:
                prompt = 'Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n'
                
                # topk retrieval
                embs, chunks = embed_passages(retrieval_model, retrieval_tokenizer, question, passages, args)
                index = select_topk(embs[0], embs[1:], topk=args.topk)
                print("Topk document index: ", index)

                formatted_documents = []
                titles = []
                for document_id, select_id in enumerate(index):
                    text = chunks[select_id]
                    formatted_documents.append(f"Document {document_id+1}: {text}")

                prompt += "\n".join(formatted_documents)
                # qa
                prompt += '\n\nQuestion: ' + question
                #print(prompt)
            
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                prompt += '\n Assistant: '
            else:
                prompt += '\nAnswer:'

            prompt_length, output = greedy_generate(model, tokenizer, prompt, max_gen_len=50)

            avg_length += prompt_length / eval_num
            correct = best_subspan_em(prediction=output, ground_truths=answers)

            num_correct += correct
            
            summary = f"Label: {answers[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)

            if idx + 1 > eval_num:
                break

        accuracy = num_correct / eval_num
        print(f"************ Finish testing {num_docs} documents QA per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
elif args.method == "nbce":
    from transformers import TopPLogitsWarper, LogitsProcessorList
    processors = LogitsProcessorList()
    processors.append(TopPLogitsWarper(0.95))
    tokenizer.padding_side = 'left'

    for num_docs in total_num_docs: 
        print(f"************ Start testing {num_docs} documents QA ***********")
        num_correct = 0
        avg_length = 0

        test_file = os.path.join(args.dataset_name, f"nq-open-{num_docs}_total_documents_gold_at_0.jsonl")
        test_cases = load_testcases(test_file)
        
        eval_num = min(args.eval_num, len(test_cases))
        for idx, test_case in enumerate(test_cases):
            question = test_case["question"]
            answers = test_case["answers"]
            passages = test_case["ctxs"]
            
            # prompt
            if "vicuna" in args.model_name_or_path or "chat" in args.model_name_or_path:
                batch = ['Question: ' + question + '\n Assistant: '] + [p["text"] + '\n Question: ' + question + '\n Assistant: ' for p in passages]
            else:
                batch = ['Question: ' + question + '\n Answer: '] + [p["text"] + '\n Question: ' + question + '\n Answer: ' for p in passages]
            
            #print(batch)
            prompt_length, output = parallel_generate(model, tokenizer, batch, max_gen_len=50)

            avg_length += prompt_length / eval_num
            correct = best_subspan_em(prediction=output, ground_truths=answers)

            num_correct += correct
            
            summary = f"Label: {answers[0]}, Predict: {output}, Correct: {correct}, prompt length: {prompt_length}".replace('\n', ' ')
            print(summary)

            if idx + 1 > eval_num:
                break

        accuracy = num_correct / eval_num
        print(f"************ Finish testing {num_docs} documents QA per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")


