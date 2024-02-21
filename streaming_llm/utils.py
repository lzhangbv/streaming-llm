import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json
import numpy as np

import string
from typing import List
import regex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--enable_pos_abs", action="store_true")
    parser.add_argument("--enable_pos_inf", action="store_true")
    parser.add_argument("--enable_rerope", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--enable_blockwise_attention", action="store_true")
    parser.add_argument("--enable_pcw_attention", action="store_true")
    parser.add_argument("--enable_kmeans_attention", action="store_true")
    parser.add_argument("--enable_kmeans_attention_v2", action="store_true")
    parser.add_argument("--chunk_infer", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--scaling_factor", type=float, default=0)
    parser.add_argument("--no_sliding_layers", type=int, default=0)
    parser.add_argument("--num_eval_tokens", type=int, default=None)
    parser.add_argument("--topic_id", type=int, default=0)
    parser.add_argument("--retrieval_all", action="store_true")

    args = parser.parse_args()
    return args


def load(model_name_or_path, factor=0):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if factor > 0 and not hasattr(config, "rope_scaling"):
        config.rope_scaling = {"type": "dynamic", "factor": factor}
    if "llama-32k" in model_name_or_path or "anima" in model_name_or_path:
        # disable modeling_flash_llama 
        config.auto_map = {}
    #if "neural-chat" in model_name_or_path:
    #    # extend sliding window length
    #    config.sliding_window = 32768
    #print(config)
    
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


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


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

def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an execute line at a random position."""
    rnd_state = np.random.get_state()
    np.random.seed(seed)

    n_garbage_prefix = np.random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage

    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]

    pass_key = np.random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    #final_question = "What is the pass key? The pass key is"
    final_question = "What is the pass key?"

    lines = [task_description, garbage_prefix, information_line, garbage_suffix, final_question]
    np.random.set_state(rnd_state)

    return "\n".join(lines), pass_key

def generate_pos_id(n=20, i=0, seed=42):
    pos_maps = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 
            'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth']
    assert len(pos_maps) >= n

    if i < 0:
        rnd_state = np.random.get_state()
        np.random.seed(seed)
        i = np.random.randint(0, n-1)
        np.random.set_state(rnd_state)
    
    return i, pos_maps[i]


