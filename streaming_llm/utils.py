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
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--enable_pos_abs", action="store_true")
    parser.add_argument("--enable_pos_inf", action="store_true")
    parser.add_argument("--enable_kmeans_attention", action="store_true")
    parser.add_argument("--chunk_infer", action="store_true")
    parser.add_argument("--scaling_factor", type=float, default=0)
    parser.add_argument("--no_sliding_layers", type=int, default=0)
    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load(model_name_or_path, factor=0):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(model_name_or_path)
    if factor > 0 and not hasattr(config, "rope_scaling"):
        config.rope_scaling = {"type": "dynamic", "factor": factor}
    if hasattr(config, "auto_map"):
        config.auto_map = {} #disable modeling_flash_llama
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

