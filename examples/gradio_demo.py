import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()
    return args

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def generation(prompt):
    # use chatglm3 as an example
    output, history = model.chat(tokenizer, prompt, history=[])
    return output

args = parse_args()
print(args)

model, tokenizer = load(args.model_name_or_path)
demo = gr.Interface(fn=generation, inputs="text", outputs="text")
demo.launch(show_api=False)   
#demo.launch(share=True)   


