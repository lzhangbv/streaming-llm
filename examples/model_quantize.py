from transformers import AutoTokenizer
import json
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str)
    parser.add_argument("--method", type=str, choices=['gptq', 'awq'])

    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=2048)

    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--sym', action="store_true")

    args = parser.parse_args()
    return args

args = parse_args()
print(args)

# Load dataset
# Make sure the dataset is available, for example, we download the Pile dataset from https://huggingface.co/datasets/mit-han-lab/pile-val-backup
print(f'Start loading dataset from {args.data_path}.')
with open(args.data_path, 'r') as json_file:
    json_list = list(json_file)
test_cases = []
for test_case in json_list:
    test_case = json.loads(test_case)['text']
    test_cases.append(test_case)

# Load tokenization
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir, trust_remote_code=True)

if args.method == "gptq":
    test_cases = test_cases[:args.num_samples]
    examples = [tokenizer(test_case, max_length=args.max_length, truncation=True) for test_case in test_cases]

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    quantize_config = BaseQuantizeConfig(
        bits=args.bits, 
        group_size=args.group_size,  
        desc_act=True,  
        sym=args.sym, 
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(args.pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    print('Start GPTQ quantization.')
    stime = time.time()
    model.quantize(examples)
    print(f"Quantization time: {time.time() - stime}")

    # save quantized model
    model.save_quantized(args.quantized_model_dir)
    tokenizer.save_pretrained(args.quantized_model_dir)
elif args.method == "awq":
    from awq import AutoAWQForCausalLM

    quant_config = { "zero_point": not args.sym, "q_group_size": args.group_size, "w_bit": args.bits, "version": "GEMM" }

    # Load un-quantized model
    model = AutoAWQForCausalLM.from_pretrained(args.pretrained_model_dir)

    # quantize model (default: n_samples=128, seqlen=512)
    print('Start AWQ quantization.')
    stime = time.time()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=test_cases)
    print(f"Quantization time: {time.time() - stime}")

    # Save quantized model
    model.save_quantized(args.quantized_model_dir)
    tokenizer.save_pretrained(args.quantized_model_dir)
