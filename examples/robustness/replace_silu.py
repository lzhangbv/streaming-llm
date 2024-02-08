from torch import nn
from torch.nn import functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN

sparse_rates = []

class Sparse_SiLU(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        assert threshold > 0
        self.threshold = threshold

    def forward(self, x):
        # set outputs between [-threshold, 0] to zeros, where ymin=−0.28 at x=-1.28
        y = F.silu(x)
        idx = (y < 0) & (y >= -self.threshold)
        sparse_rates.append((idx.sum() / idx.numel()).item())
        y[idx] = 0
        return y

if __name__ == "__main__":
    # replace silu
    #ACT2FN["silu"] = nn.ReLU # it does not work
    ACT2FN["silu"] = Sparse_SiLU

    model_id = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()

    # test
    #text = "What is deep learning?"
    text = "What is deep learning, and what is the major difference between deep leanring and machine learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"sparse rate: {np.mean(sparse_rates):.3f}")
