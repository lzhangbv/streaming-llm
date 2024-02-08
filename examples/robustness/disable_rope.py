import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def rope_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    super(LlamaRotaryEmbedding, self).__init__()

    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
    n = int((dim // 2) * 0.5)
    inv_freq[-n:] = 0
    #print("Disable rope in some dimensions.")
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Build here to make `torch.jit.trace` work.
    self._set_cos_sin_cache(
        seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
    )

if __name__ == "__main__":
    LlamaRotaryEmbedding.__init__ = rope_init
    model_id = "lmsys/vicuna-7b-v1.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()

    # test
    text = "What is deep learning?"
    #text = "What is deep learning, and what is the major difference between deep leanring and machine learning?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

