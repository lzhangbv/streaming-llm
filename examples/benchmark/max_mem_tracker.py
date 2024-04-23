import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref
import math

"""
Measure memory usage without actually using any memory. 

based on: https://github.com/albanD/subclass_zoo/blob/main/max_mem_tracker.py
"""

# Track all the memory being used by Tensors.
# Only max is tracked but others can be added.
MEMORY_USE = WeakIdKeyDictionary()
MEMORY_MAX = 0

# Minimum allocation size 
PYTORCH_MIN_ALLOCATE = 2**9

def update_stats():
    global MEMORY_MAX
    curr_use = 0
    for k, v in MEMORY_USE.items():
        curr_use += math.ceil(k.size() * k.element_size()/PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE

    if MEMORY_MAX < curr_use:
        MEMORY_MAX = curr_use

# Should be called on every Tensor created
def track(t:torch.Tensor):
    def cb(_):
        update_stats()
    st = t.untyped_storage()
    wt = weakref.ref(st, cb)
    MEMORY_USE[st] = wt
    update_stats()

# Use this Mode to call track on every Tensor being created by functions
class MemoryTrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        res = func(*args, **kwargs or {})

        tree_map_only(torch.Tensor, track, res)
        return res


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Use FakeTensorMode to run the code without any actual data
    with FakeTensorMode(), MemoryTrackingMode():
        model_id = "lmsys/vicuna-7b-v1.3"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(
            config, 
            torch_dtype=torch.float16, 
            #attn_implementation="flash_attention_2",
        )

        batch_size, seq_len = 1, 2048
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size-1, (batch_size, seq_len), dtype=torch.long)
        outputs = model(input_ids=input_ids, use_cache=True)
        
        print(f"Max memory allocated: {MEMORY_MAX / 1e9:.02f} GB")
        
    print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated()} B")
