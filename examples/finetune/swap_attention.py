"""
Attention Swap: Activation Offloading. 
"""

import types
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class SwapAttention:
    def __init__(self, stream, limit=None):
        self.gpu_tensor_list = []
        self.cpu_tensor_list = []
        self.transfer_stream = stream

        self.current_gpu_tensors = []
        self.current_cpu_tensors = []
        self.limit = limit  # max number of tensors to offload
    
    def offload(self, gpu_tensor):
        # add tensor to current tensors
        if self.limit is not None and len(self.current_gpu_tensors) >= self.limit:
            return
        self.current_gpu_tensors.append(gpu_tensor)
        self.current_cpu_tensors.append(None)

        #print("----start offload----") 
        # offload activations from gpu to cpu
        self.transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(self.transfer_stream):
                self.current_cpu_tensors[-1] = gpu_tensor.to('cpu', non_blocking=True)
    
    def wait(self):
        # synchronize offload operations
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        #print("----finish offload----")

        # add current tensors to list
        self.gpu_tensor_list.append(self.current_gpu_tensors)
        self.cpu_tensor_list.append(self.current_cpu_tensors)
        self.current_gpu_tensors = []
        self.current_cpu_tensors = []

        # release gpu memory
        gpu_tensors = self.gpu_tensor_list[-1]
        for gpu_tensor in gpu_tensors:    
            gpu_tensor.untyped_storage().resize_(0)

    def register_finish_reload(self, input_tensor):
        def wait_reload(ignore):
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            self.gpu_tensor_list.pop(0)
            self.cpu_tensor_list.pop(0)
            #print("----finish reload----")

        input_tensor.register_hook(wait_reload)
        
    def register_start_reload(self, output_tensor):
        def reload(ignore):
            # FIFO
            gpu_tensors = self.gpu_tensor_list[0]
            cpu_tensors = self.cpu_tensor_list[0]

            #print("----start reload----")
            # reload activations from cpu to gpu
            self.transfer_stream.wait_stream(torch.cuda.current_stream())
            for i in range(len(gpu_tensors)):
                with torch.cuda.stream(self.transfer_stream):
                    _size = cpu_tensors[i].untyped_storage().size()
                    gpu_tensors[i].untyped_storage().resize_(_size)
                    gpu_tensors[i].untyped_storage().copy_(cpu_tensors[i].untyped_storage(), non_blocking=True)

        output_tensor.register_hook(reload)


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    It is based on modeling_llama from huggingface transformers. 
    Logic:
    - Forward: Attention, offload, register_finish_reload, FFN, wait, resigter_start_reload  -->
    - Backward: Attention_BWD,     finish_reload,        , FFN_BWD  , start_reload           <--
    """

    self.swap.offload(hidden_states)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    self.swap.offload(hidden_states)

    bsz, q_len, _ = hidden_states.size()
    query_states = self.self_attn.q_proj(hidden_states)
    key_states = self.self_attn.k_proj(hidden_states)
    value_states = self.self_attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)

    # transformers==3.35.2
    cos, sin = self.self_attn.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # transformers==3.38.1
    # cos, sin = self.self_attn.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.self_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, self.self_attn.num_key_value_groups)

    self.swap.offload(query_states)
    self.swap.offload(key_states)
    self.swap.offload(value_states)

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )

    self.swap.offload(attn_output)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(hidden_states.shape)

    attn_output = self.self_attn.o_proj(attn_output)
    hidden_states = residual + attn_output

    self.swap.register_finish_reload(hidden_states)

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    self.swap.wait()
    self.swap.register_start_reload(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (None,)
    
    if use_cache:
        outputs += (None,)
    
    return outputs


def gelu_recompute_forward(self, x):
    """
    mlp forward with gelu recomputation
    """
    def func(up, gate):
        return self.down_proj(self.act_fn(gate) * up)

    up = self.up_proj(x)
    gate = self.gate_proj(x)
    down = checkpoint(func, up, gate, use_reentrant=False)
    return down


def enable_swap(model, limit=None, gelu_recompute=True):
    transfer_stream = torch.cuda.Stream()

    for layer in model.model.layers:
        layer.swap = SwapAttention(transfer_stream, limit)
        layer.forward = types.MethodType(decoder_layer_forward, layer)

        if gelu_recompute:
            layer.mlp.forward = types.MethodType(gelu_recompute_forward, layer.mlp)


def test(model):
    # inputs
    batch_size, seq_len = 4, 2048
    dim = config.hidden_size
    embeds = torch.randn((batch_size, seq_len, dim), device='cuda', dtype=torch.float16)
    labels = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    # reference
    embeds_ref = embeds.clone().requires_grad_()
    outputs = model(inputs_embeds=embeds_ref, labels=labels)
    logits_ref = outputs.logits
    loss_ref = outputs.loss
    loss_ref.backward()

    # flash checkpoint
    enable_swap(model)
    embeds.requires_grad_()
    outputs = model(inputs_embeds=embeds, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    loss.backward()

    # compare
    print(f"The maximum difference of output is {torch.max(torch.abs(logits - logits_ref))}")
    print(f"The maximum difference of gradient is {torch.max(torch.abs(embeds.grad - embeds_ref.grad))}")


def benchmark(model, swap=True, profile=False):
    if swap:
        enable_swap(model, limit=None, gelu_recompute=True)
    else:
        enable_swap(model, limit=0, gelu_recompute=False)

    # inputs
    batch_size, seq_len = 4, 4096
    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    def benchmark_step():
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()
        torch.cuda.synchronize()
    
    import timeit

    # warmup
    timeit.timeit(benchmark_step, number=5)
    # benchmark
    t = timeit.timeit(benchmark_step, number=10)
    print(f"Batch size={batch_size}, Seq len={seq_len}, Iteration time={t / 10}.")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    if profile:
        prof = torch.profiler.profile()
        with prof:
            benchmark_step()
        prof.export_chrome_trace("swap.json") # open it at chrome://tracing


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoConfig
    
    model_id = "lmsys/vicuna-7b-v1.3"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 8
    config.use_cache = False

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model.to(device='cuda')

    #test(model)

    benchmark(model, swap=True, profile=False)

