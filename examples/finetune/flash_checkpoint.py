"""
Activation checkpointing without recomputing flash-attention. 
"""

import math
import types
from typing import List, Optional, Tuple, Union

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


ACTIVATION_OFFLOAD = False

class FlashCheckpointFunction(torch.autograd.Function):
    """Two functions f1 and f2 are built, i.e., 
    1) qkv = f1(hidden_states, position_ids)
    2) attn_out, softmax_lse = flash_attention(qkv)
    3) output = f2(attn_out, hidden_states)

    To be simple, make sure no rng states are involved, such as dropout. 
    """

    @staticmethod
    def forward(ctx, run_function1, run_function2, hidden_states, position_ids, causal, softmax_scale): 
        with torch.no_grad():
            q, k, v = run_function1(hidden_states, position_ids)
            softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
            attn_out, _, _, _, _, softmax_lse, _, _ = _flash_attn_forward(
                q, 
                k, 
                v, 
                dropout_p=0,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
            )
            output = run_function2(attn_out, hidden_states)

        ctx.run_function1 = run_function1
        ctx.run_function2 = run_function2
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal

        if ACTIVATION_OFFLOAD:
            ctx.saved_device = hidden_states.device
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            saved_attn_out = attn_out.to("cpu", non_blocking=True)
        else:
            saved_hidden_states = hidden_states
            saved_attn_out = attn_out
        
        ctx.save_for_backward(saved_hidden_states, position_ids, saved_attn_out, softmax_lse)
        
        return output
    
    @staticmethod
    def backward(ctx, dout):
        hidden_states, position_ids, attn_out, softmax_lse = ctx.saved_tensors

        if ACTIVATION_OFFLOAD:
            hidden_states = hidden_states.to(ctx.saved_device, non_blocking=True)
            attn_out = attn_out.to(ctx.saved_device, non_blocking=True)

        # detach hidden states
        hidden_states = hidden_states.detach().requires_grad_()
        attn_out.requires_grad = True

        # function2 fwbw
        with torch.enable_grad():
            output = ctx.run_function2(attn_out, hidden_states)
        torch.autograd.backward(output, dout)

        # function1 fw
        with torch.enable_grad():
            q, k, v = ctx.run_function1(hidden_states, position_ids)

        # attention bw
        dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        _flash_attn_backward(
            attn_out.grad,
            q, 
            k, 
            v, 
            attn_out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p=0,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            rng_state=None,
        )

        # function1 bw
        torch.autograd.backward((q, k, v), (dq, dk, dv))
        return None, None, hidden_states.grad, None, None, None


def flash_checkpoint(run_function1, run_function2, hidden_states, position_ids, causal=True, softmax_scale=None):
    return FlashCheckpointFunction.apply(run_function1, run_function2, hidden_states, position_ids, causal, softmax_scale)


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
    Run functions are based on modeling_llama from huggingface transformers (v4.38.1). 
    """
    def run_function1(hidden_states, position_ids):
        hidden_states = self.input_layernorm(hidden_states)

        bsz, q_len, _ = hidden_states.size()
        query_states = self.self_attn.q_proj(hidden_states)
        key_states = self.self_attn.k_proj(hidden_states)
        value_states = self.self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)

        cos, sin = self.self_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)
    
    def run_function2(attn_output, hidden_states):
        attn_output = attn_output.reshape(hidden_states.shape)
        attn_output = self.self_attn.o_proj(attn_output)
        hidden_states = attn_output + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    hidden_states = flash_checkpoint(run_function1, run_function2, hidden_states, position_ids)
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (None,)
    
    if use_cache:
        outputs += (None,)
    
    return outputs


def make_checkpoint(model):
    for layer in model.model.layers:
        layer.forward = types.MethodType(decoder_layer_forward, layer)


def test(model):
    # freeze base model
    for name, p in model.named_parameters():
         p.requires_grad = False
    
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
    make_checkpoint(model)
    embeds.requires_grad_()
    outputs = model(inputs_embeds=embeds, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    loss.backward()

    # compare
    print(f"The maximum difference of output is {torch.max(torch.abs(logits - logits_ref))}")
    print(f"The maximum difference of gradient is {torch.max(torch.abs(embeds.grad - embeds_ref.grad))}")


def memory_snapshot(model, checkpoint=True):
    # inputs
    batch_size, seq_len = 4, 2048
    x = torch.randint(low=0, high=1024, size=(batch_size, seq_len), device='cuda')

    if checkpoint:
        make_checkpoint(model)
    
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    outputs = model(input_ids=x, labels=x)
    loss = outputs.loss
    del outputs
    loss.backward()

    # torch.cuda.memory._dump_snapshot("snapshot.pickle") # open it at https://pytorch.org/memory_viz
    # torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def benchmark(model, checkpoint='flash'):
    if checkpoint == 'flash':
        model.gradient_checkpointing_disable()
        make_checkpoint(model)
    elif checkpoint == 'full':
        model.gradient_checkpointing_enable()
    else:
        assert checkpoint == 'none', 'checkpoint type is from [flash, full, none]'

    # inputs
    batch_size, seq_len = 1, 4096
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
    print(f"Batch size={batch_size}, Seq len={seq_len}, Checkpoint type={checkpoint}, Iteration time={t / 10}.")


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoConfig
    
    model_id = "lmsys/vicuna-7b-v1.3"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    config.use_cache = False

    model = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    model.to(device='cuda')

    # test
    test(model)

    # memory
    # memory_snapshot(model, checkpoint=True)

    # benchmark
    # benchmark(model, checkpoint='none')
    # benchmark(model, checkpoint='full')
    # benchmark(model, checkpoint='flash')
