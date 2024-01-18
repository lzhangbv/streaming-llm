# Introduction
This folder contains several Triton kernel implementations for large language models inference, such as RMS Norm and RoPE operators, attention operators ([flash-attention](https://arxiv.org/abs/2205.14135), decode-attention, [paged-attention](https://arxiv.org/abs/2309.06180)), fused mlp (fused gate-up projection and silu), and w4a16 matmul ([GPTQ](https://arxiv.org/abs/2210.17323) style). 

These implementations are based on OpenAI [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) (e.g., vector addition, fused softmax, layer norm, and matmul operators), [LightLLM](https://github.com/ModelTC/lightllm/tree/main), [Kernl](https://github.com/ELS-RD/kernl), and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ). 

Please note that our implementations are not well optimized, and it is suggested to enable autotuning to find better configurations of meta-parameters (e.g., `BLOCK_SIZE`) and compilation options (e.g., `num_warps`). 

# Run a Simple Triton Program with Debugger Support

To run through this code with the Triton Debugger, please use the following command:

~~~
TRITON_INTERPRET=1 python vector_add.py    
~~~

# Run a Simple Triton Program with NSight Compute Profiler

First, download the Nvidia Nsight Compute System to your local system here: https://developer.nvidia.com/nsight-compute

If you are running this tool in a remote machine, you must make sure that tool is installed there as well. This can be achieved through conda:

~~~
conda install -c nvidia nsight-compute
~~~

Once both systems are setup, to profile the kernel:

~~~
ncu --target-processes all 
--set detailed 
--import-source yes 
--section SchedulerStats 
--section WarpStateStats 
--section SpeedOfLight_RooflineChart 
--section SpeedOfLight_HierarchicalTensorRooflineChart 
-o output_file_location 
python vector_add.py
~~~

Download the trace file that ncu generates to your local machine to see full detailed analysis.

An example to use ncu is given here: https://pytorch.org/blog/accelerating-triton