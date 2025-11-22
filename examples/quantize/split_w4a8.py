# Residual activation quantization:  Split-W8A16 (Double-W8A8) and Split-W4A8 (Double-W4A4)

import torch
from triton.testing import do_bench

NBITs = 4

def int8_quant(x, nbits=NBITs, dim=-1):
  maxq = 2 ** (nbits - 1) - 1
  x = x.float()
  x_absmax = x.abs().max(dim=dim, keepdim=True)[0]
  scale = x_absmax / maxq
  x_int8 = (x / scale).round().clip(min=-maxq, max=maxq)
  return x_int8.to(torch.int8), scale

def double_int8_quant(x, nbits=NBITs, dim=-1):
  maxq = 2 ** (nbits - 1) - 1
  x = x.float()
  x_absmax = x.abs().max(dim=dim, keepdim=True)[0]
  scale1 = x_absmax / maxq
  x1_scaled = x / scale1
  x1_int8 = x1_scaled.round().clip(min=-maxq, max=maxq)

  # residual quantization
  delta_x = x1_scaled - x1_int8
  delta_x_absmax = delta_x.abs().max(dim=dim, keepdim=True)[0]
  scale2 = delta_x_absmax / maxq
  x2_scaled = delta_x / scale2
  x2_int8 = x2_scaled.round().clip(min=-maxq, max=maxq)
  scale2 *= scale1
  
  return x1_int8.to(torch.int8), scale1, x2_int8.to(torch.int8), scale2

def ref_w16a16(weight, input):
  return torch.matmul(input, weight.T)

@torch.compile()
def dequant_w8a16(weight_int8, weight_scale, input):
  weight = (weight_int8 * weight_scale).to(input)
  return torch.matmul(input, weight.T)

def fake_double_w8a8(weight_int8, weight_scale, input):
  x1_int8, scale1, x2_int8, scale2 = double_int8_quant(input, dim=-1)
  x1 = x1_int8 * scale1
  x2 = x2_int8 * scale2
  x = (x1 + x2).to(input)
  # print(f"====Double-A{NBITs} MSE:", torch.nn.functional.mse_loss(input, x))
  weight = (weight_int8 * weight_scale).to(input)
  return torch.matmul(x, weight.T)

@torch.compile()
def double_w8a8(weight_int8, weight_scale, input):
  x1_int8, scale1, x2_int8, scale2 = double_int8_quant(input, dim=-1)

  x_int8 = torch.cat([x1_int8, x2_int8], dim=0)
  out = torch._int_mm(x_int8, weight_int8.T)
  out1, out2 = torch.split(out, input.shape[0], dim=0)
  
  out = out1 * scale1 * weight_scale.T + out2 * scale2 * weight_scale.T
  return out.to(input)

def fake_single_w8a8(weight_int8, weight_scale, input):
  x1_int8, scale1 = int8_quant(input, dim=-1)
  x = (x1_int8 * scale1).to(input)
  # print(f"====A{NBITs}A16 MSE:", torch.nn.functional.mse_loss(input, x))
  weight = (weight_int8 * weight_scale).to(input)
  return torch.matmul(x, weight.T)

@torch.compile()
def single_w8a8(weight_int8, weight_scale, input):
  x1_int8, scale1 = int8_quant(input, dim=-1)
  out = torch._int_mm(x1_int8, weight_int8.T)
  out = out * scale1 * weight_scale.T
  return out.to(input)

for batch_size in [32, 64, 128, 256, 512, 1024, 2048]:
  # parameters
  print(f"\n=====batch size: {batch_size}=====")
  dim_in, dim_out = 4096, 8192
  dtype = torch.float16

  input = torch.randn((batch_size, dim_in), dtype=dtype, device='cuda')
  weight = torch.randn((dim_out, dim_in), dtype=dtype, device='cuda') / 10

  # offline quantization
  weight_int8, weight_scale = int8_quant(weight, dim=-1)
  input_int8, input_scale = int8_quant(input, dim=-1)

  # MSE accuracy
  out_ref = ref_w16a16(weight, input)
  out_dequant = dequant_w8a16(weight_int8, weight_scale, input)

  out_fake_double_w8a8 = fake_double_w8a8(weight_int8, weight_scale, input)
  out_double_w8a8 = double_w8a8(weight_int8, weight_scale, input)

  out_fake_single_w8a8 = fake_single_w8a8(weight_int8, weight_scale, input)
  out_single_w8a8 = single_w8a8(weight_int8, weight_scale, input)

  print(f"W{NBITs}A16 MSE:", torch.nn.functional.mse_loss(out_ref, out_dequant))
  print(f"Double-W{NBITs}A{NBITs} MSE:", torch.nn.functional.mse_loss(out_ref, out_double_w8a8))
  # print(f"Fake Double-W{NBITs}A{NBITs} MSE:", torch.nn.functional.mse_loss(out_ref, out_fake_double_w8a8))
  print(f"Single-W{NBITs}A{NBITs} MSE:", torch.nn.functional.mse_loss(out_ref, out_single_w8a8))
  # print(f"Fake Single-W{NBITs}A{NBITs} MSE:", torch.nn.functional.mse_loss(out_ref, out_fake_single_w8a8))

  # Benchmark: note we do not use int4_gemm implementation
  time_ref = do_bench(lambda: ref_w16a16(weight, input)) * 1e3
  time_dequant = do_bench(lambda: dequant_w8a16(weight_int8, weight_scale, input)) * 1e3
  time_double_w8a8 = do_bench(lambda: double_w8a8(weight_int8, weight_scale, input)) * 1e3
  time_single_w8a8 = do_bench(lambda: single_w8a8(weight_int8, weight_scale, input)) * 1e3

  # do not include quantization and dequantization costs (in us)
  time_pure_w8a8 = do_bench(lambda: torch._int_mm(input_int8, weight_int8.T)) * 1e3

  print(f"\nRef Time={time_ref}")
  print(f"Dequant-W8A16 Time={time_dequant}")
  print(f"Double-W8A8 Time={time_double_w8a8}")
  print(f"Single-W8A8 Time={time_single_w8a8}")
  print(f"Pure-W8A8 Time={time_pure_w8a8}")
