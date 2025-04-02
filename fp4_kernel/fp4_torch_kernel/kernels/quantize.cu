/*
Kernel ops to convert the fp4 operations from and to uint8 float
*/

#include <cuda_fp4.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

// Convert float to fp4 (__nv_fp4_e2m1) and store as a uint8.
__global__ void quantize_kernel(const float* input, uint8_t* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Convert float to FP4 using the __nv_fp4_e2m1 constructor.
    __nv_fp4_e2m1 fp4_val = __nv_fp4_e2m1(input[idx]);
    output[idx] = *reinterpret_cast<uint8_t*>(&fp4_val);
  }
}

// Convert from fp4 (stored as uint8) back to float.
__global__ void dequantize_kernel(const uint8_t* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    __nv_fp4_e2m1 fp4_val = *reinterpret_cast<const __nv_fp4_e2m1*>(&input[idx]);
    output[idx] = static_cast<float>(fp4_val);
  }
}

// Wrapper to quantize a tensor.
torch::Tensor fp4_quantize(torch::Tensor input) {
  auto n = input.numel();
  auto output = torch::empty({n}, torch::dtype(torch::kUInt8).device(input.device()));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  quantize_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<uint8_t>(), n);
  return output;
}

// Wrapper to dequantize a tensor.
torch::Tensor fp4_dequantize(torch::Tensor input) {
  auto n = input.numel();
  auto output = torch::empty({n}, torch::dtype(torch::kFloat32).device(input.device()));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  dequantize_kernel<<<blocks, threads>>>(input.data_ptr<uint8_t>(), output.data_ptr<float>(), n);
  return output;
}