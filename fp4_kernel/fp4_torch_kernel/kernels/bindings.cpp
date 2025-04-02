#include <torch/extension.h>

// Forward declarations for functions defined in the CUDA files.
torch::Tensor fp4_quantize(torch::Tensor input);
torch::Tensor fp4_dequantize(torch::Tensor input);
torch::Tensor fp4_matmul(torch::Tensor A, torch::Tensor B);
torch::Tensor fp4_matmul_backward_A(torch::Tensor grad_output, torch::Tensor B);
torch::Tensor fp4_matmul_backward_B(torch::Tensor A, torch::Tensor grad_output);
torch::Tensor fp4_linear(torch::Tensor X, torch::Tensor W, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_quantize", &fp4_quantize, "Quantize tensor to FP4 using __nv_fp4_e2m1");
  m.def("fp4_dequantize", &fp4_dequantize, "Dequantize tensor from FP4");
  m.def("fp4_matmul", &fp4_matmul, "Optimized FP4 matmul with asynchronous copies, double buffering, and loop unrolling");
  m.def("fp4_matmul_backward_A", &fp4_matmul_backward_A, "Backward kernel for FP4 matmul (grad_A)");
  m.def("fp4_matmul_backward_B", &fp4_matmul_backward_B, "Backward kernel for FP4 matmul (grad_B)");
  m.def("fp4_linear", &fp4_linear, "Fused FP4 linear layer (FP4 matmul + bias)");
}
