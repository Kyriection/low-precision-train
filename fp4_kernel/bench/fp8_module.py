from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
import time

# Build and load the extension.
fp8_ext = load(name="fp8_ext", sources=["fp8_ext_kernel.cu"], verbose=True)

# A custom autograd function for FP8 quantization/dequantization.
class FP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Quantize from float (FP32) to FP8 (stored as uint8)
        output = fp8_ext.fp8_quantize(input)
        output = output.view_as(input)  # Reshape to original dimensions.
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = fp8_ext.fp8_dequantize(grad_output)
        grad_input = grad_input.view_as(input)
        return grad_input

# A custom autograd function for FP8 matrix multiplication.
class FP8MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # A and B are expected to be in FP8 (uint8)
        output = fp8_ext.fp8_matmul(A, B)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is not implemented in this demo.
        raise NotImplementedError("Backward pass not implemented for FP8 matmul.")

# A module that uses FP8 for both weights and matrix multiplication.
class FP8Module(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP8Module, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Quantize both the input and the weight.
        x_fp8 = FP8Function.apply(x)
        weight_fp8 = FP8Function.apply(self.weight)
        # Transpose the weight (since weight shape is [out, in]) and perform FP8 matmul.
        output_fp8 = FP8MatMulFunction.apply(x_fp8, weight_fp8.t())
        return output_fp8

if __name__ == "__main__":
    # Create sample input.
    x = torch.randn(10, 20, device="cuda")
    model = FP8Module(20, 30).cuda()

    # Compute the FP8 matmul output.
    st = time.time()
    output_fp8 = model(x)
    print(time.time() - st)
    print("FP8 (raw uint8) output:")
    # print(output_fp8)

    # To print as float, dequantize the FP8 output.
    output_float = fp8_ext.fp8_dequantize(output_fp8).view_as(output_fp8)
    print("Dequantized float output:")
    # print(output_float)
