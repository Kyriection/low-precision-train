from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
import time

# Build and load the extension.
fp4_ext = load(name="fp4_ext", sources=["fp4_ext_kernel.cu"], verbose=True)

# A custom autograd function for FP4 quantization/dequantization.
class FP4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Quantize from float (FP32) to FP4 (stored as uint8)
        output = fp4_ext.fp4_quantize(input)
        output = output.view_as(input)  # Reshape to original dimensions.
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = fp4_ext.fp4_dequantize(grad_output)
        grad_input = grad_input.view_as(input)
        return grad_input

# A custom autograd function for FP4 matrix multiplication.
class FP4MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # A and B are expected to be in FP4 (uint8)
        output = fp4_ext.fp4_matmul(A, B)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is not implemented in this demo.
        raise NotImplementedError("Backward pass not implemented for FP4 matmul.")

# A module that uses FP4 for both weights and matrix multiplication.
class FP4Module(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP4Module, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Quantize both the input and the weight.
        x_fp4 = FP4Function.apply(x)
        weight_fp4 = FP4Function.apply(self.weight)
        # Transpose the weight (since weight shape is [out, in]) and perform FP4 matmul.
        output_fp4 = FP4MatMulFunction.apply(x_fp4, weight_fp4.t())
        return output_fp4

if __name__ == "__main__":
    # Create sample input.
    x = torch.randn(10, 20, device="cuda")
    model = FP4Module(20, 30).cuda()

    # Compute the FP4 matmul output.
    st = time.time()
    output_fp4 = model(x)
    print(time.time() - st)
    print("FP4 (raw uint8) output:")
    # print(output_fp4, output_fp4.dtype)

    # To print as float, dequantize the FP4 output.
    output_float = fp4_ext.fp4_dequantize(output_fp4).view_as(output_fp4)
    print("Dequantized float output:")
    # print(output_float, output_float.dtype)