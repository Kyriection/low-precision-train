# import os
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
from torch.utils.cpp_extension import load
import torch
import os
import torch.nn.functional as F

#TODO: For the matmul impl check if in amp mode, bc in amp we only want the fw to be fp4 formatted.

this_dir = os.path.dirname(os.path.abspath(__file__))

fp4_ext = load(
    name="fp4_ext",
    sources=[
        f"{this_dir}/kernels/ops/matmul.cu",
        f"{this_dir}/kernels/quantize.cu",
        f"{this_dir}/kernels/layers/linear.cu",
        f"{this_dir}/kernels/bindings.cpp"
    ],
    verbose=False
)

class FP4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Quantize and then dequantize; reshape back to original.
        quantized = fp4_ext.fp4_quantize(input)
        dequantized = fp4_ext.fp4_dequantize(quantized).view_as(input)
        ctx.save_for_backward(input)
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output

class FP4MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        out_fp4 = fp4_ext.fp4_matmul(A, B)
        return out_fp4

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        # Compute gradient with respect to A and B using custom backward kernels.
        grad_A = fp4_ext.fp4_matmul_backward_A(grad_output, B)
        grad_B = fp4_ext.fp4_matmul_backward_B(A, grad_output)
        return grad_A, grad_B
    
class FP4DequantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        dequantized = fp4_ext.fp4_dequantize(input)
        return dequantized.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# class FP4LinearFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, X, W, bias):
#         ctx.save_for_backward(X, W, bias)
#         Y = fp4_ext.fp4_linear(X, W, bias)
#         return Y

#     @staticmethod
#     def backward(ctx, grad_output):
#         X, W, bias = ctx.saved_tensors
#         # Compute gradients:
#         # For Y = X * W, grad_X = grad_output * Wᵀ, grad_W = Xᵀ * grad_output.

#         grad_X = fp4_ext.fp4_matmul_backward_A(grad_output, W)
#         grad_W = fp4_ext.fp4_matmul_backward_B(X, grad_output)

#         # grad_X = grad_output @ W.t()
#         # grad_W = X.t() @ grad_output

#         grad_bias = grad_output.sum(dim=0)
#         return grad_X, grad_W, grad_bias
    


class FP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, bias):
        return fp4_ext.fp4_linear(X, W, bias)