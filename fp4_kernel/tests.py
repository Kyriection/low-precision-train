import unittest
import torch
import torch.nn as nn

from fp4_torch_kernel.layers import FP4Linear  # adjust based on your file structure
from fp4_torch_kernel.utils import fp4_ext  # assuming your extension module is imported as fp4_ext

# TODO: Impl other kernels that are being used: 
# (like loss fn etc. This is extra work yes but it will be hopefully worth it)

class FP4LinearTest(unittest.TestCase):
    def test_forward_shape(self):
        # Test that the FP4Linear layer produces an output of the expected shape.
        layer = FP4Linear(in_features=10, out_features=5).cuda()
        input_tensor = torch.randn(4, 10, device="cuda")
        output = layer(input_tensor)
        self.assertEqual(output.shape, (4, 5))
        self.assertEqual(output.dtype, torch.float32)

    def test_forward_dequantize(self):
        # Test that the output after dequantization has the expected shape.
        layer = FP4Linear(in_features=8, out_features=3).cuda()
        input_tensor = torch.randn(2, 8, device="cuda")
        output = layer(input_tensor)
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(output.dtype, torch.float32)

    def test_backward(self):
        print("TESTBW\n\n\n")
        # Test that the backward pass runs without error and computes gradients.
        layer = FP4Linear(in_features=8, out_features=3).cuda()
        opt = torch.optim.Adam(params=layer.parameters(), lr=0.001)
        opt.zero_grad()
        input_tensor = torch.randn(2, 8, device="cuda", requires_grad=True)
        output = layer(input_tensor)
        loss = output.sum()
        print("loss", loss)
        loss.backward()
        # Check that gradients for the input are computed.
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        # Check that gradients for the layer's weight are computed.
        self.assertIsNotNone(layer.weight_fp32.grad)
        self.assertEqual(layer.weight_fp32.grad.shape, layer.weight_fp32.shape)

if __name__ == "__main__":
    unittest.main()
