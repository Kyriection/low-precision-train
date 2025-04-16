import torch
import torch.nn as nn
from fp4_torch_kernel.utils import FP4LinearFunction

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP4Linear, self).__init__()
        # Create full precision master copies.
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))
    
    def forward(self, x):
        return FP4LinearFunction.apply(x, self.weight, self.bias)