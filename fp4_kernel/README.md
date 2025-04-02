# FP4_torch_kernel
This is a repository for implementing FP4 computation in PyTorch.
It implements the kernel operations, but not the kernel storing of FP4 data. It should lead to a speedup compared to FP8

## Install instructions
```python setup.py develop```

## Examples
Also found in ```./examples/train.py```:
```
import torch
import torch.nn as nn
import torch.optim as optim
from fp4_torch_kernel.layers import FP4Linear

class SimpleFP4Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleFP4Model, self).__init__()
        self.fc1 = FP4Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = FP4Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    device = "cuda"
    model = SimpleFP4Model(10, 20, 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        optimizer.zero_grad()
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == '__main__':
    train()
```

## Important
### Tune the matmul kernel
For your hardware and model size please tune the TILE_SIZE UNROLL_FACTOR in matmul.cu. Increasing these values together give you a speed benefit, but this hugely relies on your type of hardware and model size.

## TODO
- Implementation of further kernels for different types of computations
