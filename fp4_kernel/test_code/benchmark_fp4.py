import torch
import torch.nn as nn
import torch.optim as optim
from fp4_torch_kernel.utils import FP4LinearFunction

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP4Linear, self).__init__()
        # Create full precision master copies
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))

    def forward(self, x):
        # Use the custom autograd function
        # Note: Adjust the order of parameters if needed.
        return FP4LinearFunction.apply(x, self.weight, self.bias)

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

    input_dim = 10
    output_dim = 5
    num_samples = 200

    model = SimpleFP4Model(input_dim, 20, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # load weight & data 
    data = torch.load('demo/data.pth')
    X, y = data['x'].to(device), data['y'].to(device)
    model.load_state_dict(torch.load('demo/fp32_model.pth', map_location=device))

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == '__main__':
    train()



