import torch
import torch.nn as nn
import torch.optim as optim
from fp4_torch_kernel.utils import FP8LinearFunction

class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP8Linear, self).__init__()
        # Create full precision master copies
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))
    
    def forward(self, x):
        # Use the custom autograd function
        # Note: Adjust the order of parameters if needed.
        return FP8LinearFunction.apply(x, self.weight, self.bias)

class SimpleFP4Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleFP4Model, self).__init__()
        self.fc1 = FP8Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = FP8Linear(hidden_features, out_features)
    
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



