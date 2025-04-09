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

def train_and_time():
    device = "cuda"
    model = SimpleFP4Model(1000, 20000, 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    for _ in range(10):
        x_w = torch.randn(4, 1000, device=device)
        y_w = torch.randn(4, 5, device=device)
        _ = model(x_w)
        loss = criterion(model(x_w), y_w)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Set up CUDA events for timing.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record start event.
    start_event.record()

    # Run one forward and backward pass.
    x = torch.randn(4, 1000, device=device)
    y = torch.randn(4, 5, device=device)
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Record end event.
    end_event.record()

    # Wait for completion.
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"Forward+Backward pass elapsed time: {elapsed_ms:.3f} ms")

def main():
    for epoch in range(10):
        train_and_time()

if __name__ == '__main__':
    main()
