
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from fp4_torch_kernel.layers import FP4Linear

class SimpleFP32Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleFP32Model, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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


def forward_time_bench(model_precision):
    device = "cuda"

    bs = 128
    input_dim = 8192
    hidden_dim = 28672

    if model_precision == "fp32":
        model = SimpleFP32Model(input_dim, hidden_dim, input_dim).to(device)
    elif model_precision == "fp4":
        model = SimpleFP4Model(input_dim, hidden_dim, input_dim).to(device)
    else:
        raise ValueError("model_precision must be either 'fp32' or 'fp4'")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    print('**** Benchmarking ****')
    print('Warming up...')
    for _ in range(20):
        x_w = torch.randn(bs, input_dim, device=device)
        _ = model(x_w)
    print('Warm up done!')

    # Run one forward and backward pass.
    total_time = 0
    steps = 1000
    for _ in range(steps):
        x = torch.randn(bs, input_dim, device=device)
        start = time.time()
        output = model(x)
        end = time.time()
        total_time += (end - start)
    avg_time = total_time / steps
    print(f"Average forward pass time for {model_precision}: {1000 * avg_time:.4f} ms")

def train_and_time():
    device = "cuda"

    bs = 128
    input_dim = 8192
    hidden_dim = 28672

    model = SimpleFP4Model(input_dim, hidden_dim, input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    for _ in range(10):
        x_w = torch.randn(128, 8192, device=device)
        y_w = torch.randn(128, 8192, device=device)
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
    x = torch.randn(bs, input_dim, device=device)
    y = torch.randn(bs, input_dim, device=device)

    start = time.time()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    end = time.time()
    print(f"**** Forward+Backward pass elapsed time: {end - start:.3f} seconds")

    # Record end event.
    end_event.record()

    # Wait for completion.
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"Forward+Backward pass elapsed time: {elapsed_ms:.3f} ms")

def main():
    forward_time_bench("fp32")
    forward_time_bench("fp4")

if __name__ == '__main__':
    main()
