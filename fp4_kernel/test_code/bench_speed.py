
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

    bs = 512
    input_dim = 8192
    hidden_dim = 28672

    if model_precision == "fp32":
        model = SimpleFP32Model(input_dim, hidden_dim, input_dim).to(device)
    elif model_precision == "bf16":
        model = SimpleFP32Model(input_dim, hidden_dim, input_dim).to(device, dtype=torch.bfloat16)
    elif model_precision == "fp4":
        model = SimpleFP4Model(input_dim, hidden_dim, input_dim).to(device)
    else:
        raise ValueError("model_precision must be either 'fp32' or 'fp4'")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    print('**** Benchmarking forward****')
    print('Warming up...')
    for _ in range(20):
        if model_precision == 'bf16':
            x_w = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
        else:
            x_w = torch.randn(bs, input_dim, device=device)
        _ = model(x_w)
    print('Warm up done!')

    # Run one forward and backward pass.
    total_time = 0
    steps = 100
    for _ in range(steps):
        if model_precision == 'bf16':
            x = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
        else:
            x = torch.randn(bs, input_dim, device=device)
        start = time.time()
        _ = model(x)
        end = time.time()
        total_time += (end - start)
    avg_time = total_time / steps
    print(f"Average forward pass time for {model_precision}: {1000 * avg_time:.4f} ms")


def train_time_bench(model_precision):
    device = "cuda"

    bs = 512
    input_dim = 8192
    hidden_dim = 28672

    if model_precision == "fp32":
        model = SimpleFP32Model(input_dim, hidden_dim, input_dim).to(device)
    elif model_precision == "bf16":
        model = SimpleFP32Model(input_dim, hidden_dim, input_dim).to(device, dtype=torch.bfloat16)
    elif model_precision == "fp4":
        model = SimpleFP4Model(input_dim, hidden_dim, input_dim).to(device)
    else:
        raise ValueError("model_precision must be either 'fp32' or 'fp4'")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    print('**** Benchmarking forward+backward****')
    print('Warming up...')
    for _ in range(20):
        if model_precision == 'bf16':
            x_w = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
            y_w = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
        else:
            x_w = torch.randn(bs, input_dim, device=device)
            y_w = torch.randn(bs, input_dim, device=device)
        pred_w = model(x_w)
        loss = criterion(pred_w, y_w)
        loss.backward()
    print('Warm up done!')

    # Run one forward and backward pass.
    total_time = 0
    steps = 100
    for _ in range(steps):
        if model_precision == 'bf16':
            x = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
            y = torch.randn(bs, input_dim, device=device, dtype=torch.bfloat16)
        else:
            x = torch.randn(bs, input_dim, device=device)
            y = torch.randn(bs, input_dim, device=device)
        start = time.time()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        end = time.time()
        total_time += (end - start)
    avg_time = total_time / steps
    print(f"Average forward+backward pass time for {model_precision}: {1000 * avg_time:.4f} ms")


def forward_single_layer(model_precision, bs, input_dim, hidden_dim):
    device = "cuda"

    if model_precision == "fp32":
        model = nn.Linear(input_dim, hidden_dim).to(device)
    elif model_precision == "fp4":
        model = FP4Linear(input_dim, hidden_dim).to(device)
    else:
        raise ValueError("model_precision must be either 'fp32' or 'fp4'")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Warm up (run a few iterations to warm up CUDA)
    for _ in range(20):
        x_w = torch.randn(bs, input_dim, device=device)
        _ = model(x_w)
    # print('**** Benchmarking forward for Single Layer Warm up done!****')

    # Run one forward and backward pass.
    total_time = 0
    steps = 100
    for _ in range(steps):
        x = torch.randn(bs, input_dim, device=device)
        start = time.time()
        _ = model(x)
        end = time.time()
        total_time += (end - start)
    avg_time = total_time / steps
    print(f"Average forward pass time for {model_precision}: {1000 * avg_time:.4f} ms")


def main():
    forward_time_bench("fp32")
    forward_time_bench("bf16")
    forward_time_bench("fp4")

    train_time_bench("fp32")
    train_time_bench("bf16")
    train_time_bench("fp4")

    # for dim in range(8192, 8192*16+1, 8192):
    #     print(f"dim: {dim}")
    #     forward_single_layer("fp32", 128, 8192, dim)
    #     forward_single_layer("fp4", 128, 8192, dim)

if __name__ == '__main__':
    main()
