import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FP32Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP32Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device="cpu"))
        self.bias = nn.Parameter(torch.zeros(out_features, device="cpu"))

    def forward(self, x):
        return F.linear(x, self.weight.T, self.bias)

class SimpleFP32Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleFP32Model, self).__init__()
        self.fc1 = FP32Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = FP32Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    device = "cpu"

    input_dim = 10
    output_dim = 5
    num_samples = 200

    model = SimpleFP32Model(input_dim, 20, output_dim).to(device)
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



