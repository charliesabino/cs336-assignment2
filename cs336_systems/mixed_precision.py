import torch
from torch import nn


def accumulation_experiment():
    """
    Result: Accumulating in fp16 reduces accuracy drastically (2 OOMs over fp32 -> fp32)
    """
    expected_result = 10.0
    print(f"--- Starting Floating-Point Precision Demo ---")
    print(f"Expected result for all tests: {expected_result}\n")

    print("--- Test 1: Accumulator (float32) += Value (float32) ---")
    s1 = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s1 += torch.tensor(0.01, dtype=torch.float32)
    print(f"Result: {s1.item():.10f}")
    print(f"Data type of sum: {s1.dtype}")
    print(f"Difference from expected: {
          abs(expected_result - s1.item()):.10f}\n")

    print("--- Test 2: Accumulator (float16) += Value (float16) ---")
    s2 = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s2 += torch.tensor(0.01, dtype=torch.float16)
    print(f"Result: {s2.item():.10f}")
    print(f"Data type of sum: {s2.dtype}")
    print(f"Difference from expected: {
          abs(expected_result - s2.item()):.10f}\n")

    print(
        "--- Test 3: Accumulator (float32) += Value (float16) [Implicit Upcasting] ---")
    s3 = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s3 += torch.tensor(0.01, dtype=torch.float16)
    print(f"Result: {s3.item():.10f}")
    print(f"Data type of sum: {s3.dtype}")
    print(f"Difference from expected: {
          abs(expected_result - s3.item()):.10f}\n")

    print(
        "--- Test 4: Accumulator (float32) += Value (float16).type(float32) [Explicit Upcasting] ---")
    s4 = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s4 += x.type(torch.float32)
    print(f"Result: {s4.item():.10f}")
    print(f"Data type of sum: {s4.dtype}")
    print(f"Difference from expected: {
          abs(expected_result - s4.item()):.10f}\n")


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
        print(f"FC1 weight dtype: {self.fc1.weight.dtype}")
        print(f"LayerNorm weight dtype: {self.ln.weight.dtype}")
        print(f"LayerNorm bias dtype: {self.ln.bias.dtype}")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"FC1 output dtype: {x.dtype}")
        x = self.ln(x)
        print(f"LayerNorm output dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"Logits dtype: {x.dtype}")
        return x


def model_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_fn = nn.CrossEntropyLoss()
        model = ToyModel(10, 10).to(device)
        data = torch.randn((10, 10)).to(device)
        logits = model(data)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = loss_fn(logits, data)
        print(f"Loss dtype: {loss.dtype}")
        loss.backward()
        
        print("\n--- Gradient Datatypes ---")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.dtype}")
            else:
                print(f"{name}: No gradient")
        
        optim.step()


if __name__ == "__main__":
    model_experiment()

