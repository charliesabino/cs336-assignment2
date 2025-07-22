"""
Result: Accumulating in fp16 reduces accuracy drastically (2 OOMs over fp32 -> fp32)
"""

import torch

expected_result = 10.0
print(f"--- Starting Floating-Point Precision Demo ---")
print(f"Expected result for all tests: {expected_result}\n")

print("--- Test 1: Accumulator (float32) += Value (float32) ---")
s1 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s1 += torch.tensor(0.01, dtype=torch.float32)
print(f"Result: {s1.item():.10f}")
print(f"Data type of sum: {s1.dtype}")
print(f"Difference from expected: {abs(expected_result - s1.item()):.10f}\n")

print("--- Test 2: Accumulator (float16) += Value (float16) ---")
s2 = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s2 += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s2.item():.10f}")
print(f"Data type of sum: {s2.dtype}")
print(f"Difference from expected: {abs(expected_result - s2.item()):.10f}\n")

print(
    "--- Test 3: Accumulator (float32) += Value (float16) [Implicit Upcasting] ---")
s3 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s3 += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s3.item():.10f}")
print(f"Data type of sum: {s3.dtype}")
print(f"Difference from expected: {abs(expected_result - s3.item()):.10f}\n")

print(
    "--- Test 4: Accumulator (float32) += Value (float16).type(float32) [Explicit Upcasting] ---")
s4 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s4 += x.type(torch.float32)
print(f"Result: {s4.item():.10f}")
print(f"Data type of sum: {s4.dtype}")
print(f"Difference from expected: {abs(expected_result - s4.item()):.10f}\n")
