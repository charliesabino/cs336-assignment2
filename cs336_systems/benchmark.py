import argparse
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
import time
import torch
import numpy as np
from annotated import annotated_scaled_dot_product_attention
from cs336_basics import model as cs336_model

cs336_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


parser = argparse.ArgumentParser()

parser.add_argument("--vocab_size", type=int, default=10_000)
parser.add_argument("--context_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--rope_theta", type=int, default=10_000)
parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--model_config", type=str, default="small")

args = parser.parse_args()


class ModelConfig:
    def __init__(self, vocab_size, context_length, d_model, d_ff, num_layers, num_heads, batch_size, rope_theta):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.rope_theta = rope_theta


model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    **model_configs[args.model_config],
    rope_theta=args.rope_theta,
).to(args.device)

model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

data = np.random.randint(0, args.vocab_size, (1 << 12,))
x, y = get_batch(data, args.batch_size,
                 args.context_length, device=args.device)

warmup_steps = 5
benchmark_steps = 10

for _ in range(warmup_steps):
    model(x)

optimizer = AdamW(model.parameters())
forward_times = []
backward_times = []

for _ in range(benchmark_steps):
    s = time.time_ns()
    torch.cuda.nvtx.range_push("forward")
    logits = model(x)
    if args.device == "cuda":
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    e = time.time_ns()

    forward_times.append(e - s)

    loss = cross_entropy(logits, y)

    s = time.time_ns()
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if args.device == "cuda":
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    e = time.time_ns()

    backward_times.append(e - s)

    optimizer.zero_grad()
    optimizer.step()

print(f"Forward pass mean time: {np.mean(forward_times) * 1e-6} ms")
print(f"Forward pass std dev: {np.std(forward_times) * 1e-6} ms")

print(f"Backward pass mean time: {np.mean(backward_times) * 1e-6} ms")
print(f"Backward pass std dev: {np.std(backward_times) * 1e-6} ms")
