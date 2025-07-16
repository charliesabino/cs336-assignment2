import numpy as np
import torch
import time
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import argparse

parser = argparse.ArgumentParser()

batch_size = 4

parser.add_argument("--vocab_size", type=int, default=10_000)
parser.add_argument("--context_length", type=int, default=1024)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--d_ff", type=int, default=3072)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--rope_theta", type=int, default=10_000)
parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
).to(args.device)

data = np.random.randint(0, args.vocab_size, (1 << 12,))
x, y = get_batch(data, batch_size, args.context_length, device=args.device)

warmup_steps = 5
benchmark_steps = 10

for _ in range(warmup_steps):
    model(x)

optimizer = AdamW(model.parameters())

for _ in range(benchmark_steps):
    s = time.time()
    logits = model(x)
    if args.device == "cuda":
        torch.cuda.synchronize()
    e = time.time()
    print(f"Forward pass time: {e - s}")

    optimizer.zero_grad()
    loss = cross_entropy(logits, y)

    s = time.time()
    loss.backward()
    if args.device == "cuda":
        torch.cuda.synchronize()
    e = time.time()
    print(f"Backward pass time: {e - s}")
