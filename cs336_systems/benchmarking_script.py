import torch
from cs336_basics.model import BasicsTransformerLM
import argparse

parser = argparse.ArgumentParser()

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

t = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)

data = torch.randint(0, args.vocab_size, (1024,), device=args.device)
print(data)
