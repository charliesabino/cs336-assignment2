import argparse
import logging
import time
import torch
import numpy as np
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from annotated import annotated_scaled_dot_product_attention
from cs336_basics import model as cs336_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cs336_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

parser = argparse.ArgumentParser(description="Benchmark Transformer models with cleaner logging.")
parser.add_argument("--vocab_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--rope_theta", type=int, default=10_000)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    # "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
warmup_steps = 2
benchmark_steps = 10
context_lengths = [128, 256, 512, 1024]
results = []

logging.info(f"Starting benchmark on device: {args.device}")
data = np.random.randint(0, args.vocab_size, (1 << 12,))

for model_name, config_params in model_configs.items():
    logging.info(f"--- Starting benchmark for model: '{model_name}' ---")
    
    max_context_length = max(context_lengths)
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=max_context_length,
        rope_theta=args.rope_theta,
        **config_params,
    ).to(args.device)
    
    optimizer = AdamW(model.parameters())

    for context_length in context_lengths:
        logging.info(f"Benchmarking with context length: {context_length}")
        
        forward_times = []
        backward_times = []

        x, y = get_batch(data, args.batch_size, context_length, device=args.device)

        torch.cuda.nvtx.range_push("warmup")
        for _ in range(warmup_steps):
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.zero_grad()
        torch.cuda.nvtx.range_pop()
        
        for i in range(benchmark_steps):
            torch.cuda.synchronize()
            s_fwd = time.time_ns()
            torch.cuda.nvtx.range_push(f"Fwd-{model_name}-{context_length}-{i}")
            logits = model(x)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            e_fwd = time.time_ns()
            forward_times.append(e_fwd - s_fwd)

            loss = cross_entropy(logits, y)

            torch.cuda.synchronize()
            s_bwd = time.time_ns()
            torch.cuda.nvtx.range_push(f"Bwd-{model_name}-{context_length}-{i}")
            loss.backward()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            e_bwd = time.time_ns()
            backward_times.append(e_bwd - s_bwd)

            optimizer.zero_grad()
            optimizer.step()

        results.append({
            "model": model_name,
            "context_length": context_length,
            "fwd_mean_ms": np.mean(forward_times) * 1e-6,
            "fwd_std_ms": np.std(forward_times) * 1e-6,
            "bwd_mean_ms": np.mean(backward_times) * 1e-6,
            "bwd_std_ms": np.std(backward_times) * 1e-6,
        })

logging.info("Benchmark finished. Displaying summary table.")

print("\n" + "="*80)
print(f"{'Benchmark Summary':^80}")
print("="*80)
print(f"| {'Model':<10} | {'Context Length':<15} | {'Fwd Pass (ms)':<15} | {'Bwd Pass (ms)':<15} |")
print(f"|{'-'*12}|{'-'*17}|{'-'*17}|{'-'*17}|")

for res in results:
    fwd_str = f"{res['fwd_mean_ms']:.2f} ± {res['fwd_std_ms']:.2f}"
    bwd_str = f"{res['bwd_mean_ms']:.2f} ± {res['bwd_std_ms']:.2f}"
    print(f"| {res['model']:<10} | {res['context_length']:<15} | {fwd_str:<15} | {bwd_str:<15} |")

print("="*80)
