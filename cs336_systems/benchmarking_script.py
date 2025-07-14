from cs336_basics.model import BasicsTransformerLM
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_size", default=32000)
parser.add_argument("--context_length", default=1024)
parser.add_argument("--d_model", default=768)
parser.add_argument("--num_layers", default=12)

args = parser.parse_args()
t = BasicsTransformerLM()
