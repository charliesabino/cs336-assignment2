benchmarking_script:

b) Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

Forward pass mean time: 20.956710100000002 ms
Forward pass std dev: 2.265681259441603 ms
Backward pass mean time: 34.1995732 ms
Backward pass std dev: 34.42309040537658 ms

c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

No warmup:

Forward pass mean time: 58.3085317 ms
Forward pass std dev: 114.87121496467516 ms
Backward pass mean time: 33.1610003 ms
Backward pass std dev: 32.42718067524797 ms

We don't warm our caches.

1 warmup:

Forward pass mean time: 20.3682256 ms
Forward pass std dev: 1.8161996007968506 ms
Backward pass mean time: 32.3503146 ms
Backward pass std dev: 29.964810661511788 ms

Our caches are warmed!

nsys_profile:

================================================================================

                               Benchmark Summary

================================================================================

| Model      | Context Length  | Fwd Pass (ms)   | Bwd Pass (ms)   |
|------------|-----------------|-----------------|-----------------|
| small      | 128             | 33.29 ± 0.71    | 46.72 ± 0.15    |
| small      | 256             | 33.48 ± 0.06    | 46.75 ± 0.14    |
| small      | 512             | 33.98 ± 0.18    | 46.91 ± 0.59    |
| small      | 1024            | 45.92 ± 0.07    | 91.94 ± 0.06    |
| medium     | 128             | 66.56 ± 0.69    | 90.77 ± 1.65    |
| medium     | 256             | 65.87 ± 0.19    | 90.53 ± 1.25    |
| medium     | 512             | 67.23 ± 1.11    | 116.87 ± 0.62   |
| medium     | 1024            | 127.85 ± 0.08   | 258.95 ± 0.07   |
| large      | 128             | 100.19 ± 2.55   | 138.09 ± 4.66   |
| large      | 256             | 101.18 ± 2.08   | 140.42 ± 1.37   |
| large      | 512             | 129.17 ± 0.11   | 267.12 ± 0.84   |
| large      | 1024            | 280.65 ± 0.11   | 571.43 ± 0.12   |
================================================================================
