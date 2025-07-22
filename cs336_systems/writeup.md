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
a) What is the total time spent on your forward pass? Does it match what we had measured before
with the Python standard library?

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

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many
times is this kernel invoked during a single forward pass of your model? Is it the same kernel
that takes the most runtime when you do both forward and backward passes? (Hint: look at the
“CUDA GPU Kernel Summary” under “Stats Systems View”, and filter using NVTX ranges to
identify which parts of the model are responsible for which kernels.)

sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
takes longest time in the forward pass (46.8%). It is invoked 145 times per pass.

(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that
several other kernels still take a non-trivial amount of the overall runtime. What other kernels
besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward
pass?

void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
also takes a non trivial amount of time (4.1%).

(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within
the self-attention layer of your model during a forward pass. How does the difference in runtimes
compare to the difference in FLOPs?

Despite requiring far fewer FLOPs, softmax takes roughly the same amount of time
on average as computing the attention scores.

benchmarking_mixed_precision:

(a)
FC1 weight dtype: torch.float32
LayerNorm weight dtype: torch.float32
LayerNorm bias dtype: torch.float32
FC1 output dtype: torch.bfloat16
LayerNorm output dtype: torch.float32
Logits dtype: torch.bfloat16
Loss dtype: torch.float32

--- Gradient Datatypes ---
fc1.weight: torch.float32
ln.weight: torch.float32
ln.bias: torch.float32
fc2.weight: torch.float32

(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer
differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed
precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently?
Why or why not?

Maybe the variance might underflow and we end up with div by 0?
BF16 might fix this because it has higher dynamic range?

(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on any trends as model size changes.

================================================================================
                               Benchmark Summary
================================================================================

| Model      | Context Length  | Fwd Pass (ms)   | Bwd Pass (ms)   |
|------------|-----------------|-----------------|-----------------|
| small      | 128             | 22.11 ± 0.09    | 24.94 ± 0.16    |
| small      | 256             | 22.12 ± 0.12    | 25.51 ± 0.13    |
| small      | 512             | 23.25 ± 0.11    | 25.84 ± 0.13    |
| small      | 1024            | 24.28 ± 0.22    | 46.08 ± 0.02    |
| medium     | 128             | 45.25 ± 0.39    | 49.41 ± 0.40    |
| medium     | 256             | 45.19 ± 0.19    | 50.24 ± 0.10    |
| medium     | 512             | 45.74 ± 0.14    | 50.33 ± 0.07    |
| medium     | 1024            | 59.72 ± 0.02    | 117.27 ± 0.09   |
| large      | 128             | 68.26 ± 1.07    | 74.59 ± 0.40    |
| large      | 256             | 68.47 ± 0.22    | 75.80 ± 0.51    |
| large      | 512             | 69.77 ± 0.12    | 94.88 ± 0.16    |
| large      | 1024            | 111.83 ± 0.05   | 221.46 ± 0.19   |
================================================================================

Gains appear more drastic the greater the model size.
