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
