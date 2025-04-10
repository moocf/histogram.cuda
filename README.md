Histogram represents a count of the frequency of each element in a data set.

```c
Each thread computes pairwise product of multiple components of vector.
Since there are 10 components, but only a maximum of 4 total threads,
each thread pairwise product of its component, and shifts by a stride
of the total number of threads. This is done as long as it does not
exceed the length of the vector. Each thread maintains the sum of the
pairwise products it calculates.

kernel():
1. Get byte at buffer for this thread.
2. Atomically increment appropriate index in histogram.
3. Shift to the next byte, by a stride.
```

```
Each thread atomically increments the bytes in buffer meant for it.
This is done in the shared thread block memory first, until the
buffer is consumed. Then each thread in the block updates the
histogram in the global memory atomically. This reduces global
memory contention.

kernelShared():
1. Initialize shared memory (of size 256).
2. Get byte at buffer for this thread.
3. Atomically increment appropriate index in shared memory.
4. Shift to the next byte, by a stride.
5. Wait for all threads within the block to finish.
5. Reduce the sum in the cache to a single value in binary tree fashion.
6. Atomically update per-block histogram into global histogram.
```

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# CPU Histogram ...
# CPU execution time: 0.0 ms
# CPU Histogram sum: 1000000
#
# GPU Histogram: atomic ...
# GPU execution time: 1.1 ms
# GPU Histogram sum: 1000000
# GPU Histogram verified.
#
# GPU Histogram: shared + atomic ...
# GPU execution time: 0.5 ms
# GPU Histogram sum: 1000000
# GPU Histogram verified.
```

See [main.cu] for code.

[main.cu]: main.cu

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)

![](https://ga-beacon.deno.dev/G-G1E8HNDZYY:v51jklKGTLmC3LAZ4rJbIQ/github.com/moocf/histogram.cuda)
