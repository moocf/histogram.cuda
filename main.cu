#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "support.h"


const int threads = 256;

void memset_rand(uint8 *buf, int N) {
  for (int i=0; i<N; i++)
    buf[i] = rand() & 0xFF;
}

void histogram(int *hist, uint8 *buff, int N) {
  for (int i=0; i<N; i++) // 1
    hist[buff[i]]++;      // 1
}

int histogram_sum(int *hist, int N) {
  int sum = 0;
  for (int i=0; i<N; i++)
    sum += hist[i];
  return sum;
}


// Each thread atomically increments the bytes in buffer meant for it.
// This however leads to high contention to the 256 locations in the
// global memory.
// 
// 1. Get byte at buffer for this thread.
// 2. Atomically increment appropriate index in histogram.
// 3. Shift to the next byte, by a stride.
__global__ void kernel(int *hist, uint8 *buff, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < N) {
    atomicAdd(&hist[buff[i]], 1); // 1, 2
    i += blockDim.x * gridDim.x;  // 3
  }
}


// Each thread atomically increments the bytes in buffer meant for it.
// This is done in the shared thread block memory first, until the
// buffer is consumed. Then each thread in the block updates the
// histogram in the global memory atomically. This reduces global
// memory contention.
// 
// 1. Initialize shared memory (of size 256).
// 2. Get byte at buffer for this thread.
// 3. Atomically increment appropriate index in shared memory.
// 4. Shift to the next byte, by a stride.
// 5. Wait for all threads within the block to finish.
// 5. Reduce the sum in the cache to a single value in binary tree fashion.
// 6. Atomically update per-block histogram into global histogram.
__global__ void kernel_shared(int *hist, uint8 *buff, int N) {
  __shared__ int temp[threads];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int t = threadIdx.x;
  
  temp[t] = 0;     // 1
  __syncthreads(); // 1

  while (i < N) {
    atomicAdd(&temp[buff[i]], 1); // 2, 3
    i += blockDim.x * gridDim.x;  // 4
  }
  __syncthreads(); // 5

  atomicAdd(&hist[t], temp[t]); // 6
}


int run_cpu(uint8* buff, int N) {
  int H = 256;
  int H1 = H * sizeof(int);

  int *hist = (int*) malloc(H1);
  memset(hist, 0, H1);

  clock_t begin = clock();
  histogram(hist, buff, N);
  clock_t end = clock();
  
  double duration = (double) (end - begin) / CLOCKS_PER_SEC;
  printf("CPU execution time: %3.1f ms\n", duration * 1000);
  printf("CPU Histogram sum: %d\n", histogram_sum(hist, H));

  free(hist);
  return 0;
}


int run_gpu(uint8 *buff, int N, int shared) {
  int H = 256;
  int N1 = N * sizeof(uint8);
  int H1 = H * sizeof(int);

  uint8 *buffD;
  int *hist, *histD;
  hist = (int*) malloc(H1);

  cudaEvent_t start, stop;
  TRY(cudaEventCreate(&start));
  TRY(cudaEventCreate(&stop));
  TRY(cudaEventRecord(start, 0));

  TRY(cudaMalloc(&buffD, N1));
  TRY(cudaMemcpy(buffD, buff, N1, cudaMemcpyHostToDevice));
  TRY(cudaMalloc(&histD, H1));
  TRY(cudaMemset(histD, 0, H));

  cudaDeviceProp p;
  TRY(cudaGetDeviceProperties(&p, 0));
  int blocks = 2 * p.multiProcessorCount;
  if (!shared) kernel<<<blocks, threads>>>(histD, buffD, N);
  else kernel_shared<<<blocks, threads>>>(histD, buffD, N);

  float duration;
  TRY( cudaMemcpy(hist, histD, H1, cudaMemcpyDeviceToHost) );
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );
  printf("GPU execution time: %3.1f ms\n", duration);
  printf("GPU Histogram sum: %d\n", histogram_sum(hist, H));

  int* histH = (int*) malloc(H1);
  memset(histH, 0, H1);
  histogram(histH, buff, N);
  int cmp = memcmp(hist, histH, H1);
  if (cmp == 0) printf("GPU Histogram verified.\n");
  else printf("GPU Histogram is wrong!\n");

  TRY(cudaEventDestroy(start));
  TRY(cudaEventDestroy(stop));
  TRY(cudaFree(histD));
  TRY(cudaFree(buffD));
  free(histH);
  free(hist);
  return 0;
}


int main() {
  int N = 1000000;
  int N1 = N * sizeof(uint8);

  uint8 *buff = (uint8*) malloc(N1);
  memset_rand(buff, N1);

  printf("CPU Histogram ...\n");
  run_cpu(buff, N);
  printf("\n");

  printf("GPU Histogram: atomic ...\n");
  run_gpu(buff, N, 0);
  printf("\n");

  printf("GPU Histogram: shared + atomic ...\n");
  run_gpu(buff, N, 1);
  printf("\n");

  free(buff);
  return 0;
}
