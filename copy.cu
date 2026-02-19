#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>

// include available copy implementations
#include "copy/basic_copy.cu"
#include "copy/vector_copy.cu"
#include "copy/async_copy.cu"
#include "copy/swizzled_128B_copy.cu"
#include "copy/copy_for_mma.cu"

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// ********
// From https://github.com/SzymonOzog/CUDA_Matmul.git
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void clear_l2() 
{
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2; // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));

    // Synchronize to ensure memset is done before proceeding
    gpuErrchk(cudaDeviceSynchronize());
}
// ********

std::default_random_engine generator(69);

void randomize_matrix(bf16 *mat, long N) {
  std::normal_distribution<float> distribution(0, 1);
  for (long i = 0; i < N; i++) {
    mat[i] = __float2bfloat16(distribution(generator));
  }
}

bool verify_matrix(bf16 *ref, bf16 *out, long N) {
  for (long i = 0; i < N; ++i) {
    float a = __bfloat162float(ref[i]);
    float b = __bfloat162float(out[i]);
    if (fabs(a - b) > 1e-2f) {
      printf("Mismatch at %ld: ref=%f out=%f\n", i, a, b);
      return false;
    }
  }
  return true;
}

void run_kernel(int kernel_num, int M, int K, bf16 *dIn, bf16 *dOut) {
  clear_l2();
  switch (kernel_num) {
    case 0: {
      runKernelCuteBasicCopy(M, K, dIn, dOut);
      break;
    }
    case 1: {
      runKernelCuteVectorCopy(M, K, dIn, dOut);
      break;
    }
    case 2: {
      runKernelCuteAsyncCopy(M, K, dIn, dOut);
      break;
    }
    case 3: {
      runKernelCuteSwizzle128BCopy(M, K, dIn, dOut);
      break;
    }
    case 4: {
      runKernelCuteSwizzleMMACopy(M, K, dIn, dOut);
      break;
    }
    default:
      break;
  }
}

__global__ void warmupKernel() { __shared__ int s[1]; s[0] = s[0] + 1; }

int main() {
  for (int i = 0; i < 1000; ++i){
    clear_l2();
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int SIZE = 8192;
  const long total = (long)SIZE * SIZE;

  bf16 *hA = (bf16 *)malloc(sizeof(bf16) * total);
  bf16 *hOut = (bf16 *)malloc(sizeof(bf16) * total);

  randomize_matrix(hA, total);

  bf16 *dIn = nullptr, *dOut = nullptr;
  cudaCheck(cudaMalloc((void **)&dIn, sizeof(bf16) * total));
  cudaCheck(cudaMalloc((void **)&dOut, sizeof(bf16) * total));
  cudaCheck(cudaMemcpy(dIn, hA, sizeof(bf16) * total, cudaMemcpyHostToDevice));

  std::vector<int> kernels = {0,1,2,3,4};
  int repeat = 8;

  for (int k : kernels) {
    // warmup
    cudaCheck(cudaMemset(dOut, 0, sizeof(bf16) * total));
    run_kernel(k, SIZE, SIZE, dIn, dOut);
    cudaCheck(cudaDeviceSynchronize());

    // verification
    cudaCheck(cudaMemcpy(hOut, dOut, sizeof(bf16) * total, cudaMemcpyDeviceToHost));
    bool ok = verify_matrix(hA, hOut, total);
    std::cout << "Kernel " << k << (ok ? " PASSED" : " FAILED") << std::endl;

    // benchmark
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
      run_kernel(k, SIZE, SIZE, dIn, dOut);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    double avg_ms = elapsed_time / repeat;
    double bytes = double(sizeof(bf16)) * double(total) * 2.0; // read + write
    double bw_gb_s = (bytes / 1e9) / (avg_ms / 1000.0);
    printf("Kernel %d: avg time = %7.3f ms, bandwidth = %7.3f GB/s\n", k, avg_ms, bw_gb_s);
  }

  free(hA);
  free(hOut);
  cudaFree(dIn);
  cudaFree(dOut);
  return 0;
}