#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
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
#include <unistd.h>

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

#include "matmuls/cublaslt_matmul.cu"
#if defined(SM_ARCH) && (SM_ARCH >= 80)
#include "matmuls/cute_mma_matmul.cu"
#include "matmuls/cute_mma_matmul_2stage-pip.cu"
#include "matmuls/cute_mma_matmul_2stage-pip_grid_swizzle.cu"
#include "matmuls/cute_mma_matmul_3stage-pip_grid_swizzle.cu"
#endif
auto MATMUL_NUMS = std::vector<int>{
  0
#if defined(SM_ARCH) && (SM_ARCH >= 80)
  ,1,2,3,4
#endif
};

std::default_random_engine generator(42);
cublasHandle_t cublas_handle;
// void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
//   float alpha = 1, beta = 0;
//   // C(column major) = A(row major) * B(column major)
//   cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
//     N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

//   if (status != CUBLAS_STATUS_SUCCESS) {
//     std::cout << "CUBLAS error: " << status << std::endl;
//     exit(1);
//   }
// }

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

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr) {
  clear_l2();
  switch (kernel_num) {
    case 0:
      runCublasMatmulBF16(M, N, K, A, B, C, DB);
      break;
#if defined(SM_ARCH) && (SM_ARCH >= 80)
    case 1:
      runKernelCuteSwizzleMMAMatmul(M, N, K, A, B, C);
      break;
    case 2:
      runkernelCutePipeline2Stage(M, N, K, A, B, C);
      break;
    case 3:
      runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul2StageImproved(M, N, K, A, B, C);
      break;
    case 4:
      runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul3StageImproved(M, N, K, A, B, C);
      break;
#endif
  }
}
int yo = 0;
void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
  ++yo;
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N, int max_size = 4096) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    int r = i / max_size, c = i % max_size;
    int it = c*max_size+r;
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

int main() {
  cublasCreate(&cublas_handle);
  // initialize cuBLASLt (creates handle and workspace) used by runCublasMatmulBF16
  initCublasLt();
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long max_size = 2048;
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr;  // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices
  
  int *DB = nullptr; int *dDB = nullptr;  

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);
  
  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));
  
  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size,
  cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size,
      cudaMemcpyHostToDevice));

  int repeat_times = 8;
  bool run_verif = true;

  printf("Warmup: Running cuBLAS...\n");
  memset(C, 0, sizeof(bf16) * max_size * max_size);
  cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  memset(DB, ~0, sizeof(int) * max_size * 128);
  cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
    cudaMemcpyHostToDevice));
    
  for (int i = 0; i < 1000; i++) {
    run_kernel(0, m, n, k, dA, dB, dC_ref);
  }

  printf("Warmup Done\n");


  for (int kernel_num : MATMUL_NUMS) {
    // for (int kernel_num : {0, 11}) {
    // Give the GPU some rest to avoid thermal throttling
    sleep(5);
    std::cout << "KERNEL " << kernel_num << std::endl;
    // Verify against cuBLAS. Also serves as a warmup step.
    if (run_verif) {
      memset(C, 0, sizeof(bf16) * max_size * max_size);
      cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      memset(DB, ~0, sizeof(int) * max_size * 128);
      cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
        cudaMemcpyHostToDevice));
      run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
      run_kernel(kernel_num, m, n, k, dA, dB, dC, dDB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

      if (kernel_num > 1 && !verify_matrix(C_ref, C, m * n, max_size)) {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~" << std::endl;
        printf("%f\n", __bfloat162float(C_ref[m]));
      }

      cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n", (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute, (sumStore + .0) / cntStore, times);
      }

    }

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long flops = (2LL * m) * (n * k);
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
        elapsed_time / 1000.0 / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  return 0;
};