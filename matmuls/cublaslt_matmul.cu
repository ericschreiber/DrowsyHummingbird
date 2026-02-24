// Copied from https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/cublaslt_matmul.cu

#include <cublasLt.h>

// Random code snippets to use while development
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasLtHandle_t cublaslt_handle;

void initCublasLt() {
    cublasLtCreate(&cublaslt_handle);
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));    
}

void runCublasMatmulBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr) {
      // check alignment (some modes work unaligned but it always best to be aligned for performance)
      if(((uintptr_t)A % 16) != 0 || ((uintptr_t)B % 16) != 0 || ((uintptr_t)C % 16) != 0) {
          printf("All cuBLASLt pointers must be aligned!\n");
          exit(EXIT_FAILURE);
      }
  
      // create the operation descriptor
      cublasLtMatmulDesc_t operationDesc;
      cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  
      int returnedResults = 0;
      cublasLtMatmulPreference_t preference;
      cublasLtMatmulHeuristicResult_t heuristic;
  
      // A is row-major (MxK), B is row-major (KxN), C is row-major (MxN)
      // Need to transpose A to treat row-major as column-major
      cublasOperation_t opTranspose = CUBLAS_OP_T;
      cublasOperation_t opNoTranspose = CUBLAS_OP_N;
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose));
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose));
  
      // define matrix layouts
      cublasLtMatrixLayout_t ALayout;
      cublasLtMatrixLayout_t BLayout;
      cublasLtMatrixLayout_t CLayout;
      // A: row-major MxK means column-major view is KxM with leading dim K
      cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_16BF, K, M, K);
      // B: row-major KxN means column-major view is NxK with leading dim N  
      cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_16BF, N, K, N);
      
      // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
      // C: row-major MxN, but with column-major view it needs special handling
      // For row-major output, we need leading dimension N
      cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_16BF, M, N, N);
  
      // create a preference handle with specified max workspace
      cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size));
  
      // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
      cublasDataType_t scale_type = CUDA_R_32F;
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
  
      // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
      cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, CLayout,
                                     preference, 1, &heuristic, &returnedResults);
      if (returnedResults == 0) {
          printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d", N, M, K);
          exit(EXIT_FAILURE);
      }
  
      // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
      float alpha = 1, beta = 0;
  
      // call the matmul
    cublasLtMatmul(cublaslt_handle, operationDesc,
               &alpha, A, ALayout, B, BLayout, &beta, C, CLayout, C, CLayout,
               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0);
  
      // cleanups
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatmulDescDestroy(operationDesc);
      cublasLtMatrixLayoutDestroy(ALayout);
      cublasLtMatrixLayoutDestroy(BLayout);
      cublasLtMatrixLayoutDestroy(CLayout);
      cudaCheck(cudaGetLastError());
  }