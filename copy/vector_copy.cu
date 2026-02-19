#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace C1 {

template <class ProblemShape,
          class CtaTiler,
          class StrideA,
          class SmemLayoutA,
          class CopyA>
__global__ void kernelCuteVectorCopy(ProblemShape problem_shape, CtaTiler cta_tiler,
                                       bf16 *In, StrideA a_stride, SmemLayoutA a_smem_layout, CopyA copyA,
                                       bf16 *Out, int M, int K) {
    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});

    int smem_a_elems = int(size(a_smem_layout));
    extern __shared__ bf16 smem[];
    Tensor sA = make_tensor(make_smem_ptr<bf16>(smem), a_smem_layout);

    // Copy the first tile of A into shared memory for demonstration
    // Create a tensor view of global memory A
    Tensor gIn = make_tensor(make_gmem_ptr(In), make_shape(M, K), a_stride);
    Tensor gOut = make_tensor(make_gmem_ptr(Out), make_shape(M, K), a_stride);

    // Define the tile to copy (use the shared-memory layout's shape)
    auto tile_shape = shape(a_smem_layout);
    auto gIn_tile = local_tile(gIn, tile_shape, make_coord(blockIdx.x, blockIdx.y));
    auto gOut_tile = local_tile(gOut, tile_shape, make_coord(blockIdx.x, blockIdx.y));

    // Partition the tensors for this thread using the TiledCopy
    auto thr_copy_a = copyA.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gIn_tile);
    auto tAsA = thr_copy_a.partition_D(sA);

    // Partition the tensors for the write back
    // auto thr_copy_out = copyOut.get_thread_slice(threadIdx.x);
    auto tAsOut = thr_copy_a.partition_S(sA);
    auto tAgOut = thr_copy_a.partition_D(gOut_tile);

    // Copy global memory A → shared memory sA using CuTe copy
    copy(copyA, tAgA, tAsA);

    __syncthreads();

    copy(copyA, tAsOut, tAgOut); // Copy back to global memory for verification
    __syncthreads();
}


void runKernelCuteVectorCopy(int M, int K, bf16 *In, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BK = 64;

    // Problem shape and strides
    auto prob_shape = make_shape(M, K);
    auto dA = make_stride(K, Int<1>{}); 

    // Create tiling sizes
    auto cta_tiler = make_shape(Int<BM>{}, Int<BK>{});

    // Create shared memory layouts
    auto sA = make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}));

    // Thread copy layouts (simple)
    TiledCopy copyA = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
                        Layout<Shape<_4,_8>, Stride<_8,_1>>{},   // thread layout: 32 threads
                        Layout<Shape<_1,_8>, Stride<_8,_1>>{});     // per-thread value layout: 8 contiguous elements

    // I don't optimise the layout of the copy here.
    // Kernel Launch Details
    int smem_elems = int(size(sA));
    size_t smem_bytes = size_t(smem_elems) * sizeof(bf16);
    int grid_x = (M + BM - 1) / BM;
    int grid_y = (K + BK - 1) / BK;
    int block_threads = int(size(copyA));
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_threads);
    
    kernelCuteVectorCopy<<<dimGrid, dimBlock, smem_bytes>>>(
        prob_shape,
        cta_tiler,
        In, dA, sA, copyA,
        Out, M, K
      );
    // Check for launch errors
    cudaError_t _err = cudaGetLastError();
    if (_err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(_err));
    }
    // Optionally synchronize here to catch runtime errors early
    _err = cudaDeviceSynchronize();
    if (_err != cudaSuccess) {
        printf("Kernel runtime error (synchronize): %s\n", cudaGetErrorString(_err));
    } 
}

} // namespace C1

using C1::runKernelCuteVectorCopy;