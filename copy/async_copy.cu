#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace C2 {

template <class ProblemShape,
          class CtaTiler,
          class StrideA,
          class SmemLayoutA,
          class CopyA,
          class CopyOut>
__global__ void kernelCuteAsyncCopy(ProblemShape problem_shape, CtaTiler cta_tiler,
                                       bf16 *In, StrideA a_stride, SmemLayoutA a_smem_layout, 
                                       CopyA copyA, CopyOut copyOut,
                                       bf16 *Out, int M, int K) {
    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});

    extern __shared__ bf16 smem[];
    Tensor sA = make_tensor(make_smem_ptr<bf16>(smem), a_smem_layout);

    Tensor gIn = make_tensor(make_gmem_ptr(In), make_shape(M, K), a_stride);
    Tensor gOut = make_tensor(make_gmem_ptr(Out), make_shape(M, K), a_stride);

    auto tile_shape = shape(a_smem_layout);
    auto gIn_tile = local_tile(gIn, tile_shape, make_coord(blockIdx.x, blockIdx.y));
    auto gOut_tile = local_tile(gOut, tile_shape, make_coord(blockIdx.x, blockIdx.y));

    // Partition the tensors for this thread using the TiledCopy
    auto thr_copy_a = copyA.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gIn_tile);
    auto tAsA = thr_copy_a.partition_D(sA);

    // Async copy from global to shared
    copy(copyA, tAgA, tAsA);

    // Compute overlap
    // Partition the tensors for the write back
    auto thr_copy_out = copyOut.get_thread_slice(threadIdx.x);
    auto tAsOut = thr_copy_out.partition_S(sA);
    auto tAgOut = thr_copy_out.partition_D(gOut_tile);
    
    
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Regular copy back
    copy(copyOut, tAsOut, tAgOut);
}


void runKernelCuteAsyncCopy(int M, int K, bf16 *In, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BK = 64;

    auto prob_shape = make_shape(M, K);
    auto dA = make_stride(K, Int<1>{}); 

    auto cta_tiler = make_shape(Int<BM>{}, Int<BK>{});

    // Shared memory layout - row major  
    auto sA = make_layout(make_shape(Int<BM>{}, Int<BK>{}), 
                          make_stride(Int<BK>{}, Int<1>{}));

    // Simple async copy that works: 32 threads, each thread copies 64 bf16
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, bf16>{},
        Layout<Shape<_4, _8>, Stride<_8, _1>>{},     // 32 threads
        Layout<Shape<_1,_8>, Stride<_8, _1>>{}     // Each copies 8 contiguous elements
    );
    
    // Regular copy for shared → global (Async not available)
    TiledCopy copyOut = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
        Layout<Shape<_4, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1,_8>, Stride<_8, _1>>{}
    );
    
    int smem_elems = int(size(sA));
    size_t smem_bytes = size_t(smem_elems) * sizeof(bf16);
    int grid_x = (M + BM - 1) / BM;
    int grid_y = (K + BK - 1) / BK;
    int block_threads = int(size(copyA));
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_threads);
    
    kernelCuteAsyncCopy<<<dimGrid, dimBlock, smem_bytes>>>(
        prob_shape, cta_tiler,
        In, dA, sA, copyA, copyOut,
        Out, M, K
    );
    
    cudaError_t _err = cudaGetLastError();
    if (_err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(_err));
    }
    _err = cudaDeviceSynchronize();
    if (_err != cudaSuccess) {
        printf("Kernel runtime error: %s\n", cudaGetErrorString(_err));
    } 
}

} // namespace C0

using C2::runKernelCuteAsyncCopy;