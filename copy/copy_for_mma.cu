#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace C5 {

template <class ProblemShape,
          class CtaTiler,
          class StrideA,
          class SmemLayoutA,
          class CopyA,
          class CopyOut>
__global__ void kernelCuteSwizzleMMACopy(ProblemShape problem_shape, CtaTiler cta_tiler,
                                       bf16 *In, StrideA a_stride, SmemLayoutA a_smem_layout, 
                                       CopyA copyA, CopyOut copyOut,
                                       bf16 *Out, int M, int K) {
    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});

    extern __shared__ bf16 smem[];
    Tensor sA = make_tensor(make_smem_ptr<bf16>(smem), a_smem_layout);

    Tensor gIn = make_tensor(make_gmem_ptr(In), make_shape(M, K), a_stride);
    Tensor gOut = make_tensor(make_gmem_ptr(Out), make_shape(M, K), a_stride);

    // Use the plain CTA tiler for global memory; swizzled layout only applies to shared memory.
    auto gIn_tile = local_tile(gIn, cta_tiler, make_coord(blockIdx.x, blockIdx.y));
    auto gOut_tile = local_tile(gOut, cta_tiler, make_coord(blockIdx.x, blockIdx.y));

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

void runKernelCuteSwizzleMMACopy(int M, int K, bf16 *In, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BK = 64;

    auto prob_shape = make_shape(M, K);
    auto dA = make_stride(K, Int<1>{}); 

    auto cta_tiler = make_shape(Int<BM>{}, Int<BK>{});

    // Swizzled shared memory layout
    auto swizzled_128B_atom = composition(
                    Swizzle<3,3,3>{},
                    make_layout(
                        make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                        make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))
                    );
    auto sA = tile_to_shape(
      swizzled_128B_atom,
      make_shape(Int<BM>{},Int<BK>{})
    );

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_8, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (64 threads)
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}   // Each thread copies 8 bf16 (16 bytes)
    );

    // Regular copy for shared → global (Async not available)
    TiledCopy copyOut = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
        Layout<Shape<_8, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (128 threads)
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}   // Each thread copies 8 bf16 (16 bytes)
    );
    
    int smem_elems = int(size(sA));
    size_t smem_bytes = size_t(smem_elems) * sizeof(bf16);
    int grid_x = (M + BM - 1) / BM;
    int grid_y = (K + BK - 1) / BK;
    int block_threads = max(int(size(copyA)), int(size(copyOut)));
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_threads);

    kernelCuteSwizzleMMACopy<<<dimGrid, dimBlock, smem_bytes>>>(
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

} // namespace C5

using C5::runKernelCuteSwizzleMMACopy;