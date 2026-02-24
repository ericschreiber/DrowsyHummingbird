#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace M0 {

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape,
          class CtaTiler,
          class StrideA, class SmemLayoutA,
          class StrideB, class SmemLayoutB,
          class StrideC, class SmemLayoutC,
          class CopyA, class CopyB, class CopyOut, 
          class CopyS2R_A, class CopyS2R_B, class TiledMMA>
__global__ void kernelCuteSwizzleMMAMatmul(ProblemShape problem_shape, CtaTiler cta_tiler,
                                        bf16 *A, StrideA a_stride, SmemLayoutA a_smem_layout,
                                        bf16 *B, StrideB b_stride, SmemLayoutB b_smem_layout,
                                        bf16 *Out, StrideC out_stride, SmemLayoutC c_smem_layout,
                                        CopyA copyA, CopyB copyB, CopyOut copyOut,
                                        CopyS2R_A copyS2R_A, CopyS2R_B copyS2R_B, TiledMMA mma,
                                        int M, int N, int K) {
    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size(copyA) == size(mma));

    CUTE_STATIC_ASSERT_V(size<0>(a_smem_layout) == size<0>(cta_tiler));  // BM
    CUTE_STATIC_ASSERT_V(size<0>(b_smem_layout) == size<1>(cta_tiler));  // BN
    CUTE_STATIC_ASSERT_V(size<1>(a_smem_layout) == size<2>(cta_tiler));  // BK
    CUTE_STATIC_ASSERT_V(size<1>(b_smem_layout) == size<2>(cta_tiler));  // BK

    // extern __shared__ bf16 smem[];
    // Tensor sA = make_tensor(make_smem_ptr<bf16>(smem), a_smem_layout);
    // Tensor sB = make_tensor(make_smem_ptr<bf16>(smem + size<0>(a_smem_layout)*size<1>(a_smem_layout)), b_smem_layout);
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("size<0>(a_smem_layout): %d\n", size<0>(a_smem_layout));
    //     printf("size<1>(a_smem_layout): %d\n", size<1>(a_smem_layout));
    // }
    __shared__ bf16 smemA[cosize_v<SmemLayoutA>];
    __shared__ bf16 smemB[cosize_v<SmemLayoutB>];  
    Tensor sA = make_tensor(make_smem_ptr(smemA), a_smem_layout);  // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), b_smem_layout);
    // We reuse the Shared Memory of A and B to store C.
    Tensor sC = make_tensor(make_smem_ptr((bf16*)smemA), c_smem_layout);

    // extern __shared__ char shared_memory[];
    // using SharedStorage = SharedStorage<bf16, bf16, SmemLayoutA, SmemLayoutB>;
    // SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    // Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), a_smem_layout);
    // Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), b_smem_layout);

    Tensor gA = make_tensor(make_gmem_ptr(A), select<0,2>(problem_shape), a_stride);
    Tensor gB = make_tensor(make_gmem_ptr(B), select<1,2>(problem_shape), b_stride);
    Tensor gOut = make_tensor(make_gmem_ptr(Out), select<0,1>(problem_shape), out_stride);

    // Use the plain CTA tiler for global memory; swizzled layout only applies to shared memory.
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);    
    auto gA_tile = local_tile(gA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    auto gB_tile = local_tile(gB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    auto gOut_tile = local_tile(gOut, cta_tiler, cta_coord, Step<_1,_1, X>{});

    /* For tensor A, we are left with a rank-3 tensor of shape (BLK_M,BLK_K,k). 
    The first two modes are precisely the modes of the CTA tile and the last mode indexes 
    over all of the tiles that will be reduced by this CTA. In the mainloop section below, 
    this mode is iterated over via the k_tile loop. */

    // Partition the tensors for this thread using the TiledCopy
    auto thr_copy_a = copyA.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA_tile);
    auto tAsA = thr_copy_a.partition_D(sA);  

    // Partition the tensors for this thread using the TiledCopy
    auto thr_copy_b = copyB.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB_tile);
    auto tBsB = thr_copy_b.partition_D(sB);  

    auto thr_mma = mma.get_slice(threadIdx.x);

    // Allocate MMA accumulator registers
    Tensor tCrA = thr_mma.partition_fragment_A(sA);               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);               // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCgC = thr_mma.partition_C(gOut_tile);                     // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

    clear(tCrC);

    // Partition the tensors for the copy to registers and MMA
    ThrCopy s2r_thr_copy_a = copyS2R_A.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);   

    ThrCopy s2r_thr_copy_b = copyS2R_B.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    // Partition the tensors for the write back
    auto thr_copy_out = copyOut.get_thread_slice(threadIdx.x);
    auto tCrOut = thr_copy_out.partition_S(tCrC);
    auto tCgOut = thr_copy_out.partition_D(gOut_tile);

    // Main MMA loop
    for (int k = 0; k < (K + 64 - 1) / 64; k += 1) { // BK=64
        // Async copy from global to shared
        copy(copyA, tAgA(_,_,_,k), tAsA);
        copy(copyB, tBgB(_,_,_,k), tBsB);
        
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Load fragments from shared memory to registers
        copy(copyS2R_A, tXsA, tXrA);
        copy(copyS2R_B, tXsB, tXrB);

        // MMA operation
        gemm(mma, tCrA, tCrB, tCrC);

        __syncthreads();
        }
    // Write back to global memory
    copy(tCrC, tCgC);  
    // copy(copyOut, tCrOut, tCgOut);
    // copy(copyOut, tCrC, tCgOut);
    
    // // SWIZZLED WRITE-BACK VIA SHARED MEMORY (slower than not aligned but direct register → global)

    // // -----------------------------------------------------------------
    // // Step 1: MMA Registers (F32) -> Shared Memory (BF16)
    // // -----------------------------------------------------------------
    
    // // Create a view of Shared Memory C from the MMA's perspective
    // auto tCsC_mma = thr_mma.partition_C(sC); 
    
    // // Direct conversion loop from F32 registers to Shared Memory (BF16)
    // // This avoids allocating a full register tensor for BF16 values.
    // #pragma unroll
    // for(int i = 0; i < size(tCrC); ++i) {
    //     tCsC_mma(i) = static_cast<bf16>(tCrC(i));
    // }

    // // Wait for all threads to finish writing to Shared Memory
    // __syncthreads(); 

    // // -----------------------------------------------------------------
    // // Step 2: Shared Memory (BF16) -> Global Memory (BF16)
    // // -----------------------------------------------------------------

    // // Create a view of Shared Memory C from the CopyOut (linear/tiled) perspective
    // auto thr_copy_out_slice = copyOut.get_thread_slice(threadIdx.x);
    
    // // Partition Shared Memory (Source)
    // Tensor tOsC_copy = thr_copy_out_slice.partition_S(sC); 
    
    // // Partition Global Memory (Dest)
    // Tensor tOgC_copy = thr_copy_out_slice.partition_D(gOut_tile);

    // // Copy directly from Shared Memory to Global Memory
    // // This avoids allocating a full register tensor for the copy.
    // copy(copyOut, tOsC_copy, tOgC_copy);
}

void runKernelCuteSwizzleMMAMatmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;

    auto prob_shape = make_shape(M, N, K);
    auto dA = make_stride(K, Int<1>{}); 
    auto dB = make_stride(K, Int<1>{});
    // auto dOut = make_stride(N, Int<1>{}); // Check because default is (1,M)
    auto dOut = make_stride(Int<1>{}, N); // Check because default is (1,M)

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    // Swizzled shared memory layout
    auto swizzled_128B_atom = composition(
                    Swizzle<3,3,3>{},
                    make_layout(
                        make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                        make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))
                    );
    auto sA = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{},Int<BK>{}));
    auto sB = tile_to_shape(swizzled_128B_atom, make_shape(Int<BN>{},Int<BK>{}));
    // auto sOut = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{},Int<BN>{})); // TODO: Do we need swizzling here?

    // printf("sA layout:\n");
    // print(sA);
    // printf("sB layout:\n");
    // print(sB);

    // Define Smem Layout for C (128x128)
    // We reuse the same atom, but tile it to (BM, BN)
    auto sC_layout = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{}, Int<BN>{}));

    // Simple async copy that works: 128 threads, each thread copies 128 bf16
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (64 threads)
        Layout<Shape<_1, _8>>{}                   // Each thread copies 8 bf16 (16 bytes)
    );

    // Simple async copy that works: 128 threads, each thread copies 128 bf16
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (64 threads)
        Layout<Shape<_1, _8>>{}                   // Each thread copies 8 bf16 (16 bytes)
    );
    
    // Regular copy for shared → global (Async not available)
    TiledCopy copyOut = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (128 threads)
        Layout<Shape<_1, _8>>{}                   // Each thread copies 8 bf16 (16 bytes)
    );
    // Regular copy for shared → global (Async not available)
    // TiledCopy copyOut = make_tiled_copy(
    //     Copy_Atom<UniversalCopy<uint16_t>, bf16>{},
    //     Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thread layout 8x8 (128 threads)
    //     Layout<Shape<_1, _8>>{}
    // );

    // MMA definition
    TiledMMA mmaABOut = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom;
    auto copyS2R_A = make_tiled_copy_A(s2r_atom, mmaABOut);
    auto copyS2R_B = make_tiled_copy_B(s2r_atom, mmaABOut);
    
    // printf("copyS2R_A:\n");
    // print(copyS2R_A);

    // int smem_elems = int(size(sA));
    // size_t smem_bytes = size_t(smem_elems) * sizeof(bf16);
    int grid_x = (M + BM - 1) / BM;
    int grid_y = (N + BN - 1) / BN;
    int block_threads = max(int(size(copyA)), int(size(copyB)), int(size(copyOut)), int(size(mmaABOut)));
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_threads);

    kernelCuteSwizzleMMAMatmul<<<dimGrid, dimBlock>>>(
        prob_shape, cta_tiler,
        A, dA, sA, 
        B, dB, sB,
        Out, dOut, sC_layout,
        copyA, copyB, copyOut, copyS2R_A, copyS2R_B, mmaABOut,
        M, N, K
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

} // namespace M0

using M0::runKernelCuteSwizzleMMAMatmul;