#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace M2 {

template <class ProblemShape,
          class CtaTiler,
          class StrideA, class SmemLayoutA,
          class StrideB, class SmemLayoutB,
          class StrideC, class SmemLayoutC,
          class CopyA, class CopyB, class CopyOut, 
          class CopyS2R_A, class CopyS2R_B, class TiledMMA>
__global__ void kernelCuteSwizzledPipeline2StageOptimized(
                                        ProblemShape problem_shape, CtaTiler cta_tiler,
                                        bf16 *A, StrideA a_stride, SmemLayoutA a_smem_layout,
                                        bf16 *B, StrideB b_stride, SmemLayoutB b_smem_layout,
                                        bf16 *Out, StrideC out_stride, SmemLayoutC c_smem_layout,
                                        CopyA copyA, CopyB copyB, CopyOut copyOut,
                                        CopyS2R_A copyS2R_A, CopyS2R_B copyS2R_B, TiledMMA mma,
                                        int M, int N, int K) {
    
    // ----------------------------------------------------------------
    // 1. GRID SWIZZLING
    // ----------------------------------------------------------------
    int m_idx = blockIdx.x;
    int n_idx = blockIdx.y;
    constexpr int swizzle_factor = 8; 
    if (gridDim.x >= swizzle_factor) {
        int tid = blockIdx.x + blockIdx.y * gridDim.x;
        int idx_outer = tid / swizzle_factor;
        int idx_inner = tid % swizzle_factor;
        int m_grid = gridDim.x;
        int n_swizzled = idx_outer / ((m_grid + swizzle_factor - 1) / swizzle_factor);
        int m_swizzled = (idx_outer % ((m_grid + swizzle_factor - 1) / swizzle_factor)) * swizzle_factor + idx_inner;
        if (m_swizzled < m_grid) {
             m_idx = m_swizzled;
             n_idx = n_swizzled;
        }
    }
    auto cta_coord = make_coord(m_idx, n_idx, _);

    // ----------------------------------------------------------------
    // 2. SHARED MEMORY SETUP
    // ----------------------------------------------------------------
    extern __shared__ char smem_[];
    bf16* smem_ptr = reinterpret_cast<bf16*>(smem_);
    
    Tensor sA = make_tensor(make_smem_ptr(smem_ptr), a_smem_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem_ptr + cosize_v<SmemLayoutA>), b_smem_layout);
    Tensor sC = make_tensor(make_smem_ptr(smem_ptr), c_smem_layout);

    // ----------------------------------------------------------------
    // 3. GLOBAL MEMORY TENSORS
    // ----------------------------------------------------------------
    Tensor gA = make_tensor(make_gmem_ptr(A), select<0,2>(problem_shape), a_stride);
    Tensor gB = make_tensor(make_gmem_ptr(B), select<1,2>(problem_shape), b_stride);
    Tensor gOut = make_tensor(make_gmem_ptr(Out), select<0,1>(problem_shape), out_stride);

    auto gA_tile = local_tile(gA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    auto gB_tile = local_tile(gB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    auto gOut_tile = local_tile(gOut, cta_tiler, cta_coord, Step<_1,_1, X>{});

    // ----------------------------------------------------------------
    // 4. PARTITIONING
    // ----------------------------------------------------------------
    auto thr_copy_a = copyA.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA_tile);
    auto tAsA = thr_copy_a.partition_D(sA);  

    auto thr_copy_b = copyB.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB_tile);
    auto tBsB = thr_copy_b.partition_D(sB);  

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gOut_tile);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    auto s2r_thr_copy_a = copyS2R_A.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); 
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);   

    auto s2r_thr_copy_b = copyS2R_B.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0)); 
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    // ----------------------------------------------------------------
    // 5. PIPELINED MAIN LOOP (2-STAGE, SIMPLE LOOP)
    // ----------------------------------------------------------------
    constexpr int STAGES = 2;
    int k_tile_max = size<3>(tAgA);
    int smem_write = 0;
    int smem_read = 0;

    // PROLOGUE: Load k=0
    copy(copyA, tAgA(_,_,_,0), tAsA(_,_,_,smem_write));
    copy(copyB, tBgB(_,_,_,0), tBsB(_,_,_,smem_write));
    cp_async_fence();
    smem_write = (smem_write + 1) % STAGES;

    // Wait for k=0
    cp_async_wait<0>();
    __syncthreads();

    // Load Regs from 0
    auto K_BLOCK_MAX = size<2>(tCrA);
    copy(copyS2R_A, tXsA(_,_,Int<0>{},smem_read), tXrA(_,_,Int<0>{}));
    copy(copyS2R_B, tXsB(_,_,Int<0>{},smem_read), tXrB(_,_,Int<0>{}));

    int k_tile_next = 1;

    for (int k = 0; k < k_tile_max; k++) {
        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {
            
            // 1. Issue Global Load for NEXT tile
            if (k_block == 0 && k_tile_next < k_tile_max) {
                 copy(copyA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_write));
                 copy(copyB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_write));
                 cp_async_fence();
                 smem_write = (smem_write + 1) % STAGES;
                 k_tile_next++;
            }

            // 2. Handle Smem Switch
            if (k_block == K_BLOCK_MAX - 1) {
                smem_read = (smem_read + 1) % STAGES;
                cp_async_wait<0>();
                __syncthreads();
            }

            // 3. Issue Register Load for NEXT block
            int next_k_block = (k_block + 1) % K_BLOCK_MAX;
            copy(copyS2R_A, tXsA(_,_,next_k_block,smem_read), tXrA(_,_,next_k_block));
            copy(copyS2R_B, tXsB(_,_,next_k_block,smem_read), tXrB(_,_,next_k_block));

            // 4. Math for CURRENT block
            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
    }

    copy(tCrC, tCgC);
}

// -----------------------------------------------------------------------------
// HOST RUNNER
// -----------------------------------------------------------------------------
void runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul2StageImproved(int M, int N, int K, bf16 *A, bf16 *B, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int STAGES = 2; // Reverted to 2 stages for occupancy

    auto prob_shape = make_shape(M, N, K);
    auto dA = make_stride(K, Int<1>{}); 
    auto dB = make_stride(K, Int<1>{});
    auto dOut = make_stride(Int<1>{}, N); 

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    // Swizzle Atom: Swizzle<3,3,3> (Standard)
    auto swizzled_128B_atom = composition(
                    Swizzle<3,3,3>{},
                    make_layout(
                        make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                        make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))
                    );

    auto sA = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{}));
    auto sB = tile_to_shape(swizzled_128B_atom, make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{}));
    auto sC_layout = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{}, Int<BN>{}));

    // Tiled Copies (Async) - CACHEALWAYS (Faster than CACHEGLOBAL here)
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );
    
    // Optimized Write Back (32-bit alignment)
    TiledCopy copyOut = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _2>>{}
    );

    TiledMMA mmaABOut = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                 Layout<Shape<_2,_2>>{},
                                 Tile<_32,_32,_16>{});

    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom;
    auto copyS2R_A = make_tiled_copy_A(s2r_atom, mmaABOut);
    auto copyS2R_B = make_tiled_copy_B(s2r_atom, mmaABOut);
    
    int smem_elems_A = int(cosize(sA));
    int smem_elems_B = int(cosize(sB));
    size_t smem_bytes = (smem_elems_A + smem_elems_B) * sizeof(bf16);

    int grid_x = (M + BM - 1) / BM;
    int grid_y = (N + BN - 1) / BN;
    int block_threads = max(int(size(copyA)), int(size(copyB)), int(size(copyOut)), int(size(mmaABOut)));
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_threads);

    void (*kernel_ptr)(
        decltype(prob_shape), decltype(cta_tiler),
        bf16*, decltype(dA), decltype(sA),
        bf16*, decltype(dB), decltype(sB),
        bf16*, decltype(dOut), decltype(sC_layout), 
        decltype(copyA), decltype(copyB), decltype(copyOut),
        decltype(copyS2R_A), decltype(copyS2R_B), decltype(mmaABOut),
        int, int, int
    ) = kernelCuteSwizzledPipeline2StageOptimized;

    cudaError_t err = cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes
    );
    
    if (err != cudaSuccess) {
        printf("SetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    kernel_ptr<<<dimGrid, dimBlock, smem_bytes>>>(
        prob_shape, cta_tiler,
        A, dA, sA, 
        B, dB, sB,
        Out, dOut, sC_layout, 
        copyA, copyB, copyOut, copyS2R_A, copyS2R_B, mmaABOut,
        M, N, K
    );
}

} // namespace M2

using M2::runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul2StageImproved;
    