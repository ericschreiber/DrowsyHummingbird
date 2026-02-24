#include <cute/tensor.hpp>

using namespace cute;
typedef __nv_bfloat16 bf16;

namespace M3 {

template <class ProblemShape,
          class CtaTiler,
          class StrideA, class SmemLayoutA,
          class StrideB, class SmemLayoutB,
          class StrideC, class SmemLayoutC,
          class CopyA, class CopyB, class CopyOut, 
          class CopyS2R_A, class CopyS2R_B, class TiledMMA,
          int STAGES>
__global__ void kernelCuteSwizzledPipeline3StageOptimized(
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
    // 4. PARTITIONING - COPY (gmem -> smem)
    // ----------------------------------------------------------------
    auto thr_copy_a = copyA.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA_tile);  // (CPY, CPY_M, CPY_K, k_tiles)
    auto tAsA = thr_copy_a.partition_D(sA);       // (CPY, CPY_M, CPY_K, STAGES)

    auto thr_copy_b = copyB.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB_tile);  // (CPY, CPY_N, CPY_K, k_tiles)
    auto tBsB = thr_copy_b.partition_D(sB);       // (CPY, CPY_N, CPY_K, STAGES)

    // ----------------------------------------------------------------
    // 5. PARTITIONING - MMA (for fragment creation)
    // ----------------------------------------------------------------
    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gOut_tile);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // ----------------------------------------------------------------
    // 6. PARTITIONING - COPY (smem -> rmem) using ldmatrix
    // ----------------------------------------------------------------
    auto s2r_thr_copy_a = copyS2R_A.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); 
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);   

    auto s2r_thr_copy_b = copyS2R_B.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0)); 
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    // ----------------------------------------------------------------
    // 7. PIPELINED MAIN LOOP (3-STAGE with register prefetch)
    // ----------------------------------------------------------------
    int k_tile_max = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);
    constexpr int num_smem_stages = STAGES;

    int smem_write = num_smem_stages - 1;
    int smem_read = 0;
    int k_tile_index = 0;
    
    // smem_read_current tracks the buffer we're currently reading from
    int smem_read_current = 0;

    // Zero-initialize the smem partitions
    CUTE_UNROLL
    for (int i = 0; i < size(tAsA); ++i) { tAsA(i) = bf16(0); }
    CUTE_UNROLL
    for (int i = 0; i < size(tBsB); ++i) { tBsB(i) = bf16(0); }
    __syncthreads();

    // PROLOGUE: Prefetch first (num_smem_stages - 1) k-tiles
    CUTE_UNROLL
    for (int k_tile = 0; k_tile < num_smem_stages - 1; k_tile++) {
        if (k_tile < k_tile_max) {
            copy(copyA, tAgA(_,_,_,k_tile_index), tAsA(_,_,_,k_tile));
            copy(copyB, tBgB(_,_,_,k_tile_index), tBsB(_,_,_,k_tile));
        }
        k_tile_index++;
        cp_async_fence();
    }

    // Register prefetch: Wait for first tile and load first k-block to registers
    if (K_BLOCK_MAX > 1) {
        cp_async_wait<num_smem_stages - 2>();
        __syncthreads();
        // Prefetch first k-block from first smem buffer
        copy(copyS2R_A, tXsA(_,_,Int<0>{},smem_read_current), tXrA(_,_,Int<0>{}));
        copy(copyS2R_B, tXsB(_,_,Int<0>{},smem_read_current), tXrB(_,_,Int<0>{}));
    }

    // Main loop over k-tiles
    for (int k_tile = 0; k_tile < k_tile_max; k_tile++) {
        // Inner loop over k-blocks within a k-tile
        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {
            
            // At the last k_block: switch to next smem buffer and wait
            if (k_block == K_BLOCK_MAX - 1) {
                smem_read_current = smem_read;  // Switch to next buffer for next k_tile
                cp_async_wait<num_smem_stages - 2>();
                __syncthreads();
            }

            // 1. Issue Register Load for NEXT k-block (register pipeline)
            int next_k_block = (k_block + 1) % K_BLOCK_MAX;
            copy(copyS2R_A, tXsA(_,_,next_k_block,smem_read_current), tXrA(_,_,next_k_block));
            copy(copyS2R_B, tXsB(_,_,next_k_block,smem_read_current), tXrB(_,_,next_k_block));

            // 2. Issue Global Load for NEXT k-tile (only on first k_block)
            // Interleave: copy A, then gemm, then copy B for better latency hiding
            if (k_block == 0) {
                if (k_tile + num_smem_stages - 1 < k_tile_max) {
                    copy(copyA, tAgA(_,_,_,k_tile_index), tAsA(_,_,_,smem_write));
                }
            }

            // 3. Compute GEMM for CURRENT k-block
            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);

            // 4. Complete Global Load and update pipeline state
            if (k_block == 0) {
                if (k_tile + num_smem_stages - 1 < k_tile_max) {
                    copy(copyB, tBgB(_,_,_,k_tile_index), tBsB(_,_,_,smem_write));
                }
                k_tile_index++;
                cp_async_fence();
                // Prepare smem pointers for NEXT k_tile
                // smem_write takes the old smem_read value (buffer we just finished reading)
                // smem_read advances to next buffer (will be applied at k_block_max-1)
                smem_write = smem_read;
                smem_read = smem_read + 1;
                if (smem_read == num_smem_stages) {
                    smem_read = 0;
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // 8. EPILOGUE - Write back to global memory
    // ----------------------------------------------------------------
    // Wait for all async copies to complete
    cp_async_wait<0>();
    __syncthreads();

    // Implicit bf16 conversion during copy out
    copy(tCrC, tCgC);
}

// -----------------------------------------------------------------------------
// HOST RUNNER
// -----------------------------------------------------------------------------
void runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul3StageImproved(int M, int N, int K, bf16 *A, bf16 *B, bf16 *Out) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int STAGES = 3;

    auto prob_shape = make_shape(M, N, K);
    auto dA = make_stride(K, Int<1>{}); 
    auto dB = make_stride(K, Int<1>{});
    auto dOut = make_stride(Int<1>{}, N); 

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    // Swizzle Atom: Swizzle<3,3,3>
    auto swizzled_128B_atom = composition(
                    Swizzle<3,3,3>{},
                    make_layout(
                        make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                        make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))
                    );

    auto sA = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{}));
    auto sB = tile_to_shape(swizzled_128B_atom, make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{}));
    auto sC_layout = tile_to_shape(swizzled_128B_atom, make_shape(Int<BM>{}, Int<BN>{}));

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // 16x8 thread layout
        Layout<Shape<_1, _8>>{}                      // 1x8 value layout (8 bf16 = 128 bits)
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );
    
    TiledCopy copyOut = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, bf16>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _2>>{}
    );

    // TiledMMA: SM80 16x8x16 with 2x2 atom layout -> 32x32x16 tile
    TiledMMA mmaABOut = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                 Layout<Shape<_2,_2>>{},
                                 Tile<_32,_32,_16>{});

    // Smem to Register copy using ldmatrix
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
    ) = kernelCuteSwizzledPipeline3StageOptimized<
            decltype(prob_shape), decltype(cta_tiler),
            decltype(dA), decltype(sA),
            decltype(dB), decltype(sB),
            decltype(dOut), decltype(sC_layout),
            decltype(copyA), decltype(copyB), decltype(copyOut),
            decltype(copyS2R_A), decltype(copyS2R_B), decltype(mmaABOut),
            STAGES>;

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

} // namespace M3

using M3::runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul3StageImproved;