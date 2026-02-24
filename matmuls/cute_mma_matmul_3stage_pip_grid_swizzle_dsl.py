"""
CuTe DSL Python port of cute_mma_matmul_2stage-pip_grid_swizzle.cu

A 2-stage pipelined GEMM kernel with grid swizzling for SM80 (Ampere) using BFloat16.
This is a port of kernel 11 (runkernelCutePipelinedSwizzleGridSwizzleMMAMatmul2StageImproved).

Key features:
- BFloat16 input with Float32 accumulator
- 2-stage async copy pipelining (gmem -> smem)
- Grid swizzling for improved L2 cache locality
- Swizzled shared memory layout (Swizzle<3,3,3>)
- ldmatrix for shared memory to register transfers
- SM80 tensor core MMA (16x8x16)

To run this example:
    python cute_mma_matmul_2stage_pip_grid_swizzle_dsl.py --mnk 8192,8192,8192
"""

import argparse
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
from cutlass.cute.runtime import from_dlpack

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Data types
io_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32

# Tile sizes (CTA Tiler)
BM = 128
BN = 128
BK = 64

# Pipeline stages
STAGES = 3

# Thread block size
NUM_THREADS = 128

# MMA instruction shape
MMA_INST_SHAPE = (16, 8, 16)

# Swizzle factor for grid swizzling
SWIZZLE_FACTOR = 8

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

class L2CacheClearer:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        # Get L2 cache size
        l2_size = torch.cuda.get_device_properties(self.device).L2_cache_size
        # Allocate 2x L2 size to be safe (like the C++ code)
        self.scratch_size = l2_size * 2
        # Allocate scratch buffer
        self.scratch = torch.zeros(self.scratch_size, dtype=torch.uint8, device=self.device)
    
    def clear(self):
        # Fill with zeros to flush L2 cache
        self.scratch.zero_()
        # Ensure operation completes
        torch.cuda.synchronize()

clearer = L2CacheClearer()

def clear_l2():
    clearer.clear()

# -----------------------------------------------------------------------------
# KERNEL
# -----------------------------------------------------------------------------
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
):
    """
    2-stage pipelined GEMM kernel with grid swizzling.
    
    :param mA: Input tensor A (M x K), row-major (K-major)
    :param mB: Input tensor B (N x K), row-major (K-major)  
    :param mC: Output tensor C (M x N), row-major (N-major)
    :param sA_layout: Shared memory layout for A
    :param sB_layout: Shared memory layout for B
    :param tiled_copy_A: Tiled copy for A (gmem -> smem)
    :param tiled_copy_B: Tiled copy for B (gmem -> smem)
    :param tiled_mma: Tiled MMA for tensor core operations
    :param M: Problem size M
    :param N: Problem size N
    :param K: Problem size K
    """
    # Thread and block indices
    tidx, tidy, tidz = arch.thread_idx()
    bidx, bidy, bidz = arch.block_idx()
    gdimx, gdimy, gdimz = arch.grid_dim()
    
    # -------------------------------------------------------------------------
    # 1. GRID SWIZZLING
    # -------------------------------------------------------------------------
    m_idx = bidx
    n_idx = bidy
    
    if gdimx >= SWIZZLE_FACTOR:
        tid = bidx + bidy * gdimx
        idx_outer = tid // SWIZZLE_FACTOR
        idx_inner = tid % SWIZZLE_FACTOR
        m_grid = gdimx
        n_swizzled = idx_outer // ((m_grid + SWIZZLE_FACTOR - 1) // SWIZZLE_FACTOR)
        m_swizzled = (idx_outer % ((m_grid + SWIZZLE_FACTOR - 1) // SWIZZLE_FACTOR)) * SWIZZLE_FACTOR + idx_inner
        if m_swizzled < m_grid:
            m_idx = m_swizzled
            n_idx = n_swizzled
    
    # -------------------------------------------------------------------------
    # 2. SHARED MEMORY ALLOCATION
    # -------------------------------------------------------------------------
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(io_dtype, sA_layout, 128)
    sB = smem.allocate_tensor(io_dtype, sB_layout, 128)
    
    # -------------------------------------------------------------------------
    # 3. GLOBAL MEMORY TENSORS & TILING
    # -------------------------------------------------------------------------
    # Get the tile for this CTA
    cta_tiler = (BM, BN, BK)
    tiler_coord = (m_idx, n_idx, None)
    
    # local_tile for A: (BM, BK, k_tiles) from (M, K)
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1))
    # local_tile for B: (BN, BK, k_tiles) from (N, K)
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
    # local_tile for C: (BM, BN) from (M, N)
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=tiler_coord, proj=(1, 1, None))
    
    # -------------------------------------------------------------------------
    # 4. PARTITIONING - COPY (gmem -> smem)
    # -------------------------------------------------------------------------
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)  # (CPY, CPY_M, CPY_K, k_tiles)
    tAsA = thr_copy_A.partition_D(sA)  # (CPY, CPY_M, CPY_K, STAGES)
    
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tBgB = thr_copy_B.partition_S(gB)  # (CPY, CPY_N, CPY_K, k_tiles)
    tBsB = thr_copy_B.partition_D(sB)  # (CPY, CPY_N, CPY_K, STAGES)
    
    # -------------------------------------------------------------------------
    # 5. PARTITIONING - MMA (for fragment creation)
    # -------------------------------------------------------------------------
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA_mma = thr_mma.partition_A(sA)  # MMA partitioning of sA
    tCsB_mma = thr_mma.partition_B(sB)  # MMA partitioning of sB
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA_mma[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB_mma[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)
    
    # -------------------------------------------------------------------------
    # 6. PARTITIONING - COPY (smem -> rmem) using ldmatrix
    # -------------------------------------------------------------------------
    # Create ldmatrix copy atoms for smem -> register transfer
    # Both A and B are row-major (K is contiguous), so transpose=False for both
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        io_dtype,
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        io_dtype,
    )
    
    # Create tiled copies matching the MMA layout
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
    
    thr_s2r_copy_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_s2r_copy_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsA_copy_view = thr_s2r_copy_A.partition_S(sA)
    tCrA_copy_view = thr_s2r_copy_A.retile(tCrA)
    tCsB_copy_view = thr_s2r_copy_B.partition_S(sB)
    tCrB_copy_view = thr_s2r_copy_B.retile(tCrB)
    
    # -------------------------------------------------------------------------
    # 7. PIPELINED MAIN LOOP (3-STAGE with register prefetch)
    # -------------------------------------------------------------------------
    k_tile_max = cute.size(tAgA, mode=[3])
    k_block_max = cute.size(tCrA, mode=[2])
    num_smem_stages = STAGES
    
    smem_write = cutlass.Int32(num_smem_stages - 1)
    smem_read = cutlass.Int32(0)
    k_tile_index = cutlass.Int32(0)
    
    # smem_read_current tracks the buffer we're currently reading from
    # smem_read is prepared for the next k_tile during k_block==0
    smem_read_current = cutlass.Int32(0)
    
    # Clear smem (for predication safety)
    tAsA.fill(0)
    tBsB.fill(0)
    arch.sync_threads()
    
    # PROLOGUE: Prefetch first (num_smem_stages - 1) k-tiles
    for k_tile in range(num_smem_stages - 1):
        if k_tile < k_tile_max:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, k_tile])
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, k_tile])
        k_tile_index = k_tile_index + 1
        arch.cp_async_commit_group()
    
    # Register prefetch: Wait for first tile and load first k-block to registers
    if k_block_max > 1:
        arch.cp_async_wait_group(num_smem_stages - 2)
        arch.sync_threads()
        # Prefetch first k-block from first smem buffer
        cute.copy(tiled_copy_s2r_A, tCsA_copy_view[None, None, 0, smem_read_current], tCrA_copy_view[None, None, 0])
        cute.copy(tiled_copy_s2r_B, tCsB_copy_view[None, None, 0, smem_read_current], tCrB_copy_view[None, None, 0])
    
    # Main loop over k-tiles
    for k_tile in range(k_tile_max):
        # Inner loop over k-blocks within a k-tile
        for k_block in cutlass.range_constexpr(k_block_max):
            
            # At the last k_block: switch to next smem buffer and wait
            # smem_read was updated at k_block==0, so we switch for the NEXT k_tile
            if k_block == k_block_max - 1:
                smem_read_current = smem_read  # Switch to next buffer for next k_tile
                arch.cp_async_wait_group(num_smem_stages - 2)
                arch.sync_threads()
            
            # 1. Issue Register Load for NEXT k-block (register pipeline)
            # Use smem_read_current which is stable during this k_tile's inner loop
            next_k_block = (k_block + 1) % k_block_max
            cute.copy(tiled_copy_s2r_A, tCsA_copy_view[None, None, next_k_block, smem_read_current], tCrA_copy_view[None, None, next_k_block])
            cute.copy(tiled_copy_s2r_B, tCsB_copy_view[None, None, next_k_block, smem_read_current], tCrB_copy_view[None, None, next_k_block])
            
            # 2. Issue Global Load for NEXT k-tile (only on first k_block)
            # Interleave: copy A, then gemm, then copy B for better latency hiding
            if k_block == 0:
                if k_tile + num_smem_stages - 1 < k_tile_max:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_write])
            
            # 3. Compute GEMM for CURRENT k-block
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)
            
            # 4. Complete Global Load and update pipeline state
            if k_block == 0:
                if k_tile + num_smem_stages - 1 < k_tile_max:
                    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_write])
                k_tile_index = k_tile_index + 1
                arch.cp_async_commit_group()
                # Prepare smem pointers for NEXT k_tile
                # smem_write takes the old smem_read value (buffer we just finished reading)
                # smem_read advances to next buffer (will be applied at k_block_max-1)
                smem_write = smem_read
                smem_read = smem_read + 1
                if smem_read == num_smem_stages:
                    smem_read = cutlass.Int32(0)
    
    # -------------------------------------------------------------------------
    # 8. EPILOGUE - Write back to global memory
    # -------------------------------------------------------------------------
    # Wait for all async copies to complete
    arch.cp_async_wait_group(0)
    arch.sync_threads()
    
    # Create output fragment with correct element type (BFloat16)
    tCrD = cute.make_fragment_like(tCrC, mC.element_type)
    # Convert accumulator (Float32) to output type (BFloat16)
    tCrD.store(tCrC.load().to(mC.element_type))
    
    # Copy to global memory using autovec_copy for better vectorization
    cute.autovec_copy(tCrD, tCgC)


# -----------------------------------------------------------------------------
# HOST FUNCTION
# -----------------------------------------------------------------------------
@cute.jit
def run_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
):
    """
    Host function to set up and launch the GEMM kernel.
    
    :param mA: Input tensor A
    :param mB: Input tensor B
    :param mC: Output tensor C
    :param stream: CUDA stream for execution
    """
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]
    
    # -------------------------------------------------------------------------
    # Swizzled Shared Memory Layout
    # -------------------------------------------------------------------------
    # Swizzle<3,3,3> composition with 128B atom for bank-conflict-free access
    # Must match CUDA layout exactly:
    #   make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{}))  // (8, (8, 8))
    #   make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{}))  // (8, (1, 64))
    # This is equivalent to (8, 64) with stride (8, 1) in flat form
    layout_atom_outer = cute.make_layout(
        (8, (8, 8)),
        stride=(8, (1, 64)),
    )
    
    # Create swizzle and compose with layout atom
    swizzle = cute.make_swizzle(3, 3, 3)
    swizzle_atom = cute.make_composed_layout(swizzle, 0, layout_atom_outer)
    
    sA_layout = cute.tile_to_shape(swizzle_atom, (BM, BK, STAGES), order=(0, 1, 2))
    sB_layout = cute.tile_to_shape(swizzle_atom, (BN, BK, STAGES), order=(0, 1, 2))
    
    # -------------------------------------------------------------------------
    # Tiled Copies (Async) - GMEM -> SMEM
    # -------------------------------------------------------------------------
    # CP.ASYNC with 128-bit (16 bytes = 8 bf16) loads
    atom_async_copy_A = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        io_dtype,
        num_bits_per_copy=128,
    )
    
    # Thread layout: 16x8 threads, each thread handles 1x8 elements
    thr_layout_copy = cute.make_layout((16, 8), stride=(8, 1))
    val_layout_copy = cute.make_layout((1, 8))
    
    tiled_copy_A = cute.make_tiled_copy_tv(
        atom_async_copy_A,
        thr_layout_copy,
        val_layout_copy,
    )
    
    atom_async_copy_B = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        io_dtype,
        num_bits_per_copy=128,
    )
    
    tiled_copy_B = cute.make_tiled_copy_tv(
        atom_async_copy_B,
        thr_layout_copy,
        val_layout_copy,
    )
    
    # -------------------------------------------------------------------------
    # Tiled MMA
    # -------------------------------------------------------------------------
    # SM80 BF16 MMA: 16x8x16, with 2x2x1 atom layout (M=2, N=2, K=1)
    # This creates a 32x32x16 tile per warp group (2*16 x 2*16 x 1*16)
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        MMA_INST_SHAPE,
    )
    
    # Atom layout: (M, N, K) = (2, 2, 1) -> 4 warps = 128 threads
    # This tiles the MMA atom 2x in M, 2x in N, and 1x in K
    atom_layout_mnk = (2, 2, 1)
    atom_layout = cute.make_layout(atom_layout_mnk)
    
    # Permutation tiler for MNK
    # After atom arrangement: M=2*16=32, N=2*8*2=32 (8*2 to get 16 for coalesced access), K=1*16=16
    permutation_mnk = (
        atom_layout_mnk[0] * MMA_INST_SHAPE[0],  # 2 * 16 = 32
        atom_layout_mnk[1] * MMA_INST_SHAPE[1] * 2,  # 2 * 8 * 2 = 32 (multiply by 2 for coalesced smem->rmem)
        atom_layout_mnk[2] * MMA_INST_SHAPE[2],  # 1 * 16 = 16
    )
    
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout,
        permutation_mnk=permutation_mnk,
    )
    
    # -------------------------------------------------------------------------
    # Grid and Block dimensions
    # -------------------------------------------------------------------------
    grid_x = (M + BM - 1) // BM
    grid_y = (N + BN - 1) // BN
    
    # Launch kernel using .launch() method
    gemm_kernel(
        mA, mB, mC,
        sA_layout, sB_layout,
        tiled_copy_A, tiled_copy_B,
        tiled_mma,
        M, N, K,
    ).launch(
        grid=[grid_x, grid_y, 1],
        block=[NUM_THREADS, 1, 1],
        stream=stream,
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_dense_gemm(mnk: Tuple[int, int, int], tolerance: float = 4.0):
    """
    Run the GEMM kernel and verify against PyTorch reference.
    
    :param mnk: Problem size (M, N, K)
    :param tolerance: Maximum absolute difference tolerance (default 4.0 for BF16 precision)
    """
    M, N, K = mnk
    
    # Create input tensors
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Create output tensor in column-major format to match CUDA kernel
    # CUDA kernel uses stride (1, N) which is column-major
    # We create (N, M) contiguous and transpose to get (M, N) with stride (1, N)
    C_storage = torch.zeros(N, M, dtype=torch.bfloat16, device="cuda")
    C = C_storage.T  # (M, N) view with stride (1, N) - column-major!
    
    # Create CuTe tensors from PyTorch tensors
    # A: (M, K) row-major (K is contiguous/leading dim)
    mA = from_dlpack(A, assumed_align=16)
    # B: (N, K) row-major (K is contiguous/leading dim)
    mB = from_dlpack(B, assumed_align=16)
    # C: (M, N) column-major (M is contiguous/leading dim) to match CUDA kernel
    mC = from_dlpack(C, assumed_align=16)
    
    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    run_gemm(mA, mB, mC, stream)
    
    # Synchronize
    torch.cuda.synchronize()
    
    # Compute reference using PyTorch's BF16 matmul (uses cuBLAS with F32 accumulation)
    # This should match the kernel's behavior since both use tile-based accumulation
    # Note: Using A @ B.T directly with BF16 goes through cuBLAS which matches our kernel's approach
    C_ref = torch.matmul(A, B.T)
    
    # For relative error comparison, use both absolute and relative tolerance
    # BFloat16 has limited precision (~7 bits mantissa), so we use relative tolerance
    abs_diff = (C - C_ref).abs()
    max_abs_diff = abs_diff.max().item()
    
    # Mean difference is a better indicator of overall correctness
    mean_diff = abs_diff.mean().item()
    
    # Pass if mean diff is very small (indicates kernel is fundamentally correct)
    # Mean diff threshold scales with K dimension (more accumulation = more error)
    mean_threshold = 0.001 * (K / 256)  # Allow more error for larger K
    passed = mean_diff < mean_threshold and max_abs_diff < tolerance
    
    if passed:
        print(f"PASSED: Max diff = {max_abs_diff:.6f}, Mean diff = {mean_diff:.6f}")
    else:
        print(f"FAILED: Max diff = {max_abs_diff:.6f}, Mean diff = {mean_diff:.6f} (tolerance = {tolerance})")
    
    return passed


def benchmark(mnk: Tuple[int, int, int], num_iters: int = 100):
    """
    Benchmark the GEMM kernel.
    
    :param mnk: Problem size (M, N, K)
    :param num_iters: Number of iterations for benchmarking
    """
    M, N, K = mnk
    
    # Create input tensors
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Create output tensor in column-major format to match CUDA kernel
    C_storage = torch.zeros(N, M, dtype=torch.bfloat16, device="cuda")
    C = C_storage.T  # (M, N) view with stride (1, N) - column-major!
    
    # Create CuTe tensors
    mA = from_dlpack(A, assumed_align=16)
    mB = from_dlpack(B, assumed_align=16)
    mC = from_dlpack(C, assumed_align=16)
    
    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    
    # Pre-compile the kernel once for better performance
    print("Compiling kernel with cute.compile ...")
    compiled_gemm = cute.compile(run_gemm, mA, mB, mC, stream)
    
    # Warmup with compiled kernel
    for _ in range(1000):
        clear_l2()
        compiled_gemm(mA, mB, mC, stream)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    timings = []
    for _ in range(num_iters):
        clear_l2()
        start.record()
        compiled_gemm(mA, mB, mC, stream)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    elapsed_ms = sum(timings) / num_iters
    
    # Calculate TFLOPS
    flops = 2 * M * N * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Average time: {elapsed_ms:.4f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")
    
    return elapsed_ms, tflops


def benchmark_cublas(mnk: Tuple[int, int, int], num_iters: int = 100):
    """
    Benchmark using cuBLAS via PyTorch's matmul (uses optimized backend).

    :param mnk: Problem size (M, N, K)
    :param num_iters: Number of iterations for benchmarking
    """
    M, N, K = mnk

    # Create input tensors (BF16 inputs - cuBLAS will use tensor cores with BF16)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # Standard layout for matmul


    # Warmup - use native BF16 matmul (cuBLAS uses tensor cores internally)
    for _ in range(1000):
        clear_l2()
        torch.matmul(A, B.T)
    torch.cuda.synchronize()

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(num_iters):
        start.record()
        clear_l2()
        torch.matmul(A, B.T)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    elapsed_ms = sum(timings) / num_iters
    flops = 2 * M * N * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    print(f"PyTorch BF16 Problem size: {M}x{N}x{K}")
    print(f"PyTorch Average time: {elapsed_ms:.4f} ms")
    print(f"PyTorch Performance: {tflops:.2f} TFLOPS")

    return elapsed_ms, tflops


def benchmark_sgemm_example(mnk: Tuple[int, int, int], num_iters: int = 10):
    """
    Benchmark the CUTLASS example tensorop_gemm.py for comparison.
    This uses tensor cores like our kernel for a fair comparison.

    :param mnk: Problem size (M, N, K)
    :param num_iters: Number of iterations for benchmarking
    """
    import sys
    import os
    
    # Add the CUTLASS examples path to import tensorop_gemm
    cutlass_examples_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../third_party/cutlass/examples/python/CuTeDSL/ampere"
    )
    cutlass_examples_path = os.path.normpath(cutlass_examples_path)
    
    if cutlass_examples_path not in sys.path:
        sys.path.insert(0, cutlass_examples_path)
    
    try:
        from tensorop_gemm import run as tensorop_run
    except ImportError as e:
        print(f"Could not import tensorop_gemm example: {e}")
        return None, None
    
    M, N, K = mnk
    
    print("Running tensorop_gemm example...")
    try:
        # Use the run function from tensorop_gemm
        avg_time_us = tensorop_run(
            a_major="m",  # Column-major for A
            b_major="n",  # Column-major for B
            c_major="n",  # Row-major for C
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            mnkl=(M, N, K, 1),
            atom_layout_mnk=(2, 2, 1),
            warmup_iterations=1000,
            iterations=num_iters,
            skip_ref_check=True,
            use_cold_l2=True,
        )
        
        elapsed_ms = avg_time_us / 1000.0
        flops = 2 * M * N * K
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        
        print(f"tensorop_gemm (FP16 TensorCore) Problem size: {M}x{N}x{K}")
        print(f"tensorop_gemm Average time: {elapsed_ms:.4f} ms")
        print(f"tensorop_gemm Performance: {tflops:.2f} TFLOPS")
        
        return elapsed_ms, tflops
    except Exception as e:
        print(f"tensorop_gemm benchmark failed: {e}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Stage Pipelined GEMM with Grid Swizzling")
    parser.add_argument(
        "--mnk",
        type=str,
        default="2048,2048,2048",
        help="Problem size M,N,K (comma-separated)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=4.0,
        help="Maximum absolute difference tolerance (default 4.0 for BF16 precision)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument(
        "--no-cute-own",
        action="store_true",
        help="Skip running the CuTe own kernel benchmark",
    )
    parser.add_argument(
        "--no-cublas",
        action="store_true",
        help="Skip running the cuBLAS benchmark",
    )
    parser.add_argument(
        "--no-cute-example",
        action="store_true",
        help="Skip running the CuTe example benchmark",
    )
    
    args = parser.parse_args()
    
    # Parse MNK
    mnk = tuple(int(x) for x in args.mnk.split(","))
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this example")
    
    print(f"Running 2-Stage Pipelined GEMM with Grid Swizzling")
    print(f"Problem size: {mnk[0]}x{mnk[1]}x{mnk[2]}")
    print(f"Tile size: {BM}x{BN}x{BK}")
    print(f"Stages: {STAGES}")
    print()
    
    # Verify correctness
    run_dense_gemm(mnk, args.tolerance)
    
    if args.benchmark:
        if not args.no_cute_own:
            print()
            # Run CuTe kernel benchmark
            cuda_elapsed, cuda_tflops = benchmark(mnk)

        if not args.no_cublas:
            print()
            # Run cuBLAS (PyTorch) benchmark for comparison
            cublas_elapsed, cublas_tflops = benchmark_cublas(mnk)
        
        if not args.no_cute_example:
            print()
            # Run sgemm example benchmark for comparison
            sgemm_elapsed, sgemm_tflops = benchmark_sgemm_example(mnk)

        # Print comparison summary
        print()
        print("=" * 60)
        print("Benchmark comparison:")
        print("=" * 60)
        if not args.no_cute_own:
            print(f"  CuTe BF16 kernel:          {cuda_elapsed:.4f} ms, {cuda_tflops:.2f} TFLOPS")
        if not args.no_cublas:
            print(f"  cuBLAS (PyTorch BF16):     {cublas_elapsed:.4f} ms, {cublas_tflops:.2f} TFLOPS")
        if not args.no_cute_example and sgemm_elapsed is not None:
            print(f"  tensorop_gemm (FP16 TC):   {sgemm_elapsed:.4f} ms, {sgemm_tflops:.2f} TFLOPS")
        print()
        if not args.no_cute_own:
            if not args.no_cublas and cublas_elapsed > 0:
                cute_vs_cublas = cuda_tflops / cublas_tflops * 100
                print(f"  CuTe BF16 vs cuBLAS: {cute_vs_cublas:.1f}%")
            if not args.no_cute_example and sgemm_elapsed is not None and sgemm_elapsed > 0:
                cute_vs_tensorop = cuda_tflops / sgemm_tflops * 100
                print(f"  CuTe BF16 vs tensorop_gemm: {cute_vs_tensorop:.1f}%")
        print("=" * 60)
