// https://github.com/NVIDIA/cutlass/discussions/1142

/* copy_tiled_tex.cpp
 * $$ make copy_tiled_tex && ./copy_tiled_tex > copy_tiled.tex && pdflatex copy_tiled.pdf
 */

#include <cute/tensor.hpp>

using namespace cute;

typedef __nv_bfloat16 bf16;

int main()
{
#if 0
  {
    // Base Copy
    auto tiled_copy =  make_tiled_copy(
        Copy_Atom<DefaultCopy, bf16>{},
        Layout<Shape<_1, _64>, Stride<_64, _1>>{}, // ThrLayout
        Layout<Shape<_1, _1>>{});                  // ValLayout
    print_latex(tiled_copy);
  }
#endif
#if 0
  {
    // Vectorized Copy
    auto tiled_copy =  make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
                        Layout<Shape<_4,_8>, Stride<_8,_1>>{},   // thread layout: 32 threads
                        Layout<Shape<_1,_8>, Stride<_8,_1>>{});     // per-thread value layout: 8 contiguous elements
    print_latex(tiled_copy);
    }
#endif
#if 0
  {
    // Async Copy
    auto tiled_copy =  make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, bf16>{},
        Layout<Shape<_4, _8>, Stride<_8, _1>>{},     // 32 threads
        Layout<Shape<_1,_8>, Stride<_8, _1>>{}     // Each copies 8 contiguous elements
    );
    print_latex(tiled_copy);
    }
#endif
#if 0
  {
    Copy_Atom<UniversalCopy<double>, double> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_32,_1>>{},  // 32x1 threads
                                      Layout<Shape< _1,_4>>{}); //  1x4 values
    print_latex(tiled_copy);

    }
#endif

#if 0
  {
    // The canonical LDSM_N image
    Copy_Atom<SM75_U32x1_LDSM_N, uint32_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_8,_4>, Stride<_4,_1>>{}); // 8x4 RowMajor threads
    print_latex(tiled_copy);

    }
#endif

#if 0
  {
    // The canonical LDSM_T image
    Copy_Atom<SM75_U16x2_LDSM_T, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_8,_4>, Stride<_4,_1>>{},  // 8x4 RowMajor threads
                                      Layout<Shape<_1,_2>>{});                // 1x2 values per thread
    print_latex(tiled_copy);

    }
#endif

#if 1
  {
    // // Generate a TiledCopy layout from a TiledMMA
    TiledMMA mmaABOut = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}); // 32x32x16 Tiled MMA for LDSM

    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom;
    auto copyS2R_A = make_tiled_copy_A(s2r_atom, mmaABOut);
    print_latex(copyS2R_A);
  }
#endif

#if 0
  {
    Copy_Atom<SM75_U16x2_LDSM_T, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                      Layout<Shape<_2,_1>>{});
    print_latex(tiled_copy);
    }
#endif

#if 0
  {
    Copy_Atom<SM75_U16x2_LDSM_N, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_32,_1>>{},
                                      Layout<Shape< _1,_8>>{});
    print_latex(tiled_copy);
    }
#endif
}