#include <cute/tensor.hpp>

using namespace cute;

typedef __nv_bfloat16 bf16;

int main()
{
#if 0
  {
    print_latex(make_layout(
                make_shape(Int<4>{}, Int<8>{}),
                make_stride(Int<8>{}, Int<1>{})
              ));
     }
#endif 
#if 0
  {
    8x 64 matrix
    print_latex(make_layout(
                make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{}))
              ));
     }
#endif 
#if 0
  {
    16x 64
    print_latex(make_layout(
                make_shape(make_shape(Int<2>{}, Int<8>{}), make_shape(Int<8>{}, Int<8>{})),
                make_stride(make_stride(Int<8>{}, Int<16>{}), make_stride(Int<1>{}, Int<128>{}))
              ));
     }
#endif 
#if 1
  {
    print_latex(
     composition(
        Swizzle<3,3,3>{},                               // the swizzle transform
        make_layout(make_shape(Int<64>{}, Int<64>{}),   // the base (row-major) layout
                    make_stride(Int<64>{}, Int<1>{}))
    )
  );
     }
#endif 
#if 0
  {
  // 16x64 matrix mma layout for ldmatrix
  // print_latex(
  //   composition(Swizzle<3,3,3>{},
  //                                 Layout<Shape <_8,Shape <_8, _8>>,
  //                                        Stride<_8,Stride<_1,_64>>>{})     // 16x64x16 Tiled MMA for LDSM
  // );
  // Repeat it
  print_latex(
    tile_to_shape(
      composition(Swizzle<3,3,3>{},
                  make_layout(
                    make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                    make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))),
      make_shape(Int<64>{},Int<64>{})
    )
  );

  // print_latex(
  //   make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
  //                                Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
  //                                Tile<_32,_32,_16>{})     // 32x32x16 Tiled MMA for LDSM
  // );
              
  }
#endif
}