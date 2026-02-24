#include <cute/tensor.hpp>

using namespace cute;

typedef __nv_bfloat16 bf16;

int main()
{
#if 1
  {
  print_latex(
     make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                 Layout<Shape<_2,_2>>{},
                                 Tile<_32,_32,_16>{})
  );


  }
#endif
}