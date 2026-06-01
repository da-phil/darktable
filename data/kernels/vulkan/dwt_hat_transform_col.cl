// Vulkan port of dwt.cl :: dwt_hat_transform_col.
//
// Vertical arm of the a-trous hat transform; applies the 1/16
// `lpass_mult` normalisation on the way out so the row+col pair
// produces the low-pass output for the current scale.
//
// Binding layout (2 storage buffers):
//   0: lpass       (float4)  — source (row-transformed input)
//   1: temp_buffer (float4)  — destination (low-pass output for this scale)
// Push constants: 3 ints + 1 float = 16 bytes (width, height, sc, lpass_mult).

#include "dt_vulkan_common.h"

kernel void dwt_hat_transform_col(global const float4 *lpass,
                                  const int width,
                                  const int height,
                                  const int sc,
                                  global       float4 *temp_buffer,
                                  const float lpass_mult)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width) return;

  const float hat_mult = 2.0f;
  const int size = height;

  if(y < sc)
  {
    temp_buffer[y * width + x] =
        (hat_mult * lpass[y * width + x]
         + lpass[(sc - y) * width + x]
         + lpass[(y + sc) * width + x]) * lpass_mult;
  }
  else if(y + sc < size)
  {
    temp_buffer[y * width + x] =
        (hat_mult * lpass[y * width + x]
         + lpass[(y - sc) * width + x]
         + lpass[(y + sc) * width + x]) * lpass_mult;
  }
  else if(y < size)
  {
    temp_buffer[y * width + x] =
        (hat_mult * lpass[y * width + x]
         + lpass[(y - sc) * width + x]
         + lpass[(2 * size - 2 - (y + sc)) * width + x]) * lpass_mult;
  }
}
