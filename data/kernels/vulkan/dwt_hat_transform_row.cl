// Vulkan port of dwt.cl :: dwt_hat_transform_row.
//
// Horizontal arm of the a-trous hat (`B3 spline`) transform: at every
// pixel reads the centre tap and the two ±sc neighbours, mirrors at
// the image edges, and writes the weighted sum into the output (no
// scaling here — the col pass applies the 1/16 normalisation).
//
// Binding layout (2 storage buffers):
//   0: lpass (float4)  — destination (transformed row output)
//   1: hpass (float4)  — source (current-scale image buffer)
// Push constants: 3 ints + 1 int = 16 bytes (width, height, sc, _pad).

#include "dt_vulkan_common.h"

kernel void dwt_hat_transform_row(global       float4 *lpass,
                                  global const float4 *hpass,
                                  const int width,
                                  const int height,
                                  const int sc)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(y >= height) return;

  const float hat_mult = 2.0f;
  const int size = width;

  if(x < sc)
  {
    const int idx = y * width + x;
    lpass[idx] = hat_mult * hpass[y * width + x]
               + hpass[y * width + (sc - x)]
               + hpass[y * width + (x + sc)];
  }
  else if(x + sc < size)
  {
    const int idx = y * width + x;
    lpass[idx] = hat_mult * hpass[y * width + x]
               + hpass[y * width + (x - sc)]
               + hpass[y * width + (x + sc)];
  }
  else if(x < size)
  {
    const int idx = y * width + x;
    lpass[idx] = hat_mult * hpass[y * width + x]
               + hpass[y * width + (x - sc)]
               + hpass[y * width + (2 * size - 2 - (x + sc))];
  }
}
