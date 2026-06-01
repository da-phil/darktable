// Vulkan port of locallaplacian.cl :: write_back.
//
// Writes the processed L (×100 to restore the [0, 100] Lab scale)
// back into the output float4 buffer, copying chroma + alpha from
// the input. Padded buffer indexing offsets by (max_supp, max_supp)
// to strip the border.
//
// Binding layout (3 storage buffers):
//   0: in        (float4)  — original image at (wd, ht)
//   1: processed (float)   — padded monochrome output at
//                            (wd + 2*max_supp, ht + 2*max_supp)
//   2: out       (float4)  — write-back at (wd, ht)
// Push constants: 4 ints = 16 bytes (max_supp, wd, ht, padded_w).

#include "dt_vulkan_common.h"

kernel void ll_write_back(global const float4 *in,
                          global const float  *processed,
                          global       float4 *out,
                          const int max_supp,
                          const int wd,
                          const int ht,
                          const int padded_w)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= wd || y >= ht) return;

  const int idx = y * wd + x;
  const int pidx = (y + max_supp) * padded_w + (x + max_supp);

  float4 pixel = in[idx];
  pixel.x = 100.0f * processed[pidx];
  out[idx] = pixel;
}
