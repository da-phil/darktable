// Vulkan port of locallaplacian.cl :: pad_input.
//
// Pads the input float4 image (`wd`×`ht`) with `max_supp` border
// pixels using edge replication, taking the L channel only and
// scaling by 0.01 so the curve operates in [0, 1]. Writes a flat
// single-channel `padded` buffer sized `wd2`×`ht2`.
//
// Image-shortcut port: the OpenCL kernel reads `input` as image2d_t
// with `(int2)(cx, cy)` integer-clamp coords, which translates to
// flat buffer access with a manual `clamp` (no real sampler
// filtering needed). Same trick used by overlay / sigmoid / agx.
//
// Binding layout (2 storage buffers):
//   0: in     (float4) — original image
//   1: padded (float)  — monochrome padded output
// Push constants: 5 ints = 20 bytes (wd, ht, max_supp, wd2, ht2).

#include "dt_vulkan_common.h"

kernel void ll_pad_input(global const float4 *in,
                         global       float  *padded,
                         const int wd,
                         const int ht,
                         const int max_supp,
                         const int wd2,
                         const int ht2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= wd2 || y >= ht2) return;

  const int cx = clamp(x - max_supp, 0, wd - 1);
  const int cy = clamp(y - max_supp, 0, ht - 1);

  padded[y * wd2 + x] = in[cy * wd + cx].x * 0.01f;
}
