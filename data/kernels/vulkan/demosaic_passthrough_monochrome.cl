// Vulkan port of demosaic_other.cl :: passthrough_monochrome.
//
// Reads a single-channel float input (the raw mosaic for a true-mono
// sensor) and replicates the value across all 3 colour channels of a
// float4 output. The simplest demosaic "algorithm" — used when the
// sensor has no CFA at all.
//
// Binding layout (2 storage buffers):
//   0: in   (float, one value per pixel)
//   1: out  (float4)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void demosaic_passthrough_monochrome(
    global const float  *in_buf,
    global       float4 *out_buf,
    const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  const float v = in_buf[idx];
  out_buf[idx] = (float4)(v, v, v, v);
}
