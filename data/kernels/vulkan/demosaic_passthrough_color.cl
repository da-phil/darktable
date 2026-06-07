// Vulkan port of demosaic_other.cl :: passthrough_color.
//
// Reads a single-channel mosaic and writes a float4 where the value
// is placed in the R, G, or B channel based on the Bayer / X-Trans
// CFA pattern at this pixel. This is a "debug" demosaic that shows
// the raw photosite colors without any interpolation.
//
// Binding layout (3 storage buffers):
//   0: in       (float, one value per pixel)
//   1: out      (float4)
//   2: xtrans   (uint[36] — only read for X-Trans; ignored for Bayer)
// Push constants: 12 B (width, height, filters).

#include "dt_vulkan_common.h"

kernel void demosaic_passthrough_color(
    global const float  *in_buf,
    global       float4 *out_buf,
    global const uint   *xtrans_flat,
    const int width, const int height,
    const uint filters)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  const float v = in_buf[idx];
  const int c = (filters == 9u)
                    ? vk_FCxtrans(y, x, xtrans_flat)
                    : vk_FC(y, x, filters);
  float4 out = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  if(c == 0)      out.x = v;
  else if(c == 1) out.y = v;
  else            out.z = v;
  out_buf[idx] = out;
}
