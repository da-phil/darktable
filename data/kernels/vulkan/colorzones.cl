// Vulkan port of basic.cl :: colorzones (the "strong" mode).
//
// Per-zone Lab tone/chroma/hue adjustment. A selector value (lightness,
// chroma, or hue of the pixel) indexes three 65536-entry LUTs that
// scale L, scale C, and shift h in LCH space.
//
// Bindings (5 storage buffers): 0 in, 1 out, 2 table_L, 3 table_C, 4 table_h.
// Push constants: 3 ints (width, height, channel) = 12 bytes.

#include "dt_vulkan_common.h"

// Mirror dt_iop_colorzones_channel_t.
#define VK_COLORZONES_L 0
#define VK_COLORZONES_C 1
#define VK_COLORZONES_h 2

#ifndef M_SQRT2_F
#define M_SQRT2_F 1.41421356237309504880f
#endif

kernel void colorzones(global const float4 *in,
                       global       float4 *out,
                       global const float  *table_L,
                       global const float  *table_C,
                       global const float  *table_h,
                       const int width,
                       const int height,
                       const int channel)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float normalize_C = 1.0f / (128.0f * M_SQRT2_F);

  float4 LCh = vk_Lab_2_LCH(pixel);

  float select = 0.0f;
  switch(channel)
  {
    case VK_COLORZONES_L:
      select = LCh.x * 0.01f;
      break;
    case VK_COLORZONES_C:
      select = LCh.y * normalize_C;
      break;
    case VK_COLORZONES_h:
    default:
      select = LCh.z;
      break;
  }
  select = clipf(select);

  LCh.x *= pow(2.0f, 4.0f * (vk_lookup(table_L, select) - 0.5f));
  LCh.y *= 2.0f * vk_lookup(table_C, select);
  LCh.z += vk_lookup(table_h, select) - 0.5f;

  const float4 lab = vk_LCH_2_Lab(LCh);
  pixel.x = lab.x;
  pixel.y = lab.y;
  pixel.z = lab.z;

  out[idx] = pixel;
}
