// Vulkan port of basic.cl :: colorzones_v3 (the "smooth" mode).
//
// Smooth per-zone adjustment working directly in (h, C) polar Lab. The
// hue-channel path blends toward neutral near the achromatic axis. Note
// the LUT roles: L scale <- table_L, hue shift <- table_b, C scale <-
// table_a (matching the OpenCL kernel's argument use).
//
// Bindings (5 storage buffers): 0 in, 1 out, 2 table_L, 3 table_a, 4 table_b.
// Push constants: 3 ints (width, height, channel) = 12 bytes.

#include "dt_vulkan_common.h"

#define VK_COLORZONES_L 0
#define VK_COLORZONES_C 1
#define VK_COLORZONES_h 2

static inline float vk_fsquare(const float x) { return x * x; }

kernel void colorzones_v3(global const float4 *in,
                          global       float4 *out,
                          global const float  *table_L,
                          global const float  *table_a,
                          global const float  *table_b,
                          const int width,
                          const int height,
                          const int channel)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float a = pixel.y;
  const float b = pixel.z;
  const float h = fmod(atan2(b, a) + DT_2PI_F, DT_2PI_F) / DT_2PI_F;
  const float C = vk_dt_fast_hypot(b, a);

  float select = 0.0f;
  float blend = 0.0f;
  switch(channel)
  {
    case VK_COLORZONES_L:
      select = fmin(1.0f, pixel.x / 100.0f);
      break;
    case VK_COLORZONES_C:
      select = fmin(1.0f, C / 128.0f);
      break;
    default:
    case VK_COLORZONES_h:
      select = h;
      blend = vk_fsquare(1.0f - C / 128.0f);
      break;
  }

  const float Lm = (blend * 0.5f + (1.0f - blend) * vk_lookup(table_L, select)) - 0.5f;
  const float hm = (blend * 0.5f + (1.0f - blend) * vk_lookup(table_b, select)) - 0.5f;
  blend *= blend; // saturation isn't as prone to artifacts (kept for parity)
  const float Cm = 2.0f * vk_lookup(table_a, select);
  const float L = pixel.x * pow(2.0f, 4.0f * Lm);

  pixel.x = L;
  pixel.y = cos(DT_2PI_F * (h + hm)) * Cm * C;
  pixel.z = sin(DT_2PI_F * (h + hm)) * Cm * C;

  out[idx] = pixel;
}
