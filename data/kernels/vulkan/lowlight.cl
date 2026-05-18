// Vulkan port of lowlight (basic.cl :: kernel lowlight).
//
// Blends the input Lab pixel toward a scotopic (low-light) variant
// of itself, with the blend coefficient driven by a per-piece tone
// curve (1 LUT, 65536 entries). The scotopic white-point XYZ_sw
// is precomputed host-side from the user's blueness slider.
//
// Bindings (3 storage buffers):
//   0: in   (float4)  — Lab pixels (.x is L*)
//   1: out  (float4)
//   2: lut  (float)   — 65536-entry blend curve
//
// Push constants: 2 ints + 4 floats = 24 bytes (XYZ_sw components
// passed as 3 scalars + .w padding to dodge the std430 vec4 trap).

#include "dt_vulkan_common.h"

kernel void lowlight(global const float4 *in,
                     global       float4 *out,
                     global const float  *lut,
                     const int   width,
                     const int   height,
                     const float XYZ_sw_x,
                     const float XYZ_sw_y,
                     const float XYZ_sw_z,
                     const float XYZ_sw_w)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float c = 0.5f;
  const float threshold = 0.01f;

  float4 pixel = in[idx];
  float4 XYZ = vk_Lab_to_XYZ(pixel);

  // scotopic luminance V — guarded against tiny XYZ.x to avoid
  // dark-area "snow" amplification, same as the OpenCL kernel.
  const float denom = (XYZ.x > threshold) ? XYZ.x : threshold;
  const float V = clipf(c * XYZ.y * (1.33f * (1.0f + (XYZ.y + XYZ.z) / denom) - 1.68f));

  const float w = vk_lookup(lut, pixel.x / 100.0f);
  const float4 sw = (float4)(XYZ_sw_x, XYZ_sw_y, XYZ_sw_z, XYZ_sw_w);

  XYZ = w * XYZ + (1.0f - w) * V * sw;
  out[idx] = vk_XYZ_to_Lab(XYZ);
}
