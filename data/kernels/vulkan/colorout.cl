// Vulkan port of basic.cl :: colorout (matrix + shaper-LUT fast path).
//
// Lab → XYZ → linear RGB via a 3×3 matrix, followed by a per-channel
// shaper LUT with linear-extrapolation tails (the lookup_unbounded
// pattern). Mirrors the OpenCL fast path used when the output profile
// has a valid colour-matrix; the slow lcms2 path stays on CPU.
//
// The DT_COLORSPACE_LAB pass-through case is handled in host code via
// dt_vulkan_copy_device_to_device — this kernel is only reached when a
// matrix transform actually runs.
//
// Bindings (5 storage buffers):
//   0: in  (float4, Lab)
//   1: out (float4, RGB)
//   2: lut_r (float, 65536 entries)
//   3: lut_g (float, 65536 entries)
//   4: lut_b (float, 65536 entries)
//
// Push constants: 2 ints + 18 floats = 80 bytes
//   width, height,
//   m00..m22  (9 floats: 3×3 matrix in row-major order),
//   ar0,ar1,ar2  (R-channel extrapolation: out-of-range scale/exp),
//   ag0,ag1,ag2  (G channel),
//   ab0,ab1,ab2  (B channel).

#include "dt_vulkan_common.h"

kernel void colorout(global const float4 *in,
                     global       float4 *out,
                     global const float  *lut_r,
                     global const float  *lut_g,
                     global const float  *lut_b,
                     const int   width,
                     const int   height,
                     const float m00, const float m01, const float m02,
                     const float m10, const float m11, const float m12,
                     const float m20, const float m21, const float m22,
                     const float ar0, const float ar1, const float ar2,
                     const float ag0, const float ag1, const float ag2,
                     const float ab0, const float ab1, const float ab2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 pixel = in[idx];
  const float4 xyz = vk_Lab_to_XYZ(pixel);

  const float r = m00 * xyz.x + m01 * xyz.y + m02 * xyz.z;
  const float g = m10 * xyz.x + m11 * xyz.y + m12 * xyz.z;
  const float b = m20 * xyz.x + m21 * xyz.y + m22 * xyz.z;

  float4 outp;
  outp.x = vk_lerp_lookup_unbounded(lut_r, r, ar0, ar1, ar2, 0, VK_LUT_SIZE);
  outp.y = vk_lerp_lookup_unbounded(lut_g, g, ag0, ag1, ag2, 0, VK_LUT_SIZE);
  outp.z = vk_lerp_lookup_unbounded(lut_b, b, ab0, ab1, ab2, 0, VK_LUT_SIZE);
  outp.w = pixel.w;
  out[idx] = outp;
}
