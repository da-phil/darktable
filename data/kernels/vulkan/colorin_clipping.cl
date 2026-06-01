// Vulkan port of basic.cl :: colorin_clipping.
//
// Same shape as colorin_unbound but uses two matrices: cmat converts
// cam→RGB (clipped to [0, 1]), then lmat converts the clipped RGB to
// XYZ. The host computes both matrices from the input ICC profile.
// Used when the profile defines a non-RGB intermediate (`nrgb`).
//
// Binding layout (7 storage buffers):
//   0: in    (float4)
//   1: out   (float4)
//   2: lutr / 3: lutg / 4: lutb  (float, 65536 entries each)
//   5: cmat  (float, 9 — cam→RGB)
//   6: lmat  (float, 9 — RGB→XYZ)
// Push constants: 3 ints + 13 floats = 64 bytes (same as unbound).

#include "dt_vulkan_common.h"

kernel void colorin_clipping(global const float4 *in,
                             global       float4 *out,
                             global const float  *lutr,
                             global const float  *lutg,
                             global const float  *lutb,
                             global const float  *cmat,
                             global const float  *lmat,
                             const int width,
                             const int height,
                             const int blue_mapping,
                             const float corr_x, const float corr_y,
                             const float corr_z, const float corr_w,
                             const float a_r0, const float a_r1, const float a_r2,
                             const float a_g0, const float a_g1, const float a_g2,
                             const float a_b0, const float a_b1, const float a_b2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  const float4 corr = (float4)(corr_x, corr_y, corr_z, corr_w);
  float4 pixel = corr * in[idx];

  float cam[3];
  cam[0] = vk_lerp_lookup_unbounded(lutr, pixel.x, a_r0, a_r1, a_r2, 0, VK_LUT_SIZE);
  cam[1] = vk_lerp_lookup_unbounded(lutg, pixel.y, a_g0, a_g1, a_g2, 0, VK_LUT_SIZE);
  cam[2] = vk_lerp_lookup_unbounded(lutb, pixel.z, a_b0, a_b1, a_b2, 0, VK_LUT_SIZE);

  if(blue_mapping)
  {
    const float YY = cam[0] + cam[1] + cam[2];
    if(YY > 0.0f)
    {
      const float zz = cam[2] / YY;
      const float bound_z = 0.5f, bound_Y = 0.8f;
      const float amount = 0.11f;
      if(zz > bound_z)
      {
        const float t = (zz - bound_z) / (1.0f - bound_z)
                        * fmin(1.0f, YY / bound_Y);
        cam[1] += t * amount;
        cam[2] -= t * amount;
      }
    }
  }

  float RGB[3];
  for(int j = 0; j < 3; j++)
  {
    RGB[j] = 0.0f;
    for(int i = 0; i < 3; i++) RGB[j] += cmat[3 * j + i] * cam[i];
  }
  for(int i = 0; i < 3; i++) RGB[i] = clamp(RGB[i], 0.0f, 1.0f);

  float XYZ[3];
  for(int j = 0; j < 3; j++)
  {
    XYZ[j] = 0.0f;
    for(int i = 0; i < 3; i++) XYZ[j] += lmat[3 * j + i] * RGB[i];
  }
  float4 xyz = (float4)(XYZ[0], XYZ[1], XYZ[2], 0.0f);
  const float4 lab = vk_XYZ_to_Lab(xyz);
  pixel.x = lab.x; pixel.y = lab.y; pixel.z = lab.z;
  out[idx] = pixel;
}
