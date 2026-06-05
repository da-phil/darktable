// Vulkan port of colorequal.cl :: sample_input.
//
// Entry kernel: reads input float4, computes per-pixel saturation
// (delta/dmax), luminance (Y_to_UCS_L_star), and UV chromaticity
// (xyY_to_UCS_UV). Stages: input → XYZ_D65 (via matrix) → xyY →
// UCS L*UV. Image-shortcut port: OpenCL uses samplerA at integer
// coords on `dev_in` → flat float4 storage buffer.
//
// Binding layout (5 storage buffers):
//   0: in          (float4)
//   1: saturation  (float)
//   2: lum         (float, the L_star "Lscharr" buffer)
//   3: uv          (float2)
//   4: pix_out     (float4, only .w is written here)
//   5: mat         (float, 12 entries — input_matrix flat)
// Push constants: 2 ints = 8 bytes (width, height).

#include "dt_vulkan_common.h"

#define VK_CE_NORM_MIN 1.52587890625e-05f

kernel void ce_sample_input(global const float4 *in,
                            global       float  *saturation,
                            global       float  *lum,
                            global       float2 *uv,
                            global       float4 *pix_out,
                            global const float  *mat,
                            const int width,
                            const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  const float4 pix_in = in[k];
  const float dmin  = fmin(pix_in.x, fmin(pix_in.y, pix_in.z));
  const float dmax  = fmax(pix_in.x, fmax(pix_in.y, pix_in.z));
  const float delta = dmax - dmin;
  saturation[k] = (dmax > VK_CE_NORM_MIN && delta > VK_CE_NORM_MIN)
                  ? delta / dmax : 0.0f;

  const float4 XYZ_D65 = vk_matrix_dot(pix_in, mat);
  const float4 xyY     = vk_dt_D65_XYZ_to_xyY(XYZ_D65);
  lum[k] = vk_Y_to_dt_UCS_L_star(xyY.z);
  uv[k]  = vk_xyY_to_dt_UCS_UV(xyY);
  pix_out[k].w = pix_in.w;
}
