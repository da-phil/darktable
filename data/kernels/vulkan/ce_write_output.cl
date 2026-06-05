// Vulkan port of colorequal.cl :: write_output.
//
// Final stage: applies the per-pixel corrections (hue shift on .x,
// saturation gain on .y, brightness gain on .z) to the cached HSB
// pixel, gamut-maps the saturation, converts dt-UCS HSB → XYZ_D65 →
// output RGB via the output matrix, and writes the result.
//
// Image-shortcut port: OpenCL writes `dev_out` at integer coords →
// flat float4 storage buffer.
//
// Binding layout (5 storage buffers):
//   0: out           (float4)
//   1: pixout        (float4, HSB from process_data)
//   2: corrections   (float2)
//   3: b_corrections (float)
//   4: mat           (float, 12 — output_matrix flat)
//   5: gamut_LUT     (float, LUT_ELEM)
// Push constants: 1 float + 2 ints = 12 bytes (white, width, height).

#include "dt_vulkan_common.h"

#define VK_CE_SAT_EFFECT 2.0f
#define VK_CE_BRIGHT_EFFECT 8.0f

kernel void ce_write_output(global       float4 *out,
                            global       float4 *pixout,
                            global const float2 *corrections,
                            global const float  *b_corrections,
                            global const float  *mat,
                            global const float  *gamut_LUT,
                            const float white,
                            const int width,
                            const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  float4 hsb = pixout[k];
  hsb.x += corrections[k].x;
  hsb.y = fmax(0.0f, hsb.y * (1.0f + VK_CE_SAT_EFFECT * (corrections[k].y - 1.0f)));
  hsb.z = fmax(0.0f, hsb.z * (1.0f + VK_CE_BRIGHT_EFFECT * b_corrections[k]));

  hsb.y = vk_gamut_map_HSB(hsb, gamut_LUT, white);
  const float4 XYZ_D65 = vk_dt_UCS_HSB_to_XYZ(hsb, white);
  const float4 pout = vk_matrix_dot(XYZ_D65, mat);
  out[k] = pout;
}
