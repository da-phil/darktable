// Vulkan port of colorequal.cl :: process_data.
//
// Computes per-pixel hue/saturation/brightness in dt-UCS HSB from the
// UV chromaticity + L* luminance, then looks up the per-hue
// correction LUTs (hue shift, saturation gain, brightness gain). When
// `guiding` is set it also computes the Scharr gradient of the
// saturation map into Lscharr (used by the guided-filter path).
//
// Binding layout (9 storage buffers):
//   0: uv             (float2)
//   1: Lscharr        (float, in/out — read as L*, optionally written
//                      with the gradient amplitude when guiding)
//   2: saturation     (float)
//   3: corrections    (float2 — .x hue shift, .y saturation gain)
//   4: b_corrections  (float — brightness gain)
//   5: pixout         (float4 — HSB written into .xyz)
//   6: LUT_saturation (float, LUT_ELEM)
//   7: LUT_hue        (float, LUT_ELEM)
//   8: LUT_brightness (float, LUT_ELEM)
// Push constants: 2 floats + 3 ints = 20 bytes
//   (white, gradient_amp, guiding, width, height).

#include "dt_vulkan_common.h"

#define VK_CE_NORM_MIN 1.52587890625e-05f

static inline float vk_ce_scharr_gradient(global const float *in, const int k, const int w)
{
  const float gx = 47.0f / 255.0f * (in[k-w-1] - in[k-w+1] + in[k+w-1] - in[k+w+1])
                + 162.0f / 255.0f * (in[k-1]   - in[k+1]);
  const float gy = 47.0f / 255.0f * (in[k-w-1] - in[k+w-1] + in[k-w+1] - in[k+w+1])
                + 162.0f / 255.0f * (in[k-w]   - in[k+w]);
  return sqrt(gx * gx + gy * gy);
}

kernel void ce_process_data(global const float2 *uv,
                            global       float  *Lscharr,
                            global const float  *saturation,
                            global       float2 *corrections,
                            global       float  *b_corrections,
                            global       float4 *pixout,
                            global const float  *LUT_saturation,
                            global const float  *LUT_hue,
                            global const float  *LUT_brightness,
                            const float white,
                            const float gradient_amp,
                            const int guiding,
                            const int width,
                            const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  const float4 JCH = vk_dt_UCS_LUV_to_JCH(Lscharr[k], white, uv[k]);
  const float4 HSB = vk_dt_UCS_JCH_to_HSB(JCH);

  const float hue = HSB.x;
  const float sat = HSB.y;
  pixout[k].x = hue;
  pixout[k].y = sat;
  pixout[k].z = HSB.z;

  if(guiding)
  {
    const int kk = clamp(row, 1, height - 2) * width + clamp(col, 1, width - 2);
    const float kscharr = fmax(0.0f, vk_ce_scharr_gradient(saturation, kk, width) - 0.02f);
    Lscharr[k] = gradient_amp * kscharr * kscharr;
  }

  if(JCH.y > VK_CE_NORM_MIN)
  {
    corrections[k].x  = vk_lookup_gamut(LUT_hue, hue);
    corrections[k].y  = vk_lookup_gamut(LUT_saturation, hue);
    b_corrections[k]  = sat * (vk_lookup_gamut(LUT_brightness, hue) - 1.0f);
  }
  else
  {
    corrections[k].x  = 0.0f;
    corrections[k].y  = 1.0f;
    b_corrections[k]  = 0.0f;
  }
}
