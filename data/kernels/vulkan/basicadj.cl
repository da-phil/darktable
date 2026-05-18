// Vulkan-targeted port of basicadj (basicadj.cl :: kernel basicadj).
//
// Exercise both arms of the ICC profile plumbing in §5.11 of the
// dev-doc:
//   - vk_get_rgb_matrix_luminance (the highlight compression branch)
//   - vk_dt_rgb_norm                (the preserve-colors branch)
// plus two per-piece LUTs (lut_gamma, lut_contrast) using the
// LUT-on-storage-buffer pattern from §5.8.
//
// Binding layout (6 storage buffers):
//   0: in   (float4)  — original pixels
//   1: out  (float4)  — destination
//   2: lut_gamma      — 65536-entry gamma curve
//   3: lut_contrast   — 65536-entry contrast curve
//   4: profile_info   — vk_dt_colorspaces_iccprofile_info_t (156 bytes)
//   5: profile_lut    — 6·lutsize tone-curve LUT
//
// Push constants: 8 ints + 10 floats = 72 bytes (see
// vk_basicadj_pc_t in src/iop/basicadj.c).

#include "dt_vulkan_common.h"

static inline float vk_basicadj_hlcurve(const float level,
                                        const float hlcomp, const float hlrange)
{
  if(hlcomp > 0.0f)
  {
    float val = level + (hlrange - 1.0f);
    if(val == 0.0f) val = 0.000001f;
    float Y = val / hlrange;
    Y *= hlcomp;
    if(Y <= -1.0f) Y = -0.999999f;
    const float R = hlrange / (val * hlcomp);
    return log1p(Y) * R;
  }
  return 1.0f;
}

// Two LUT lookups with linear-extrapolation tails — same shape as the
// OpenCL get_lut_gamma / get_lut_contrast helpers.
static inline float vk_basicadj_lut_gamma(const float x, const float gamma,
                                           global const float *lut)
{
  return (x > 1.0f) ? pow(x, gamma) : vk_lookup(lut, x);
}

static inline float vk_basicadj_lut_contrast(const float x, const float contrast,
                                              const float middle_grey, const float inv_middle_grey,
                                              global const float *lut)
{
  return (x > 1.0f) ? pow(x * inv_middle_grey, contrast) * middle_grey
                    : vk_lookup(lut, x);
}

kernel void basicadj(global const float4 *in,
                     global       float4 *out,
                     global const float  *lut_gamma,
                     global const float  *lut_contrast,
                     global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                     global const float  *profile_lut,
                     const int   width,
                     const int   height,
                     const int   process_gamma,
                     const int   plain_contrast,
                     const int   process_saturation_vibrance,
                     const int   process_hlcompr,
                     const int   preserve_colors,
                     const int   use_work_profile,
                     const float black_point,
                     const float scale,
                     const float gamma,
                     const float contrast,
                     const float saturation,
                     const float vibrance,
                     const float hlcomp,
                     const float hlrange,
                     const float middle_grey,
                     const float inv_middle_grey)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float w = pixel.w;

  // 1. exposure (black point + scale)
  pixel = (pixel - (float4)black_point) * (float4)scale;

  // 2. highlight compression
  if(process_hlcompr)
  {
    const float lum = (use_work_profile == 0)
                       ? vk_dt_camera_rgb_luminance(pixel)
                       : vk_get_rgb_matrix_luminance(pixel, profile_info, profile_lut);
    if(lum > 0.0f)
    {
      const float ratio = vk_basicadj_hlcurve(lum, hlcomp, hlrange);
      pixel *= (float4)ratio;
    }
  }

  // 3. gamma
  if(process_gamma)
  {
    if(pixel.x > 0.0f) pixel.x = vk_basicadj_lut_gamma(pixel.x, gamma, lut_gamma);
    if(pixel.y > 0.0f) pixel.y = vk_basicadj_lut_gamma(pixel.y, gamma, lut_gamma);
    if(pixel.z > 0.0f) pixel.z = vk_basicadj_lut_gamma(pixel.z, gamma, lut_gamma);
  }

  // 4. plain contrast (per-channel)
  if(plain_contrast)
  {
    if(pixel.x > 0.0f) pixel.x = vk_basicadj_lut_contrast(pixel.x, contrast, middle_grey, inv_middle_grey, lut_contrast);
    if(pixel.y > 0.0f) pixel.y = vk_basicadj_lut_contrast(pixel.y, contrast, middle_grey, inv_middle_grey, lut_contrast);
    if(pixel.z > 0.0f) pixel.z = vk_basicadj_lut_contrast(pixel.z, contrast, middle_grey, inv_middle_grey, lut_contrast);
  }

  // 5. contrast with preserve colors (norm-preserving)
  if(preserve_colors != VK_RGB_NORM_NONE)
  {
    float ratio = 1.0f;
    const float lum = vk_dt_rgb_norm(pixel, preserve_colors, use_work_profile,
                                       profile_info, profile_lut);
    if(lum > 0.0f)
    {
      const float contrast_lum = pow(lum / middle_grey, contrast) * middle_grey;
      ratio = contrast_lum / lum;
    }
    pixel *= (float4)ratio;
  }

  // 6. saturation + vibrance
  if(process_saturation_vibrance)
  {
    const float average = (pixel.x + pixel.y + pixel.z) / 3.0f;
    const float4 delta = pixel - (float4)average;
    const float dlen = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    const float P = vibrance * (1.0f - pow(dlen, fabs(vibrance)));
    pixel = (float4)average + ((float4)saturation + (float4)P) * delta;
  }

  pixel.w = w;
  out[idx] = pixel;
}
