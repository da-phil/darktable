// Vulkan-targeted port of overexposed (basic.cl :: kernel overexposed).
//
// Flags pixels that are out of gamut/luminance/saturation/RGB range and
// paints them the configured upper/lower marker colours. Mirrors the
// OpenCL kernel logic 1:1 — we just trade the read_only image2d_t LUT
// for the flat storage-buffer LUT layout described in
// dt_vulkan_common.h, and the profile_info constant buffer for a
// readonly storage buffer of the same struct.
//
// Binding layout (5 storage buffers):
//   0: in   (float4)  — original pixels
//   1: out  (float4)  — destination (may equal in if no clipping)
//   2: tmp  (float4)  — same image transformed into the histogram
//                       profile by the host before dispatch
//   3: profile_info   — vk_dt_colorspaces_iccprofile_info_t (156 bytes)
//   4: lut  (float)   — 6·lutsize tone-curve LUT
//
// Push constants: 4 ints + 10 floats = 56 bytes (all scalars; the
// upper/lower colours are passed component-wise to avoid the
// std430-vec4-alignment trap that bit channelmixerrgb).

#include "dt_vulkan_common.h"

// dt_clipping_preview_mode_t mirror (basic.cl line ~3281).
#define DT_CLIPPING_PREVIEW_GAMUT      0
#define DT_CLIPPING_PREVIEW_ANYRGB     1
#define DT_CLIPPING_PREVIEW_LUMINANCE  2
#define DT_CLIPPING_PREVIEW_SATURATION 3

kernel void overexposed(global const float4 *in,
                        global       float4 *out,
                        global const float4 *tmp,
                        global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                        global const float  *lut,
                        const int   width,
                        const int   height,
                        const int   mode,
                        const int   use_work_profile,
                        const float lower,
                        const float upper,
                        const float lower_r, const float lower_g,
                        const float lower_b, const float lower_w,
                        const float upper_r, const float upper_g,
                        const float upper_b, const float upper_w)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel      = in[idx];
  const float4 ptmp = tmp[idx];
  const float4 lo   = (float4)(lower_r, lower_g, lower_b, lower_w);
  const float4 hi   = (float4)(upper_r, upper_g, upper_b, upper_w);

  if(mode == DT_CLIPPING_PREVIEW_ANYRGB)
  {
    if(ptmp.x >= upper || ptmp.y >= upper || ptmp.z >= upper)
      pixel = (float4)(hi.x, hi.y, hi.z, pixel.w);
    else if(ptmp.x <= lower && ptmp.y <= lower && ptmp.z <= lower)
      pixel = (float4)(lo.x, lo.y, lo.z, pixel.w);
  }
  else if(mode == DT_CLIPPING_PREVIEW_GAMUT && use_work_profile)
  {
    const float luminance = vk_get_rgb_matrix_luminance(pixel, profile_info, lut);
    if(luminance >= upper)
      pixel = (float4)(hi.x, hi.y, hi.z, pixel.w);
    else if(luminance <= lower)
      pixel = (float4)(lo.x, lo.y, lo.z, pixel.w);
    else
    {
      float4 sat = ptmp - (float4)luminance;
      sat = sqrt(sat * sat / ((float4)(luminance * luminance) + ptmp * ptmp));
      if(sat.x > upper || sat.y > upper || sat.z > upper
         || ptmp.x >= upper || ptmp.y >= upper || ptmp.z >= upper)
        pixel = (float4)(hi.x, hi.y, hi.z, pixel.w);
      else if(ptmp.x <= lower && ptmp.y <= lower && ptmp.z <= lower)
        pixel = (float4)(lo.x, lo.y, lo.z, pixel.w);
    }
  }
  else if(mode == DT_CLIPPING_PREVIEW_LUMINANCE && use_work_profile)
  {
    const float luminance = vk_get_rgb_matrix_luminance(pixel, profile_info, lut);
    if(luminance >= upper)
      pixel = (float4)(hi.x, hi.y, hi.z, pixel.w);
    else if(luminance <= lower)
      pixel = (float4)(lo.x, lo.y, lo.z, pixel.w);
  }
  else if(mode == DT_CLIPPING_PREVIEW_SATURATION && use_work_profile)
  {
    const float luminance = vk_get_rgb_matrix_luminance(pixel, profile_info, lut);
    if(luminance < upper && luminance > lower)
    {
      float4 sat = ptmp - (float4)luminance;
      sat = sqrt(sat * sat / ((float4)(luminance * luminance) + ptmp * ptmp));
      if(sat.x > upper || sat.y > upper || sat.z > upper
         || ptmp.x >= upper || ptmp.y >= upper || ptmp.z >= upper)
        pixel = (float4)(hi.x, hi.y, hi.z, pixel.w);
      else if(ptmp.x <= lower && ptmp.y <= lower && ptmp.z <= lower)
        pixel = (float4)(lo.x, lo.y, lo.z, pixel.w);
    }
  }

  out[idx] = pixel;
}
