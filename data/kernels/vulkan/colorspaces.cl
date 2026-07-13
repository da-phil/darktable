// Vulkan port of colorspaces.cl :: colorspaces_transform_rgb_matrix_to_rgb.
//
// Applies a working-space → output-space RGB transform: optional
// input TRC LUT (if profile_info_from->nonlinearlut), 3×3 matrix
// product, optional output TRC LUT (if profile_info_to->nonlinearlut).
//
// Used by the host-side helper dt_ioppr_transform_image_colorspace_rgb_vk
// to remove CPU round-trips in modules that consume two different
// profiles (e.g. overexposed: working profile → histogram profile).
//
// Binding layout (6 storage buffers):
//   0: in     (float4)  — input pixels in the FROM working space
//   1: out    (float4)  — output pixels in the TO working space
//   2: profile_info_from  — vk_dt_colorspaces_iccprofile_info_t
//   3: lut_from           — 6·lutsize tone-curve LUT (FROM)
//   4: profile_info_to    — vk_dt_colorspaces_iccprofile_info_t
//   5: lut_to             — 6·lutsize tone-curve LUT (TO)
// Push constants: 2 ints + 9 floats (combined matrix) = 44 bytes.
//
// The host pre-combines profile_info_to->matrix_out * profile_info_from->matrix_in
// into the 9-float matrix passed via push constants so the kernel
// only has to do one matmul.

#include "dt_vulkan_common.h"

kernel void colorspaces_transform_rgb_matrix_to_rgb
  (global const float4 *in,
   global       float4 *out,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info_from,
   global const float  *lut_from,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info_to,
   global const float  *lut_to,
   const int   width,
   const int   height,
   const float m0, const float m1, const float m2,
   const float m3, const float m4, const float m5,
   const float m6, const float m7, const float m8)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  if(profile_info_from->nonlinearlut)
    pixel = vk_apply_trc_in(pixel, profile_info_from, lut_from);

  // matrix product: row-major 3x3 stored in 9 floats via push constants.
  const float4 r = (float4)(m0 * pixel.x + m1 * pixel.y + m2 * pixel.z,
                            m3 * pixel.x + m4 * pixel.y + m5 * pixel.z,
                            m6 * pixel.x + m7 * pixel.y + m8 * pixel.z,
                            pixel.w);

  out[idx] = profile_info_to->nonlinearlut
              ? vk_apply_trc_out(r, profile_info_to, lut_to)
              : r;
}

// Vulkan port of colorspaces.cl :: colorspaces_transform_rgb_matrix_to_lab.
// Working-space RGB -> Lab: optional input TRC LUT, matrix_in (RGB->XYZ),
// XYZ->Lab. Mirrors the CL kernel used by
// dt_ioppr_transform_image_colorspace{,_cl} on the module-input and
// blend-space transform hops. The matrix lives in the profile struct
// (matrix_in[9], row-major) rather than push constants because this
// transform uses exactly one profile's own matrix.
//
// Binding layout (4 storage buffers):
//   0: in     (float4)  — RGB working-space pixels
//   1: out    (float4)  — Lab pixels
//   2: profile_info     — vk_dt_colorspaces_iccprofile_info_t
//   3: lut              — 6·lutsize tone-curve LUT
// Push constants: 2 ints (width, height) = 8 bytes.
kernel void colorspaces_transform_rgb_matrix_to_lab
  (global const float4 *in,
   global       float4 *out,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
   global const float  *lut,
   const int width,
   const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  if(profile_info->nonlinearlut)
    pixel = vk_apply_trc_in(pixel, profile_info, lut);

  float4 xyz;
  xyz.x = profile_info->matrix_in[0] * pixel.x + profile_info->matrix_in[1] * pixel.y
        + profile_info->matrix_in[2] * pixel.z;
  xyz.y = profile_info->matrix_in[3] * pixel.x + profile_info->matrix_in[4] * pixel.y
        + profile_info->matrix_in[5] * pixel.z;
  xyz.z = profile_info->matrix_in[6] * pixel.x + profile_info->matrix_in[7] * pixel.y
        + profile_info->matrix_in[8] * pixel.z;
  xyz.w = pixel.w;

  out[idx] = vk_XYZ_to_Lab(xyz);
}

// Vulkan port of colorspaces.cl :: colorspaces_transform_lab_to_rgb_matrix.
// Lab -> working-space RGB: Lab->XYZ, matrix_out (XYZ->RGB), optional
// output TRC LUT. Same binding layout as rgb_matrix_to_lab above.
kernel void colorspaces_transform_lab_to_rgb_matrix
  (global const float4 *in,
   global       float4 *out,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
   global const float  *lut,
   const int width,
   const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 xyz = vk_Lab_to_XYZ(in[idx]);

  float4 pixel;
  pixel.x = profile_info->matrix_out[0] * xyz.x + profile_info->matrix_out[1] * xyz.y
          + profile_info->matrix_out[2] * xyz.z;
  pixel.y = profile_info->matrix_out[3] * xyz.x + profile_info->matrix_out[4] * xyz.y
          + profile_info->matrix_out[5] * xyz.z;
  pixel.z = profile_info->matrix_out[6] * xyz.x + profile_info->matrix_out[7] * xyz.y
          + profile_info->matrix_out[8] * xyz.z;
  pixel.w = xyz.w;

  out[idx] = profile_info->nonlinearlut
              ? vk_apply_trc_out(pixel, profile_info, lut)
              : pixel;
}
