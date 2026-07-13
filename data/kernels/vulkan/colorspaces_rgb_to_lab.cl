// Single-entry-point sibling of colorspaces.cl ::
// colorspaces_transform_rgb_matrix_to_lab, so the glslang fallback
// (one entry per .spv) can expose it as its own module. clspv builds
// the same entry from colorspaces.cl; the runtime addresses whichever
// program name it loaded. See colorspaces.cl for design notes.
//
// Binding layout (4 storage buffers):
//   0: in     (float4)  — RGB working-space pixels
//   1: out    (float4)  — Lab pixels
//   2: profile_info     — vk_dt_colorspaces_iccprofile_info_t
//   3: lut              — 6·lutsize tone-curve LUT
// Push constants: 2 ints (width, height) = 8 bytes.

#include "dt_vulkan_common.h"

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
