// Single-entry-point sibling of colorspaces.cl ::
// colorspaces_transform_lab_to_rgb_matrix, so the glslang fallback
// (one entry per .spv) can expose it as its own module. clspv builds
// the same entry from colorspaces.cl; the runtime addresses whichever
// program name it loaded. See colorspaces.cl for design notes.
//
// Binding layout (4 storage buffers):
//   0: in     (float4)  — Lab pixels
//   1: out    (float4)  — RGB working-space pixels
//   2: profile_info     — vk_dt_colorspaces_iccprofile_info_t
//   3: lut              — 6·lutsize tone-curve LUT
// Push constants: 2 ints (width, height) = 8 bytes.

#include "dt_vulkan_common.h"

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
