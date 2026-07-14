// Vulkan color-picker reduction for the JzCzhz (scene-referred RGB)
// picker (GPU tap, DAG milestone M5, §5.4). Separate from picker.cl
// because it needs the working-profile buffers: it runs the full
// RGB → XYZ_D50 → XYZ_D65 → JzAzBz → JzCzhz chain per pixel before the
// min/max/mean reduction, matching rgb_to_JzCzhz + _color_picker_jzczhz
// in src/common/color_picker.c. Only the valid-matrix profile path is
// ported (the lcms/no-matrix picker keeps the CPU reducer).
//
// Binding layout (4 storage buffers):
//   0: in            (float4)  — RGB working-space pixels
//   1: stats         (float)   — 12 accumulators (sum/min/max), as picker.cl
//   2: profile_info  — vk_dt_colorspaces_iccprofile_info_t
//   3: lut           — 6·lutsize tone-curve LUT
// Push constants: 5 ints = 20 bytes. width, box_x0, box_y0, box_x1, box_y1.

#include "dt_vulkan_common.h"

kernel void picker_jzczhz
  (global const float4 *in,
   global       float  *stats,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
   global const float  *lut,
   const int width,
   const int box_x0,
   const int box_y0,
   const int box_x1,
   const int box_y1)
{
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int bw = box_x1 - box_x0;
  const int bh = box_y1 - box_y0;
  if(gx >= bw || gy >= bh) return;

  const int x = box_x0 + gx;
  const int y = box_y0 + gy;
  float4 rgb = in[idx2d(x, y, width)];

  if(profile_info->nonlinearlut)
    rgb = vk_apply_trc_in(rgb, profile_info, lut);

  // RGB -> XYZ_D50 via the profile's input matrix (row-major 3x3)
  float4 xyz;
  xyz.x = profile_info->matrix_in[0] * rgb.x + profile_info->matrix_in[1] * rgb.y
        + profile_info->matrix_in[2] * rgb.z;
  xyz.y = profile_info->matrix_in[3] * rgb.x + profile_info->matrix_in[4] * rgb.y
        + profile_info->matrix_in[5] * rgb.z;
  xyz.z = profile_info->matrix_in[6] * rgb.x + profile_info->matrix_in[7] * rgb.y
        + profile_info->matrix_in[8] * rgb.z;
  xyz.w = rgb.w;

  float4 px = vk_JzAzBz_to_JzCzhz(vk_XYZ_to_JzAzBz(vk_XYZ_D50_to_XYZ_D65(xyz)));
  // rotated 4th channel for hue wraparound, as _update_stats_4ch
  px.w = (px.z < 0.5f) ? px.z + 0.5f : px.z - 0.5f;

  vk_atomic_add_f(&stats[0], px.x);
  vk_atomic_add_f(&stats[1], px.y);
  vk_atomic_add_f(&stats[2], px.z);
  vk_atomic_add_f(&stats[3], px.w);

  vk_atomic_min_f(&stats[4], px.x);
  vk_atomic_min_f(&stats[5], px.y);
  vk_atomic_min_f(&stats[6], px.z);
  vk_atomic_min_f(&stats[7], px.w);

  vk_atomic_max_f(&stats[8],  px.x);
  vk_atomic_max_f(&stats[9],  px.y);
  vk_atomic_max_f(&stats[10], px.z);
  vk_atomic_max_f(&stats[11], px.w);
}
