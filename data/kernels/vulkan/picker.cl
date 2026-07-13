// Vulkan color-picker reduction kernel (GPU tap, DAG milestone M5,
// gpu_resident_pixelpipe_dag.md §5.4). Reduces the picker box to
// per-channel sum / min / max with float atomics, matching the CPU
// reducer _color_picker_rgb_or_lab in src/common/color_picker.c (the
// common "most iop pickers and the global picker" path, where the
// image colorspace already equals the picker colorspace so no
// conversion is applied). The host divides the sum by the box pixel
// count to get the mean, exactly as _color_picker_work_4ch does.
//
// Scope of this first increment (like M0's blend subset and the
// histogram tap): the no-conversion case, all four channels. The
// LCH / HSL / JzCzhz conversion pickers and the optional denoise blur
// are later increments and keep the CPU reducer.
//
// Binding layout (2 storage buffers):
//   0: in     (float4)  — input pixels, row-major, `width` wide
//   1: stats  (float)   — 12 accumulators, initialised by the caller:
//                         [0..3]  sum  = 0
//                         [4..7]  min  = +FLT_MAX
//                         [8..11] max  = -FLT_MAX
// Push constants: 6 ints = 24 bytes. width, box_x0, box_y0, box_x1,
// box_y1 — the half-open box [x0,x1) x [y0,y1) — and mode:
//   0 = raw 4 channels (no conversion, _color_picker_rgb_or_lab)
//   1 = Lab -> LCH  (_color_picker_lch)
//   2 = RGB -> HSL  (_color_picker_hsl)
// For the converting modes the 4th channel is a rotated copy of the
// 3rd, exactly as the CPU _update_stats_4ch does, so hue min/max/mean
// avoid the 0/1 wraparound. The JzCzhz picker (needs the profile) and
// the denoise blur are not ported and refuse to the CPU path.
//
// Dispatch over the box extent; global id (gx,gy) maps to pixel
// (box_x0+gx, box_y0+gy).

#include "dt_vulkan_common.h"

kernel void picker_rgb
  (global const float4 *in,
   global       float  *stats,
   const int width,
   const int box_x0,
   const int box_y0,
   const int box_x1,
   const int box_y1,
   const int mode)
{
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int bw = box_x1 - box_x0;
  const int bh = box_y1 - box_y0;
  if(gx >= bw || gy >= bh) return;

  const int x = box_x0 + gx;
  const int y = box_y0 + gy;
  float4 px = in[idx2d(x, y, width)];

  if(mode == 1)      px = vk_Lab_2_LCH(px);
  else if(mode == 2) px = vk_RGB_to_HSL(px);
  if(mode != 0)
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
