// Vulkan histogram-reduction kernel (GPU tap, DAG milestone M5,
// gpu_resident_pixelpipe_dag.md §5.4). Bins a float4 image into a
// bins_count·4 uint histogram with atomic increments, matching the
// CPU reducers in src/common/histogram.c (_bin_rgb / _bin_Lab /
// _bin_Lab_LCh) so a module's per-pixel histogram no longer forces a
// trunk-sized device->host readback: the tiny histogram buffer is
// read back instead (and, once the tap registry lands, deferred to
// the end-of-run fence).
//
// Scope of this first increment (like M0's blend subset): the three
// 4-channel-float binnings. RAW (uint16, 1 channel) and the
// middle-grey-compensated RGB path (needs the profile TRC) are not
// ported yet and keep the CPU reducer.
//
// Binding layout (2 storage buffers):
//   0: in    (float4)  — input pixels, roi_in-sized, row-major
//   1: hist  (uint)    — bins_count*4 counters, layout hist[bin*4 + k]
//                        for channels k = 0,1,2 (k = 3 unused), zeroed
//                        by the caller before dispatch
// Push constants: 8 ints = 32 bytes.
//   width, height             image dimensions
//   crop_x, crop_y            active-box top-left (inclusive)
//   crop_right, crop_bottom   active-box margins (as in dt_histogram_roi_t)
//   bins_count                e.g. 256
//   mode                      0 = RGB, 1 = Lab, 2 = Lab->LCh

#include "dt_vulkan_common.h"

kernel void histogram_rgb
  (global const float4 *in,
   global       uint   *hist,
   const int width,
   const int height,
   const int crop_x,
   const int crop_y,
   const int crop_right,
   const int crop_bottom,
   const int bins_count,
   const int mode)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  // active box [crop_x, width-crop_right) x [crop_y, height-crop_bottom)
  if(x < crop_x || x >= width - crop_right) return;
  if(y < crop_y || y >= height - crop_bottom) return;

  const float max_bin = (float)(bins_count - 1);
  const float4 px = in[idx2d(x, y, width)];

  float4 v;
  if(mode == 0) // RGB
  {
    v = px * max_bin;
  }
  else if(mode == 1) // Lab
  {
    const float4 scale = (float4)(max_bin / 100.0f, max_bin / 256.0f,
                                  max_bin / 256.0f, 0.0f);
    const float4 shift = (float4)(0.0f, 128.0f, 128.0f, 0.0f);
    v = scale * (px + shift);
  }
  else // Lab -> LCh
  {
    const float4 lch = vk_Lab_2_LCH(px);
    const float4 scale = (float4)(max_bin / 100.0f,
                                  max_bin / (128.0f * 1.4142135623730951f),
                                  max_bin, 0.0f);
    v = scale * lch;
  }

  // clamp to [0, max_bin] and truncate, exactly as _clamp_bin's
  // CLAMP(vals, 0, max_bin) into an integer bin
  const int b0 = (int)clamp(v.x, 0.0f, max_bin);
  const int b1 = (int)clamp(v.y, 0.0f, max_bin);
  const int b2 = (int)clamp(v.z, 0.0f, max_bin);

  atomic_add(&hist[b0 * 4 + 0], 1u);
  atomic_add(&hist[b1 * 4 + 1], 1u);
  atomic_add(&hist[b2 * 4 + 2], 1u);
}
