// Vulkan port of liquify.cl :: warp_kernel.
//
// Pixel-warp via a per-pixel float2 displacement `map` covering a
// rectangular `map_extent` inside the output ROI. For each output
// pixel:
//   - If the pixel is inside map_extent and the map's warp is
//     non-zero, do a 2a×2a Lanczos / bicubic / bilinear reconstruction
//     from the warped source position (using the discrete kernel `k`).
//   - Otherwise, pass through the input pixel at the same image
//     coordinates (the OpenCL flow does this as a pre-dispatch
//     image-copy; this Vulkan twin folds it into the warp kernel so
//     a single dispatch covers the whole roi_out).
//
// Image-shortcut port: the OpenCL kernel uses `sampleri`
// (CLK_FILTER_NEAREST + CLK_ADDRESS_CLAMP_TO_EDGE) for all reads at
// integer-valued coords (`in_pos = floor(in_pos); read at in_pos +
// sample_pos` where sample_pos is integer). So no actual sampler
// filtering — the Lanczos / bicubic / bilinear reconstruction is
// computed by the kernel itself via the host-prepared `k` table.
//
// Binding layout (4 storage buffers):
//   0: in   (float4, sized in_width * in_height)
//   1: out  (float4, sized out_width * out_height)
//   2: map  (float2, sized map_extent_w * map_extent_h)
//   3: k    (float, sized kdesc_size * kdesc_resolution + 1)
// Push constants: 14 ints = 56 bytes (see PC layout below).

#include "dt_vulkan_common.h"

// Mirrors `kmix` from liquify.cl: piecewise-linear lookup into the
// discrete kernel `k`, using the kdesc resolution to map the float
// argument `t` to a fractional bin.
static inline float vk_kmix(global const float *k,
                            const int kdesc_resolution,
                            float t)
{
  t = fabs(t * (float)kdesc_resolution);
  const float flor = floor(t);
  const float frac = t - flor;
  const int i = (int)flor;
  return mix(k[i], k[i + 1], frac);
}

kernel void liquify_warp(global const float4 *in,
                         global       float4 *out,
                         global const float2 *map,
                         global const float  *k,
                         const int in_width,    const int in_height,
                         const int out_width,   const int out_height,
                         const int roi_in_x,    const int roi_in_y,
                         const int roi_out_x,   const int roi_out_y,
                         const int map_extent_x, const int map_extent_y,
                         const int map_extent_w, const int map_extent_h,
                         const int kdesc_size,  const int kdesc_resolution)
{
  const int x = get_global_id(0);  // dispatched over roi_out
  const int y = get_global_id(1);
  if(x >= out_width || y >= out_height) return;

  // Image-space coordinates of this output pixel.
  const int img_x = roi_out_x + x;
  const int img_y = roi_out_y + y;

  // Map-buffer coordinates (relative to map_extent origin).
  const int mx = img_x - map_extent_x;
  const int my = img_y - map_extent_y;

  // Default: pass-through copy from the input at the matching image
  // coordinate. roi_in must contain roi_out's range for this to be
  // well-defined; clamp defensively to dodge OOB reads at the edges.
  const int src_x = clamp(img_x - roi_in_x, 0, in_width  - 1);
  const int src_y = clamp(img_y - roi_in_y, 0, in_height - 1);
  float4 result = in[src_y * in_width + src_x];

  if(mx >= 0 && mx < map_extent_w && my >= 0 && my < map_extent_h)
  {
    const float2 warp = map[my * map_extent_w + mx];
    if(warp.x != 0.0f || warp.y != 0.0f)
    {
      // Source position in roi_in buffer coords.
      float2 in_pos = (float2)((float)(img_x - roi_in_x) + warp.x,
                               (float)(img_y - roi_in_y) + warp.y);
      const int a = kdesc_size;

      // Build the per-axis Lanczos kernel weights, mirroring the OpenCL
      // kernel's `lkernel[6]; lk = lkernel + a - 1;` indexing trick.
      float lkx[6], lky[6];
      float norm_x = 0.0f, norm_y = 0.0f;
      const float fx = floor(in_pos.x);
      const float fy = floor(in_pos.y);
      for(int i = 1 - a; i <= a; i++)
      {
        const float wx = vk_kmix(k, kdesc_resolution, in_pos.x - fx - (float)i);
        const float wy = vk_kmix(k, kdesc_resolution, in_pos.y - fy - (float)i);
        lkx[i + a - 1] = wx;  norm_x += wx;
        lky[i + a - 1] = wy;  norm_y += wy;
      }

      const int base_x = (int)fx;
      const int base_y = (int)fy;
      float4 Sxy = (float4)(0.0f);
      for(int sy = 1 - a; sy <= a; sy++)
        for(int sx = 1 - a; sx <= a; sx++)
        {
          const int ix = clamp(base_x + sx, 0, in_width  - 1);
          const int iy = clamp(base_y + sy, 0, in_height - 1);
          Sxy += in[iy * in_width + ix]
               * lkx[sx + a - 1] * lky[sy + a - 1];
        }
      result = Sxy / (norm_x * norm_y);
    }
  }

  out[y * out_width + x] = result;
}
