// Vulkan port of colorreconstruction.cl :: colorreconstruction_slice.
//
// Trilinear lookup into the bilateral grid + chroma blend back into
// the output. The lookup is hand-coded against the flat float buffer
// (the OpenCL kernel does the same — it doesn't use a 3D texture).
// The OpenCL sampler-clamp on `in` becomes an explicit bounds check
// via `if(x >= width || y >= height) return`.
//
// Bindings (3 storage buffers):
//   0: in    (float4, original input)
//   1: out   (float4, dehazed output)
//   2: grid  (float, 4 * size_x * size_y * size_z)
// Push constants: 48 B
//   (width, height, sizex, sizey, sizez,
//    sigma_s, sigma_r, threshold,
//    bx, by, roix, roiy, scale).

#include "dt_vulkan_common.h"

#ifndef CR_CLIPF
#define CR_CLIPF(a) clamp((a), 0.0f, 1.0f)
#endif

kernel void colorreconstruction_slice(global const float4 *in,
                                      global       float4 *out,
                                      global const float  *grid,
                                      const int   width,
                                      const int   height,
                                      const int   sizex,
                                      const int   sizey,
                                      const int   sizez,
                                      const float sigma_s,
                                      const float sigma_r,
                                      const float threshold,
                                      const int   bx,
                                      const int   by,
                                      const int   roix,
                                      const int   roiy,
                                      const float scale)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const int ox = 1;
  const int oy = sizex;
  const int oz = sizey * sizex;

  float4 pixel = in[idx];
  const float blend = CR_CLIPF(20.0f / threshold * pixel.x - 19.0f);

  // grid_rescale: (roi + p) * scale - b
  const float pxy_x = (float)(roix + x) * scale - (float)bx;
  const float pxy_y = (float)(roiy + y) * scale - (float)by;

  // image_to_grid
  const float gpx = clamp(pxy_x / sigma_s,    0.0f, (float)(sizex - 1));
  const float gpy = clamp(pxy_y / sigma_s,    0.0f, (float)(sizey - 1));
  const float gpz = clamp(pixel.x / sigma_r,  0.0f, (float)(sizez - 1));

  const int gix = min(sizex - 2, (int)gpx);
  const int giy = min(sizey - 2, (int)gpy);
  const int giz = min(sizez - 2, (int)gpz);
  const float fx = gpx - (float)gix;
  const float fy = gpy - (float)giy;
  const float fz = gpz - (float)giz;

  const int gi = gix + sizex * (giy + sizey * giz);

  const float4 c000 = (float4)(grid[4*(gi)         + 0], grid[4*(gi)         + 1], grid[4*(gi)         + 2], grid[4*(gi)         + 3]);
  const float4 c100 = (float4)(grid[4*(gi+ox)      + 0], grid[4*(gi+ox)      + 1], grid[4*(gi+ox)      + 2], grid[4*(gi+ox)      + 3]);
  const float4 c010 = (float4)(grid[4*(gi+oy)      + 0], grid[4*(gi+oy)      + 1], grid[4*(gi+oy)      + 2], grid[4*(gi+oy)      + 3]);
  const float4 c110 = (float4)(grid[4*(gi+ox+oy)   + 0], grid[4*(gi+ox+oy)   + 1], grid[4*(gi+ox+oy)   + 2], grid[4*(gi+ox+oy)   + 3]);
  const float4 c001 = (float4)(grid[4*(gi+oz)      + 0], grid[4*(gi+oz)      + 1], grid[4*(gi+oz)      + 2], grid[4*(gi+oz)      + 3]);
  const float4 c101 = (float4)(grid[4*(gi+ox+oz)   + 0], grid[4*(gi+ox+oz)   + 1], grid[4*(gi+ox+oz)   + 2], grid[4*(gi+ox+oz)   + 3]);
  const float4 c011 = (float4)(grid[4*(gi+oy+oz)   + 0], grid[4*(gi+oy+oz)   + 1], grid[4*(gi+oy+oz)   + 2], grid[4*(gi+oy+oz)   + 3]);
  const float4 c111 = (float4)(grid[4*(gi+ox+oy+oz)+ 0], grid[4*(gi+ox+oy+oz)+ 1], grid[4*(gi+ox+oy+oz)+ 2], grid[4*(gi+ox+oy+oz)+ 3]);

  const float4 opixel =
        c000 * (1.0f - fx) * (1.0f - fy) * (1.0f - fz)
      + c100 * (       fx) * (1.0f - fy) * (1.0f - fz)
      + c010 * (1.0f - fx) * (       fy) * (1.0f - fz)
      + c110 * (       fx) * (       fy) * (1.0f - fz)
      + c001 * (1.0f - fx) * (1.0f - fy) * (       fz)
      + c101 * (       fx) * (1.0f - fy) * (       fz)
      + c011 * (1.0f - fx) * (       fy) * (       fz)
      + c111 * (       fx) * (       fy) * (       fz);

  const float opixelx = fmax(opixel.x, 0.01f);
  pixel.y = (opixel.w > 0.0f) ? pixel.y * (1.0f - blend) + opixel.y * pixel.x / opixelx * blend : pixel.y;
  pixel.z = (opixel.w > 0.0f) ? pixel.z * (1.0f - blend) + opixel.z * pixel.x / opixelx * blend : pixel.z;

  out[idx] = pixel;
}
