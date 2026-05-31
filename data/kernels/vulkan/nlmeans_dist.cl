// Vulkan port of nlmeans.cl :: nlmeans_dist.
//
// Per-pixel squared (L/C-weighted) colour distance between pixel
// (x,y) and the q-shifted pixel (x+qx, y+qy). Out-of-bounds shifts
// contribute 0 (same multiply-by-mask trick as the OpenCL kernel —
// the shifted index is forced to 0 and the result zeroed).
//
// Bindings (2 storage buffers):
//   0: in  (float4, input image)
//   1: U4  (float, the per-pixel distance)
// Push constants: 24 B (width, height, qx, qy, nL2, nC2).

#include "dt_vulkan_common.h"

kernel void nlmeans_dist(global const float4 *in,
                         global       float  *U4,
                         const int   width,
                         const int   height,
                         const int   qx,
                         const int   qy,
                         const float nL2,
                         const float nC2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int gidx = idx2d(x, y, width);

  const float4 norm2 = (float4)(nL2, nC2, nC2, 1.0f);

  int xpq = x + qx;
  int ypq = y + qy;
  const int inb = (x + qx < width && x + qx >= 0 && y + qy < height && y + qy >= 0) ? 1 : 0;
  // force the shifted index to (0,0) when out of bounds, then mask the result.
  xpq *= (x + qx < width && x + qx >= 0) ? 1 : 0;
  ypq *= (y + qy < height && y + qy >= 0) ? 1 : 0;

  const float4 p1 = in[gidx];
  const float4 p2 = in[idx2d(xpq, ypq, width)];
  const float4 tmp = (p1 - p2) * (p1 - p2) * norm2;
  float dist = tmp.x + tmp.y + tmp.z;
  dist *= inb ? 1.0f : 0.0f;

  U4[gidx] = dist;
}
