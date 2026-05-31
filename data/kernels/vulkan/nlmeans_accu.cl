// Vulkan port of nlmeans.cl :: nlmeans_accu.
//
// Accumulates the weighted neighbour pixels at +q and -q into the U2
// buffer. The `U2[gidx] += accu` is a plain read-modify-write — no
// atomics needed: each invocation owns its gidx and the host
// serialises the q-loop (one q dispatched at a time), so there is no
// cross-thread contention.
//
// Bindings (3 storage buffers):
//   0: in  (float4, input image)
//   1: U2  (float4, the accumulator — read-modify-written in place)
//   2: U4  (float, the patch weight for this q)
// Push constants: 16 B (width, height, qx, qy).

#include "dt_vulkan_common.h"

static inline float vk_ddirac(const int qx, const int qy)
{
  return ((qx != 0) || (qy != 0)) ? 1.0f : 0.0f;
}

kernel void nlmeans_accu(global const float4 *in,
                         global       float4 *U2,
                         global const float  *U4,
                         const int width,
                         const int height,
                         const int qx,
                         const int qy)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int gidx = idx2d(x, y, width);

  int wpq = 1;
  int wmq = 1;
  wpq *= (x + qx < width)  ? 1 : 0;
  wmq *= (x - qx < width)  ? 1 : 0;
  wpq *= (x + qx >= 0)     ? 1 : 0;
  wmq *= (x - qx >= 0)     ? 1 : 0;
  wpq *= (y + qy >= 0)     ? 1 : 0;
  wmq *= (y - qy >= 0)     ? 1 : 0;
  wpq *= (y + qy < height) ? 1 : 0;
  wmq *= (y - qy < height) ? 1 : 0;

  const float4 u1_pq = wpq ? in[idx2d(x + qx, y + qy, width)] : (float4)0.0f;
  const float4 u1_mq = wmq ? in[idx2d(x - qx, y - qy, width)] : (float4)0.0f;

  const float u4    = U4[gidx];
  const float u4_mq = U4[idx2d(clamp(x - qx, 0, width - 1), clamp(y - qy, 0, height - 1), width)];
  const float u4_mq_dd = u4_mq * vk_ddirac(qx, qy);

  float4 accu = (u4 * u1_pq) + (u4_mq_dd * u1_mq);
  accu.w = (float)wpq * u4 + (float)wmq * u4_mq_dd;

  U2[gidx] += accu;
}
