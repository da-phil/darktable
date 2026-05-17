/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::primaries — applies a 3x4 row-major
    matrix (padded) to RGB, leaves alpha alone.
*/

#include "dt_vulkan_common.h"

kernel void primaries(global const float4 *in,
                      global float4 *out,
                      const int width,
                      const int height,
                      const float m00, const float m01, const float m02, const float m03,
                      const float m10, const float m11, const float m12, const float m13,
                      const float m20, const float m21, const float m22, const float m23)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  const float4 i = in[idx];

  float4 o;
  o.x = m00 * i.x + m01 * i.y + m02 * i.z;
  o.y = m10 * i.x + m11 * i.y + m12 * i.z;
  o.z = m20 * i.x + m21 * i.y + m22 * i.z;
  o.w = i.w;
  out[idx] = o;
}
