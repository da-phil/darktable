// Vulkan port of lut3d.cl :: lut3d_trilinear.
//
// 3D LUT lookup with trilinear interpolation. Maps the input RGB
// triplet (clamped to [0, 1]) into the LUT's level^3 grid, computes
// the 8 surrounding corner indices, and trilinearly blends the 8
// corner LUT values. Alpha passes through.
//
// Image-shortcut port: the OpenCL kernel reads `in` and writes `out`
// as image2d_t with fixed integer coords — both translate to flat
// float4 storage buffers. The LUT is already a flat float array
// (packed as 3 floats per LUT entry — no float3 alignment in storage
// buffers, same as the OpenCL global pointer).
//
// Binding layout (3 storage buffers):
//   0: in    (float4)
//   1: out   (float4)
//   2: clut  (float, 3 * level^3 entries)
// Push constants: 3 ints = 12 bytes (width, height, level).

#include "dt_vulkan_common.h"

kernel void lut3d_trilinear(global const float4 *in,
                            global       float4 *out,
                            global const float  *clut,
                            const int width,
                            const int height,
                            const int level)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  const int level2 = level * level;
  float4 input = clamp(in[idx], (float4)(0.0f), (float4)(1.0f));

  float4 rgbd = input * (float)(level - 1);
  int4   rgbi = clamp(convert_int4(rgbd), (int4)(0), (int4)(level - 2));
  rgbd -= convert_float4(rgbi);

  const int color = rgbi.x + rgbi.y * level + rgbi.z * level2;
  const int i000 = color * 3;
  const int i100 = i000 + 3;
  const int i010 = (color + level) * 3;
  const int i110 = i010 + 3;
  const int i001 = (color + level2) * 3;
  const int i101 = i001 + 3;
  const int i011 = (color + level + level2) * 3;
  const int i111 = i011 + 3;

  const float4 c000 = (float4)(clut[i000], clut[i000+1], clut[i000+2], 0.0f);
  const float4 c100 = (float4)(clut[i100], clut[i100+1], clut[i100+2], 0.0f);
  const float4 c010 = (float4)(clut[i010], clut[i010+1], clut[i010+2], 0.0f);
  const float4 c110 = (float4)(clut[i110], clut[i110+1], clut[i110+2], 0.0f);
  const float4 c001 = (float4)(clut[i001], clut[i001+1], clut[i001+2], 0.0f);
  const float4 c101 = (float4)(clut[i101], clut[i101+1], clut[i101+2], 0.0f);
  const float4 c011 = (float4)(clut[i011], clut[i011+1], clut[i011+2], 0.0f);
  const float4 c111 = (float4)(clut[i111], clut[i111+1], clut[i111+2], 0.0f);

  float4 tmp1 = c000 * (1.0f - rgbd.x) + c100 * rgbd.x;
  float4 tmp2 = c010 * (1.0f - rgbd.x) + c110 * rgbd.x;
  float4 output = tmp1 * (1.0f - rgbd.y) + tmp2 * rgbd.y;
  tmp1 = c001 * (1.0f - rgbd.x) + c101 * rgbd.x;
  tmp2 = c011 * (1.0f - rgbd.x) + c111 * rgbd.x;
  tmp1 = tmp1 * (1.0f - rgbd.y) + tmp2 * rgbd.y;
  output = output * (1.0f - rgbd.z) + tmp1 * rgbd.z;

  output.w = input.w;
  out[idx] = output;
}
