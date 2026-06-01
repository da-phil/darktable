// Vulkan port of extended.cl :: colormapping_mapping.
//
// Image-shortcut port: same sampler-less integer-coord trick as
// colormapping_histogram. The OpenCL kernel takes 3 image2d_t (in,
// tmp, out); the Vulkan port flattens these to float4 storage buffers.
//
// For each pixel: reads ipixel = Lab from in, dL from tmp.x (smoothed
// histogram-equalised lightness), writes opixel = (L_remapped,
// a_remapped, b_remapped, ipixel.w). The a/b remap is a soft-cluster
// weighted blend of (input − target_mean) * var_ratio + source_mean
// over MAXN clusters; weights come from `get_clusters` (inlined here).
//
// Binding layout (7 storage buffers):
//   0: in           (float4)
//   1: tmp          (float4)  — only .x is read (smoothed dL)
//   2: out          (float4)
//   3: target_mean  (float2*) — MAXN entries
//   4: source_mean  (float2*) — MAXN entries
//   5: var_ratio    (float2*) — MAXN entries
//   6: mapio        (int*)    — MAXN entries (target→source cluster map)
// Push constants: 3 ints = 12 bytes (width, height, clusters).
// MAXN = 5 (matches extended.cl's `#define MAXN`).

#include "dt_vulkan_common.h"

#define MAXN 5

kernel void colormapping_mapping(global const float4 *in,
                                 global const float4 *tmp,
                                 global       float4 *out,
                                 global const float2 *target_mean,
                                 global const float2 *source_mean,
                                 global const float2 *var_ratio,
                                 global const int    *mapio,
                                 const int width,
                                 const int height,
                                 const int clusters)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  float4 ipixel = in[idx];
  const float dL = tmp[idx].x;

  // Inlined get_clusters: soft cluster membership over (a, b) chroma.
  float weight[MAXN];
  float mdist = FLT_MAX;
  for(int k = 0; k < clusters; k++)
  {
    const float dx = ipixel.y - target_mean[k].x;
    const float dy = ipixel.z - target_mean[k].y;
    const float dist2 = dx * dx + dy * dy;
    weight[k] = dist2 > 1.0e-6f ? 1.0f / dist2 : -1.0f;
    if(dist2 < mdist) mdist = dist2;
  }
  if(mdist < 1.0e-6f)
    for(int k = 0; k < clusters; k++)
      weight[k] = weight[k] < 0.0f ? 1.0f : 0.0f;
  float sum = 0.0f;
  for(int k = 0; k < clusters; k++) sum += weight[k];
  if(sum > 0.0f)
    for(int k = 0; k < clusters; k++) weight[k] /= sum;

  float4 opixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  opixel.x = 2.0f * (dL - 50.0f) + ipixel.x;
  opixel.x = clamp(opixel.x, 0.0f, 100.0f);

  for(int c = 0; c < clusters; c++)
  {
    opixel.y += weight[c] * ((ipixel.y - target_mean[c].x) * var_ratio[c].x
                             + source_mean[mapio[c]].x);
    opixel.z += weight[c] * ((ipixel.z - target_mean[c].y) * var_ratio[c].y
                             + source_mean[mapio[c]].y);
  }
  opixel.w = ipixel.w;

  out[idx] = opixel;
}

#undef MAXN
