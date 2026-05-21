// Vulkan port of sigmoid.cl::sigmoid_loglogistic_per_channel.
//
// Per-channel sigmoid tone mapper. Transforms pipe → base primaries
// (m_pb), desaturates negatives, transforms base → rendering primaries
// (m_br), applies the per-channel loglogistic sigmoid, then mixes
// the per-channel result with a hue-preserving correction guided by
// the channel min/mid/max ordering, finally transforms back via
// rendering → pipe (m_rp).
//
// Bindings (5 storage buffers): in, out, m_pb, m_br, m_rp
//   matrices are flat 9-float arrays (compact 3x3, no padding).
// Push constants: 32 B (2 ints + 6 floats).

#include "dt_vulkan_common.h"

static inline float4 _vk_mat3x3(const float4 v, global const float *m)
{
  const float R = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  const float G = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  const float B = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  return (float4)(R, G, B, v.w);
}

static inline float4 _vk_desat_neg(const float4 i)
{
  const float avg = fmax((i.x + i.y + i.z) / 3.0f, 0.0f);
  const float mn  = fmin(fmin(i.x, i.y), i.z);
  const float sf  = (mn < 0.0f) ? -avg / (mn - avg) : 1.0f;
  return avg + sf * (i - avg);
}

static inline float4 _vk_sigmoid_vec(float4 i, const float magnitude,
                                     const float paper_exp, const float film_fog,
                                     const float film_power, const float paper_power)
{
  i = fmax(i, 0.0f);
  i = pow(film_fog + i, film_power);
  i = magnitude * pow(i / (paper_exp + i), paper_power);
  return isnan(i.x) ? (float4)magnitude : i;
}

// Sort 3 components into (min, mid, max) indices. Returns ordering as
// uchar3 packed: (max_idx, mid_idx, min_idx). Mirrors the 7-case branch
// tree in the OpenCL helper byte-for-byte.
static inline int3 _vk_pixel_order(float r, float g, float b)
{
  if(r >= g)
  {
    if(g > b)        return (int3)(0, 1, 2);  // r ≥ g >  b
    else if(b > r)   return (int3)(2, 0, 1);  // b >  r ≥ g
    else if(b > g)   return (int3)(0, 2, 1);  // r ≥ b >  g
    else             return (int3)(0, 1, 2);  // r == g == b — pick anything
  }
  else
  {
    if(r >= b)       return (int3)(1, 0, 2);  // g >  r ≥ b
    else if(b > g)   return (int3)(2, 1, 0);  // b >  g >  r
    else             return (int3)(1, 2, 0);  // g ≥ b >  r
  }
}

static inline float _idx3(const float a, const float b, const float c, const int i)
{
  return (i == 0) ? a : ((i == 1) ? b : c);
}

// Linear hue interpolation toward the per-channel result that also
// preserves the sum of channels. Matches the OpenCL helper byte for byte.
static inline float4 _vk_preserve_hue_and_energy(float4 pix_in,
                                                 float4 per_channel,
                                                 const int3 order,  // (max, mid, min)
                                                 const float hue_preservation)
{
  const int   imax = order.x;
  const int   imid = order.y;
  const int   imin = order.z;

  const float pix_max = _idx3(pix_in.x, pix_in.y, pix_in.z, imax);
  const float pix_mid = _idx3(pix_in.x, pix_in.y, pix_in.z, imid);
  const float pix_min = _idx3(pix_in.x, pix_in.y, pix_in.z, imin);
  const float pc_max  = _idx3(per_channel.x, per_channel.y, per_channel.z, imax);
  const float pc_mid  = _idx3(per_channel.x, per_channel.y, per_channel.z, imid);
  const float pc_min  = _idx3(per_channel.x, per_channel.y, per_channel.z, imin);

  const float chroma   = pix_max - pix_min;
  const float midscale = (chroma != 0.0f) ? (pix_mid - pix_min) / chroma : 0.0f;
  const float full_hc  = pc_min + (pc_max - pc_min) * midscale;
  const float naive_mid = (1.0f - hue_preservation) * pc_mid + hue_preservation * full_hc;

  const float pc_energy   = per_channel.x + per_channel.y + per_channel.z;
  const float naive_energy = pc_min + naive_mid + pc_max;
  const float pix_minmid   = pix_min + pix_mid;
  const float blend = (pix_minmid != 0.0f) ? (2.0f * pix_min / pix_minmid) : 0.0f;
  const float energy_target = blend * pc_energy + (1.0f - blend) * naive_energy;

  float new_max, new_mid, new_min;
  if(naive_mid <= pc_mid)
  {
    const float corrected_mid =
      ((1.0f - hue_preservation) * pc_mid + hue_preservation *
       (midscale * pc_max + (1.0f - midscale) * (energy_target - pc_max))) /
      (1.0f + hue_preservation * (1.0f - midscale));
    new_min = energy_target - pc_max - corrected_mid;
    new_mid = corrected_mid;
    new_max = pc_max;
  }
  else
  {
    const float corrected_mid =
      ((1.0f - hue_preservation) * pc_mid + hue_preservation *
       (pc_min * (1.0f - midscale) + midscale * (energy_target - pc_min))) /
      (1.0f + hue_preservation * midscale);
    new_min = pc_min;
    new_mid = corrected_mid;
    new_max = energy_target - pc_min - corrected_mid;
  }

  float4 r = pix_in;
  // Scatter (new_min, new_mid, new_max) back into r.{x,y,z} by index.
  if(imin == 0) r.x = new_min; else if(imid == 0) r.x = new_mid; else r.x = new_max;
  if(imin == 1) r.y = new_min; else if(imid == 1) r.y = new_mid; else r.y = new_max;
  if(imin == 2) r.z = new_min; else if(imid == 2) r.z = new_mid; else r.z = new_max;
  return r;
}

kernel void sigmoid_loglogistic_per_channel(global const float4 *in,
                                            global       float4 *out,
                                            global const float  *m_pb,
                                            global const float  *m_br,
                                            global const float  *m_rp,
                                            const int   width,
                                            const int   height,
                                            const float white_target,
                                            const float paper_exp,
                                            const float film_fog,
                                            const float contrast_power,
                                            const float skew_power,
                                            const float hue_preservation)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 i = in[idx];
  const float alpha = i.w;

  i = _vk_mat3x3(i, m_pb);
  i = _vk_desat_neg(i);
  i = _vk_mat3x3(i, m_br);

  const float4 pix_in_rendering = i;
  const float4 per_channel = _vk_sigmoid_vec(i, white_target, paper_exp,
                                             film_fog, contrast_power, skew_power);

  const int3 order = _vk_pixel_order(pix_in_rendering.x,
                                     pix_in_rendering.y,
                                     pix_in_rendering.z);
  float4 mixed = _vk_preserve_hue_and_energy(pix_in_rendering, per_channel,
                                             order, hue_preservation);
  mixed.w = alpha;

  float4 o = _vk_mat3x3(mixed, m_rp);
  o.w = alpha;
  out[idx] = o;
}
