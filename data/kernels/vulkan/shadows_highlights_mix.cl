// Vulkan port of gaussian.cl :: shadows_highlights_mix.
//
// Final stage of the shadows/highlights module: takes the original
// Lab pixel + a low-pass-filtered mask (Gaussian or bilateral,
// upstream), inverts and desaturates the mask, then composites two
// soft-light overlays (highlights with -opacity, shadows with
// +opacity). Mirrors the OpenCL kernel body 1:1.
//
// Bindings (3 storage buffers):
//   0: in    (float4) — original Lab pixels
//   1: mask  (float4) — blurred Lab pixels (L channel feeds the mask)
//   2: out   (float4)
//
// Push constants: 4 ints + 7 floats = 44 bytes.
// flags encodes the per-channel unbound bits (L/A/B for shadows and
// highlights), matching SHADHI_UNBOUND_* in src/iop/shadhi.c.

#include "dt_vulkan_common.h"

#define VK_SHADHI_UNBOUND_L              1
#define VK_SHADHI_UNBOUND_A              2
#define VK_SHADHI_UNBOUND_B              4
#define VK_SHADHI_UNBOUND_SHADOWS_L      VK_SHADHI_UNBOUND_L
#define VK_SHADHI_UNBOUND_SHADOWS_A      VK_SHADHI_UNBOUND_A
#define VK_SHADHI_UNBOUND_SHADOWS_B      VK_SHADHI_UNBOUND_B
#define VK_SHADHI_UNBOUND_HIGHLIGHTS_L   (VK_SHADHI_UNBOUND_L << 3)
#define VK_SHADHI_UNBOUND_HIGHLIGHTS_A   (VK_SHADHI_UNBOUND_A << 3)
#define VK_SHADHI_UNBOUND_HIGHLIGHTS_B   (VK_SHADHI_UNBOUND_B << 3)

// Soft-light overlay used by both halves. Same maths as the
// OpenCL helper in gaussian.cl::overlay — kept inline so the
// kernel stays self-contained (clspv handles inline-static fine).
static inline float4 vk_overlay(const float4 in_a, const float4 in_b,
                                const float opacity, const float transform,
                                const float ccorrect,
                                const int ub_x, const int ub_y, const int ub_z, const int ub_w,
                                const float low_approximation)
{
  const float4 scale = (float4)(100.0f, 128.0f, 128.0f, 1.0f);
  const float lmin = 0.0f;
  const float lmax = 1.0f;
  const float halfmax = 0.5f;
  const float doublemax = 2.0f;

  float4 a = in_a / scale;
  float4 b = in_b / scale;
  float opacity2 = opacity * opacity;

  while(opacity2 > 0.0f)
  {
    const float la = ub_x ? a.x : clamp(a.x, lmin, lmax);
    float lb = (b.x - halfmax) * sign(opacity) * sign(lmax - la) + halfmax;
    lb = ub_w ? lb : clamp(lb, lmin, lmax);
    const float lref = copysign(fabs(la) > low_approximation
                                  ? 1.0f / fabs(la)
                                  : 1.0f / low_approximation, la);
    const float href = copysign(fabs(1.0f - la) > low_approximation
                                  ? 1.0f / fabs(1.0f - la)
                                  : 1.0f / low_approximation, 1.0f - la);

    const float chunk = opacity2 > 1.0f ? 1.0f : opacity2;
    const float optrans = chunk * transform;
    opacity2 -= 1.0f;

    a.x = la * (1.0f - optrans)
        + (la > halfmax
              ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb)
              : doublemax * la * lb) * optrans;
    a.x = ub_x ? a.x : clamp(a.x, lmin, lmax);

    const float blend = a.x * lref * ccorrect + (1.0f - a.x) * href * (1.0f - ccorrect);
    a.y = a.y * (1.0f - optrans) + (a.y + b.y) * blend * optrans;
    a.y = ub_y ? a.y : clamp(a.y, -1.0f, 1.0f);
    a.z = a.z * (1.0f - optrans) + (a.z + b.z) * blend * optrans;
    a.z = ub_z ? a.z : clamp(a.z, -1.0f, 1.0f);
  }
  return a * scale;
}

kernel void shadows_highlights_mix(global const float4 *in,
                                   global const float4 *mask,
                                   global       float4 *out,
                                   const int   width,
                                   const int   height,
                                   const int   flags,
                                   const int   unbound_mask,
                                   const float shadows,
                                   const float highlights,
                                   const float compress,
                                   const float shadows_ccorrect,
                                   const float highlights_ccorrect,
                                   const float low_approximation,
                                   const float whitepoint)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 io = in[idx];
  const float w = io.w;
  float4 m = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  // Mask reads L only — chroma channels are zeroed; mask is the
  // inverse of the blurred L.
  m.x = 100.0f - mask[idx].x;

  // White-point pre-scale (skip negatives — preserves NaN-free
  // behaviour against the OpenCL kernel).
  io.x = io.x > 0.0f ? io.x / whitepoint : io.x;
  m.x  = m.x  > 0.0f ? m.x  / whitepoint : m.x;

  // Highlights overlay (negative opacity = darken bright areas).
  {
    const float xform = clipf(1.0f - 0.01f * m.x / (1.0f - compress));
    io = vk_overlay(io, m, -highlights, xform, 1.0f - highlights_ccorrect,
                    flags & VK_SHADHI_UNBOUND_HIGHLIGHTS_L,
                    flags & VK_SHADHI_UNBOUND_HIGHLIGHTS_A,
                    flags & VK_SHADHI_UNBOUND_HIGHLIGHTS_B,
                    unbound_mask, low_approximation);
  }
  // Shadows overlay (positive opacity = lighten dark areas).
  {
    const float xform = clipf(0.01f * m.x / (1.0f - compress) - compress / (1.0f - compress));
    io = vk_overlay(io, m, shadows, xform, shadows_ccorrect,
                    flags & VK_SHADHI_UNBOUND_SHADOWS_L,
                    flags & VK_SHADHI_UNBOUND_SHADOWS_A,
                    flags & VK_SHADHI_UNBOUND_SHADOWS_B,
                    unbound_mask, low_approximation);
  }

  io.w = w;
  out[idx] = io;
}
