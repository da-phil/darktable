// Vulkan port of extended.cl :: colorchecker.
//
// Thin-plate-spline colour correction in Lab. Each output pixel is an
// affine term (poly) plus a sum of radial-basis weights over the source
// patches. Already buffer-friendly in OpenCL — image2d in/out become
// storage buffers and the patch data is one float4 storage buffer:
//   [ source_Lab x num_patches | coeff_Lab x (num_patches + 4) ]
// with coeff_Lab[0..num_patches) the per-patch weights and the trailing
// 4 entries (poly_Lab) the affine term.
//
// Bindings (3 storage buffers): 0 in, 1 out, 2 params.
// Push constants: 3 ints = 12 bytes.

#include "dt_vulkan_common.h"

// Fast approximate log2 / log, byte-for-byte from extended.cl. The
// union bit-pun matches dt_vulkan_common.h::vk_atomic_add_f (clspv-safe).
static inline float cc_fastlog2(float x)
{
  union { float f; unsigned int i; } vx = { x };
  union { unsigned int i; float f; } mx = { (vx.i & 0x007FFFFFu) | 0x3f000000u };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;
  return y - 124.22551499f
    - 1.498030302f * mx.f
    - 1.72587999f / (0.3520887068f + mx.f);
}

static inline float cc_fastlog(float x)
{
  return 0.69314718f * cc_fastlog2(x);
}

static inline float cc_thinplate(const float4 x, const float4 y)
{
  const float r2 = (x.x - y.x) * (x.x - y.x)
                 + (x.y - y.y) * (x.y - y.y)
                 + (x.z - y.z) * (x.z - y.z);
  return r2 * cc_fastlog(fmax(1e-8f, r2));
}

kernel void colorchecker(global const float4 *in,
                         global       float4 *out,
                         global const float4 *params,
                         const int width,
                         const int height,
                         const int num_patches)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  global const float4 *source_Lab = params;
  global const float4 *coeff_Lab  = params + num_patches;
  global const float4 *poly_Lab   = params + 2 * num_patches;

  const float4 ipixel = in[idx];
  const float w = ipixel.w;

  float4 opixel = poly_Lab[0] + poly_Lab[1] * ipixel.x
                + poly_Lab[2] * ipixel.y + poly_Lab[3] * ipixel.z;

  for(int k = 0; k < num_patches; k++)
  {
    const float phi = cc_thinplate(ipixel, source_Lab[k]);
    opixel += coeff_Lab[k] * phi;
  }

  opixel.w = w;
  out[idx] = opixel;
}
