// Vulkan port of gaussian.cl :: lowpass_mix.
//
// Final stage of the censorize / lowpass pipelines: take a Lab
// pixel that's already been low-pass-filtered (Gaussian or
// bilateral, upstream), apply two cascaded unbounded curves to L
// (contrast and lightness), scale the a/b chroma by `saturation`,
// and optionally clamp to the Lab range.
//
// Bindings (4 storage buffers):
//   0: in (float4), 1: out (float4),
//   2: ctable (float, 65536 contrast curve),
//   3: ltable (float, 65536 lightness curve).
//
// Push constants: 3 ints + 7 floats = 40 bytes.

#include "dt_vulkan_common.h"

kernel void lowpass_mix(global const float4 *in,
                        global       float4 *out,
                        global const float  *ctable,
                        global const float  *ltable,
                        const int   width,
                        const int   height,
                        const int   unbound,
                        const float saturation,
                        const float ca0, const float ca1, const float ca2,
                        const float la0, const float la1, const float la2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 i = in[idx];
  float4 o;
  o.x = vk_lookup_unbounded(ctable, i.x / 100.0f, ca0, ca1, ca2);
  o.x = vk_lookup_unbounded(ltable, o.x / 100.0f, la0, la1, la2);
  if(unbound)
  {
    o.y = i.y * saturation;
    o.z = i.z * saturation;
  }
  else
  {
    o.y = clamp(i.y * saturation, -128.0f, 128.0f);
    o.z = clamp(i.z * saturation, -128.0f, 128.0f);
  }
  o.w = i.w;
  out[idx] = o;
}
