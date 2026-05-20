// Vulkan port of filmic.cl :: filmic.
//
// The legacy single-pass filmic transform (the original `filmic`
// IOP, kept around as `filmicrgb`'s predecessor). One pixel in, one
// pixel out, two 65536-entry shaper LUTs, all scalars in push
// constants. The newer wavelet-based `filmicrgb` is a separate port
// pending §8.5 (image2d / sampler bindings).
//
// Math mirrors basic.cl::filmic byte-for-byte:
//   Lab -> XYZ -> ProPhotoRGB
//   global saturation desat against XYZ luminance
//   per-channel or norm-preserving log -> S-curve lookup
//   derivative lookup -> selective desat around the luminance
//   power gamma
//   ProPhotoRGB -> Lab
//
// Bindings (4 storage buffers):
//   0: in    (float4, Lab)
//   1: out   (float4, Lab)
//   2: table (float, 65536 entries — S-curve)
//   3: diff  (float, 65536 entries — derivative)
//
// Push constants: 2 ints + 6 floats + 1 int = 36 bytes.

#include "dt_vulkan_common.h"

kernel void filmic(global const float4 *in,
                   global       float4 *out,
                   global const float  *table,
                   global const float  *diff,
                   const int   width,
                   const int   height,
                   const float dynamic_range,
                   const float shadows_range,
                   const float grey,
                   const float contrast,
                   const float power,
                   const float saturation,
                   const int   preserve_color)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 i = in[idx];
  const float4 xyz = vk_Lab_to_XYZ(i);
  float4 o = vk_XYZ_to_prophotorgb(xyz);

  const float noise = pow(2.0f, -16.0f);
  float derivative, luma;

  // Global desaturation against XYZ luminance.
  if(saturation != 1.0f)
  {
    const float lum = xyz.y;
    o.x = lum + saturation * (o.x - lum);
    o.y = lum + saturation * (o.y - lum);
    o.z = lum + saturation * (o.z - lum);
  }

  if(preserve_color)
  {
    // Save the ratios.
    float maxRGB = max(o.x, max(o.y, o.z));
    const float inv_max = (maxRGB > 0.0f) ? (1.0f / maxRGB) : 0.0f;
    const float rr = o.x * inv_max;
    const float rg = o.y * inv_max;
    const float rb = o.z * inv_max;

    // Log profile.
    maxRGB = maxRGB / grey;
    maxRGB = (maxRGB < noise) ? noise : maxRGB;
    maxRGB = (log2(maxRGB) - shadows_range) / dynamic_range;
    maxRGB = clamp(maxRGB, 0.0f, 1.0f);
    const float index = maxRGB;

    // S-curve.
    maxRGB = vk_lookup(table, maxRGB);

    // Re-apply ratios.
    o.x = maxRGB * rr;
    o.y = maxRGB * rg;
    o.z = maxRGB * rb;

    derivative = vk_lookup(diff, index);
    luma = maxRGB;
  }
  else
  {
    // Per-channel log profile.
    o.x = (o.x / grey < noise) ? noise : (o.x / grey);
    o.y = (o.y / grey < noise) ? noise : (o.y / grey);
    o.z = (o.z / grey < noise) ? noise : (o.z / grey);
    o.x = clamp((log2(o.x) - shadows_range) / dynamic_range, 0.0f, 1.0f);
    o.y = clamp((log2(o.y) - shadows_range) / dynamic_range, 0.0f, 1.0f);
    o.z = clamp((log2(o.z) - shadows_range) / dynamic_range, 0.0f, 1.0f);

    const float index = vk_prophotorgb_to_XYZ(o).y;

    // S-curve.
    o.x = vk_lookup(table, o.x);
    o.y = vk_lookup(table, o.y);
    o.z = vk_lookup(table, o.z);

    derivative = vk_lookup(diff, index);
    luma = vk_prophotorgb_to_XYZ(o).y;
  }

  // Selective desat around luminance.
  o.x = luma + derivative * (o.x - luma);
  o.y = luma + derivative * (o.y - luma);
  o.z = luma + derivative * (o.z - luma);
  o.x = clamp(o.x, 0.0f, 1.0f);
  o.y = clamp(o.y, 0.0f, 1.0f);
  o.z = clamp(o.z, 0.0f, 1.0f);

  // Display gamma.
  o.x = pow(o.x, power);
  o.y = pow(o.y, power);
  o.z = pow(o.z, power);

  // Back to Lab; preserve original alpha.
  const float4 lab_out = vk_prophotorgb_to_Lab(o);
  i.x = lab_out.x;
  i.y = lab_out.y;
  i.z = lab_out.z;
  out[idx] = i;
}
