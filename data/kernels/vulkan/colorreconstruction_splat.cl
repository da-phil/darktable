// Vulkan port of colorreconstruction.cl :: colorreconstruction_splat.
//
// Splats each input pixel's weighted Lab into the 3D bilateral grid
// via atomic-add. The OpenCL kernel does a workgroup-local
// reduction first; the VK twin skips that and goes straight to
// atomics (same pattern §5.13 bilateral_splat uses). The math is
// bit-equal: pixels with pixel.x >= threshold contribute zero in
// OpenCL, so we skip them outright.
//
// Bindings (2 storage buffers):
//   0: in   (float4, input image)
//   1: grid (float, 4*size_x*size_y*size_z)
// Push constants: 44 B
//   (width, height, sizex, sizey, sizez,
//    sigma_s, sigma_r, threshold,
//    precedence, hue, hue_sigma).

#include "dt_vulkan_common.h"

#define CR_PRECEDENCE_NONE   0
#define CR_PRECEDENCE_CHROMA 1
#define CR_PRECEDENCE_HUE    2

kernel void colorreconstruction_splat(global const float4 *in,
                                      global       float  *grid,
                                      const int   width,
                                      const int   height,
                                      const int   sizex,
                                      const int   sizey,
                                      const int   sizez,
                                      const float sigma_s,
                                      const float sigma_r,
                                      const float threshold,
                                      const int   precedence,
                                      const float hue,
                                      const float hue_sigma)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 pixel = in[idx2d(x, y, width)];

  // Pixels at or above threshold contribute 0 in the OpenCL kernel.
  if(pixel.x >= threshold) return;

  float weight;
  if(precedence == CR_PRECEDENCE_CHROMA)
  {
    weight = sqrt(pixel.y * pixel.y + pixel.z * pixel.z);
  }
  else if(precedence == CR_PRECEDENCE_HUE)
  {
    float m = atan2(pixel.z, pixel.y) - hue;
    m = m > M_PI_F ? m - DT_2PI_F : (m < -M_PI_F ? m + DT_2PI_F : m);
    weight = exp(-m * m / hue_sigma);
  }
  else
  {
    weight = 1.0f;
  }

  // image_to_grid: per-axis clamp into [0, size-1].
  const float gpx = clamp((float)x / sigma_s, 0.0f, (float)(sizex - 1));
  const float gpy = clamp((float)y / sigma_s, 0.0f, (float)(sizey - 1));
  const float gpz = clamp(pixel.x / sigma_r, 0.0f, (float)(sizez - 1));

  // Closest-integer splatting. Use floor(x + 0.5f) for explicit
  // half-up rounding — matches OpenCL `round()` semantics exactly
  // (GLSL's round() is implementation-defined for .5 values, and
  // can pick banker's rounding to even).
  const int xi = clamp((int)floor(gpx + 0.5f), 0, sizex - 1);
  const int yi = clamp((int)floor(gpy + 0.5f), 0, sizey - 1);
  const int zi = clamp((int)floor(gpz + 0.5f), 0, sizez - 1);
  const int gi = xi + sizex * (yi + sizey * zi);

  vk_atomic_add_f(grid + 4 * gi + 0, weight * pixel.x);
  vk_atomic_add_f(grid + 4 * gi + 1, weight * pixel.y);
  vk_atomic_add_f(grid + 4 * gi + 2, weight * pixel.z);
  vk_atomic_add_f(grid + 4 * gi + 3, weight);
}
