// Vulkan port of guided_filter.cl :: guided_filter_box_mean_x.
//
// Horizontal running-box mean (window radius w) of a single-channel
// float buffer, computed with a Kahan-compensated sliding sum exactly
// as the OpenCL kernel. 1D dispatch: one work item per image row.
//
// Bindings (2 storage buffers):
//   0: in   (float)
//   1: out  (float)
// Push constants: 12 B (width, height, w).

#include "dt_vulkan_common.h"

kernel void guided_filter_box_mean_x(global const float *in,
                                     global       float *out,
                                     const int width,
                                     const int height,
                                     const int w)
{
  const int y = get_global_id(0);
  if(y >= height) return;
  const int row = y * width;

  float m = 0.f, n_box = 0.f, c = 0.f;
  if(width > 2 * w)
  {
    for(int i = 0, i_end = w + 1; i < i_end; i++)
    {
      vk_kahan_sum(m, c, in[row + i]);
      n_box += 1.f;
    }
    for(int i = 0, i_end = w; i < i_end; i++)
    {
      out[row + i] = m / n_box;
      vk_kahan_sum(m, c, in[row + i + w + 1]);
      n_box += 1.f;
    }
    for(int i = w, i_end = width - w - 1; i < i_end; i++)
    {
      out[row + i] = m / n_box;
      vk_kahan_sum(m, c, in[row + i + w + 1]);
      vk_kahan_sum(m, c, -in[row + i - w]);
    }
    for(int i = width - w - 1, i_end = width; i < i_end; i++)
    {
      out[row + i] = m / n_box;
      vk_kahan_sum(m, c, -in[row + i - w]);
      n_box -= 1.f;
    }
  }
  else
  {
    for(int i = 0, i_end = min(w + 1, width); i < i_end; i++)
    {
      vk_kahan_sum(m, c, in[row + i]);
      n_box += 1.f;
    }
    for(int i = 0; i < width; i++)
    {
      out[row + i] = m / n_box;
      if(i - w >= 0)
      {
        vk_kahan_sum(m, c, -in[row + i - w]);
        n_box -= 1.f;
      }
      if(i + w + 1 < width)
      {
        vk_kahan_sum(m, c, in[row + i + w + 1]);
        n_box += 1.f;
      }
    }
  }
}
