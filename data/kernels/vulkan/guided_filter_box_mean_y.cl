// Vulkan port of guided_filter.cl :: guided_filter_box_mean_y.
//
// Vertical companion to guided_filter_box_mean_x. 1D dispatch: one
// work item per image column.
//
// Bindings (2 storage buffers):
//   0: in   (float)
//   1: out  (float)
// Push constants: 12 B (width, height, w).

#include "dt_vulkan_common.h"

kernel void guided_filter_box_mean_y(global const float *in,
                                     global       float *out,
                                     const int width,
                                     const int height,
                                     const int w)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  float m = 0.f, n_box = 0.f, c = 0.f;
  if(height > 2 * w)
  {
    for(int i = 0, i_end = w + 1; i < i_end; i++)
    {
      vk_kahan_sum(m, c, in[idx2d(x, i, width)]);
      n_box += 1.f;
    }
    for(int i = 0, i_end = w; i < i_end; i++)
    {
      out[idx2d(x, i, width)] = m / n_box;
      vk_kahan_sum(m, c, in[idx2d(x, i + w + 1, width)]);
      n_box += 1.f;
    }
    for(int i = w, i_end = height - w - 1; i < i_end; i++)
    {
      out[idx2d(x, i, width)] = m / n_box;
      vk_kahan_sum(m, c, in[idx2d(x, i + w + 1, width)]);
      vk_kahan_sum(m, c, -in[idx2d(x, i - w, width)]);
    }
    for(int i = height - w - 1, i_end = height; i < i_end; i++)
    {
      out[idx2d(x, i, width)] = m / n_box;
      vk_kahan_sum(m, c, -in[idx2d(x, i - w, width)]);
      n_box -= 1.f;
    }
  }
  else
  {
    for(int i = 0, i_end = min(w + 1, height); i < i_end; i++)
    {
      vk_kahan_sum(m, c, in[idx2d(x, i, width)]);
      n_box += 1.f;
    }
    for(int i = 0; i < height; i++)
    {
      out[idx2d(x, i, width)] = m / n_box;
      if(i - w >= 0)
      {
        vk_kahan_sum(m, c, -in[idx2d(x, i - w, width)]);
        n_box -= 1.f;
      }
      if(i + w + 1 < height)
      {
        vk_kahan_sum(m, c, in[idx2d(x, i + w + 1, width)]);
        n_box += 1.f;
      }
    }
  }
}
