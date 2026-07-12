/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute twin of blendop.cl::blendop_set_mask — fill the
    single-channel blend-mask plane with a constant opacity. Used by
    the uniform-mask GPU blend path (dt_develop_blend_process_vk).
*/

kernel void blendop_set_mask(global float *mask,
                             const int width,
                             const int height,
                             const float value)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  mask[y * width + x] = value;
}
