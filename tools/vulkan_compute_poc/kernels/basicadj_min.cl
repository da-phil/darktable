/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute proof-of-concept kernel.

    A pared-down version of basicadj's exposure/black-point step, written
    against the OpenCL C 1.2 subset that clspv accepts. This kernel is
    intentionally limited to:

      - global float buffers (no image2d_t),
      - scalar push-constant-shaped uniforms,
      - no atomics, no barriers, no async copies.

    Once clspv has compiled this into SPIR-V (kernel entry point per
    `kernel void` function), the Vulkan host loads the SPIR-V module and
    dispatches it via a single VkPipeline + VkDescriptorSet. The same
    SPIR-V module is consumed unchanged on Linux/Windows native Vulkan and
    on macOS via MoltenVK.

    Reference: data/kernels/basicadj.cl (production OpenCL version).
*/

kernel void basicadj_min(global const float4 *in,
                         global float4 *out,
                         const int width,
                         const int height,
                         const float black_point,
                         const float scale)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  float4 pixel = in[idx];

  // Subtract black point and apply linear exposure scale on RGB. Keep
  // the alpha channel as-is so the result still rounds-trip in dt's
  // 4-channel host buffers.
  pixel.x = (pixel.x - black_point) * scale;
  pixel.y = (pixel.y - black_point) * scale;
  pixel.z = (pixel.z - black_point) * scale;

  out[idx] = pixel;
}
