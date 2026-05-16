/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute version of diffuse-and-sharpen — SIMPLIFIED.

    SCOPE: The full CPU/OpenCL diffuse pipeline (see data/kernels/
    diffuse.cl and src/iop/diffuse.c::wavelets_process_cl) decomposes
    the image into multiple wavelet scales, applies the diffusion PDE
    per scale, and recomposes. Porting that full pipeline — including
    the à-trous wavelet transform, the per-scale low-freq/high-freq
    split, and the anisotropic diffusion step — is its own milestone.

    This kernel implements ONE pass of the simpler "sharpen" step:
    an unsharp-mask using a small 3x3 box blur as the low-pass.
    It exercises the storage-buffer dispatch and gives users of the
    Vulkan path a visible (though weaker) sharpen effect; the full
    diffuse algorithm needs further work and currently falls back to
    OpenCL/CPU at the host-side when full precision is required.
*/

static inline float4 read_clamped(global const float4 *in,
                                  const int x, const int y,
                                  const int width, const int height)
{
  const int cx = clamp(x, 0, width  - 1);
  const int cy = clamp(y, 0, height - 1);
  return in[cy * width + cx];
}

kernel void diffuse_sharpen_single(global const float4 *in,
                                   global float4 *out,
                                   const int width,
                                   const int height,
                                   const float amount,
                                   const float threshold)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  // 3x3 box blur (separability not exploited; kernel is tiny).
  float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for(int dy = -1; dy <= 1; ++dy)
    for(int dx = -1; dx <= 1; ++dx)
      sum += read_clamped(in, x + dx, y + dy, width, height);
  const float4 blurred = sum * (1.0f / 9.0f);

  const float4 pixel = in[idx];
  const float4 diff  = pixel - blurred;

  // Threshold gate: only sharpen edges, leave flat areas alone.
  const float mag = fabs(diff.x) + fabs(diff.y) + fabs(diff.z);
  const float gate = (mag > threshold) ? 1.0f : 0.0f;

  float4 result;
  result.x = pixel.x + gate * amount * diff.x;
  result.y = pixel.y + gate * amount * diff.y;
  result.z = pixel.z + gate * amount * diff.z;
  result.w = pixel.w;
  out[idx] = result;
}
