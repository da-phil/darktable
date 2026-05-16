/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute version of channelmixerrgb (color calibration).

    SCOPE: This is a simplified port of the linear-Bradford adaptation
    path, which is the most common adaptation mode in practice. The
    full kernel in data/kernels/channelmixer.cl has many additional
    paths (CAT16, full non-linear Bradford, XYZ, plus gamut mapping
    and luma/chroma adjustments) which are out of scope for this
    initial integration. Modules dispatching to this kernel are
    expected to verify the adaptation mode and fall back to OpenCL or
    CPU when it isn't DT_ADAPTATION_LINEAR_BRADFORD.

    Operates on a flat RGBA float buffer. The host (src/iop/
    channelmixerrgb.c::process_vk) pre-multiplies the input/output
    profile matrices into a single 9-float "RGB -> LMS -> RGB" matrix
    and passes it via push constants.
*/

kernel void channelmixerrgb_bradford_linear(global const float4 *in,
                                            global float4 *out,
                                            const int width,
                                            const int height,
                                            constant const float *MIX, // 12 floats: 3x4 matrix
                                            const float saturation,
                                            const float lightness)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  float4 pixel = in[idx];

  // 3x3 matrix * vec3 (column-major in the 12-float blob; columns are
  // packed as float4s with the 4th element ignored — matches what the
  // host writes in process_vk).
  const float r = MIX[0] * pixel.x + MIX[1] * pixel.y + MIX[2]  * pixel.z;
  const float g = MIX[4] * pixel.x + MIX[5] * pixel.y + MIX[6]  * pixel.z;
  const float b = MIX[8] * pixel.x + MIX[9] * pixel.y + MIX[10] * pixel.z;

  pixel.x = r;
  pixel.y = g;
  pixel.z = b;

  // Cheap saturation/lightness post-pass against per-channel mean.
  if(saturation != 0.0f || lightness != 0.0f)
  {
    const float mean = (pixel.x + pixel.y + pixel.z) * (1.0f / 3.0f);
    pixel.x = mean + (pixel.x - mean) * (1.0f + saturation);
    pixel.y = mean + (pixel.y - mean) * (1.0f + saturation);
    pixel.z = mean + (pixel.z - mean) * (1.0f + saturation);
    pixel.x += lightness;
    pixel.y += lightness;
    pixel.z += lightness;
  }

  out[idx] = pixel;
}
