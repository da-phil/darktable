// Vulkan port of extended.cl::channelmixer.
//
// Legacy 3-channel mixer with 4 operation modes:
//   - RGB:    9-coef RGB matrix only
//   - GRAY:   first row of RGB matrix collapses to single channel
//   - HSL_V1: HSL channel mixing (additive offsets), then RGB matrix
//   - HSL_V2: HSL channel mixing (replacement), then RGB matrix
//
// Bindings (4 storage buffers):
//   0: in           (float4 RGBA)
//   1: out          (float4 RGBA)
//   2: hsl_matrix   (9 floats)
//   3: rgb_matrix   (9 floats)
//
// Push constants: 12 B (width, height, operation_mode).

#include "dt_vulkan_common.h"

#define OPERATION_MODE_RGB    0
#define OPERATION_MODE_GRAY   1
#define OPERATION_MODE_HSL_V1 2
#define OPERATION_MODE_HSL_V2 3

kernel void channelmixer(global const float4 *in,
                         global       float4 *out,
                         global const float  *hsl_matrix,
                         global const float  *rgb_matrix,
                         const int width,
                         const int height,
                         const int operation_mode)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  float4 opixel = (float4)(0.0f, 0.0f, 0.0f, pixel.w);
  float gray, hmix, smix, lmix;

  switch(operation_mode)
  {
    case OPERATION_MODE_RGB:
      opixel.x = fmax(pixel.x * rgb_matrix[0] + pixel.y * rgb_matrix[1] + pixel.z * rgb_matrix[2], 0.0f);
      opixel.y = fmax(pixel.x * rgb_matrix[3] + pixel.y * rgb_matrix[4] + pixel.z * rgb_matrix[5], 0.0f);
      opixel.z = fmax(pixel.x * rgb_matrix[6] + pixel.y * rgb_matrix[7] + pixel.z * rgb_matrix[8], 0.0f);
      break;

    case OPERATION_MODE_GRAY:
      gray = fmax(pixel.x * rgb_matrix[0] + pixel.y * rgb_matrix[1] + pixel.z * rgb_matrix[2], 0.0f);
      opixel = (float4)(gray, gray, gray, pixel.w);
      break;

    case OPERATION_MODE_HSL_V1:
      hmix = clipf(pixel.x * hsl_matrix[0]) + pixel.y * hsl_matrix[1] + pixel.z * hsl_matrix[2];
      smix = clipf(pixel.x * hsl_matrix[3]) + pixel.y * hsl_matrix[4] + pixel.z * hsl_matrix[5];
      lmix = clipf(pixel.x * hsl_matrix[6]) + pixel.y * hsl_matrix[7] + pixel.z * hsl_matrix[8];

      if(hmix != 0.0f || smix != 0.0f || lmix != 0.0f)
      {
        float4 hsl = vk_RGB_to_HSL(pixel);
        hsl.x = (hmix != 0.0f) ? hmix : hsl.x;
        hsl.y = (smix != 0.0f) ? smix : hsl.y;
        hsl.z = (lmix != 0.0f) ? lmix : hsl.z;
        pixel = vk_HSL_to_RGB(hsl);
      }

      opixel.x = clipf(pixel.x * rgb_matrix[0] + pixel.y * rgb_matrix[1] + pixel.z * rgb_matrix[2]);
      opixel.y = clipf(pixel.x * rgb_matrix[3] + pixel.y * rgb_matrix[4] + pixel.z * rgb_matrix[5]);
      opixel.z = clipf(pixel.x * rgb_matrix[6] + pixel.y * rgb_matrix[7] + pixel.z * rgb_matrix[8]);
      break;

    case OPERATION_MODE_HSL_V2:
      hmix = clipf(pixel.x * hsl_matrix[0] + pixel.y * hsl_matrix[1] + pixel.z * hsl_matrix[2]);
      smix = clipf(pixel.x * hsl_matrix[3] + pixel.y * hsl_matrix[4] + pixel.z * hsl_matrix[5]);
      lmix = clipf(pixel.x * hsl_matrix[6] + pixel.y * hsl_matrix[7] + pixel.z * hsl_matrix[8]);
      if(hmix != 0.0f || smix != 0.0f || lmix != 0.0f)
      {
        pixel = (float4)(clipf(pixel.x), clipf(pixel.y), clipf(pixel.z), pixel.w);
        float4 hsl = vk_RGB_to_HSL(pixel);
        hsl.x = (hmix != 0.0f) ? hmix : hsl.x;
        hsl.y = (smix != 0.0f) ? smix : hsl.y;
        hsl.z = (lmix != 0.0f) ? lmix : hsl.z;
        pixel = vk_HSL_to_RGB(hsl);
      }
      opixel.x = fmax(pixel.x * rgb_matrix[0] + pixel.y * rgb_matrix[1] + pixel.z * rgb_matrix[2], 0.0f);
      opixel.y = fmax(pixel.x * rgb_matrix[3] + pixel.y * rgb_matrix[4] + pixel.z * rgb_matrix[5], 0.0f);
      opixel.z = fmax(pixel.x * rgb_matrix[6] + pixel.y * rgb_matrix[7] + pixel.z * rgb_matrix[8], 0.0f);
      break;
  }

  out[idx] = opixel;
}
