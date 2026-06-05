// Vulkan port of colorequal.cl :: bilinear4.
// Bilinear resample of a float4 buffer from (width_in, height_in) to
// (width_out, height_out). 2 bindings (in float4, out float4),
// PC: width_in, height_in, width_out, height_out = 16 B.
#include "dt_vulkan_common.h"

kernel void ce_bilinear4(global const float4 *in,
                         global float4 *out,
                         const int width_in,
                         const int height_in,
                         const int width_out,
                         const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const float x_in = ((float)x / (float)width_out)  * (float)width_in;
  const float y_in = ((float)y / (float)height_out) * (float)height_in;

  int x_prev = (int)floor(x_in);
  int x_next = x_prev + 1;
  int y_prev = (int)floor(y_in);
  int y_next = y_prev + 1;
  x_prev = (x_prev < width_in)  ? x_prev : width_in  - 1;
  x_next = (x_next < width_in)  ? x_next : width_in  - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float4 Q_NW = in[y_prev * width_in + x_prev];
  const float4 Q_NE = in[y_prev * width_in + x_next];
  const float4 Q_SE = in[y_next * width_in + x_next];
  const float4 Q_SW = in[y_next * width_in + x_prev];

  const float Dy_next = (float)y_next - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = (float)x_next - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[y * width_out + x] = Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
                         + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}
