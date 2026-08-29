/*
    This file is part of darktable,
    Copyright (C) 2025 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.h"

/* Bilinear interpolators on global buffers, GPU counterpart of
   interpolate_bilinear() in common/fast_guided_filter.h.

   Used to down- and upscale the grey working buffers of the guided filters,
   see dt_interpolate_bilinear_cl() in common/bilinear.h

   The channel count is carried by the vector type rather than by a loop
   counter, so that each node is fetched with a single vector load. */

typedef struct
{
  // linear offsets of the four surrounding nodes
  int nw, ne, se, sw;
  // spatial differences between the nodes
  float dx_prev, dx_next, dy_prev, dy_next;
} dt_bilinear_nodes_t;


static inline dt_bilinear_nodes_t _bilinear_nodes(const int x,
                                                  const int y,
                                                  const int width_in,
                                                  const int height_in,
                                                  const int width_out,
                                                  const int height_out)
{
  // Relative coordinates of the pixel in output space
  const float x_out = (float)x / (float)width_out;
  const float y_out = (float)y / (float)height_out;

  // Corresponding absolute coordinates of the pixel in input space
  const float x_in = x_out * (float)width_in;
  const float y_in = y_out * (float)height_in;

  // Nearest neighbours coordinates in input space
  int x_prev = (int)floor(x_in);
  int x_next = x_prev + 1;
  int y_prev = (int)floor(y_in);
  int y_next = y_prev + 1;

  x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  dt_bilinear_nodes_t n;

  // Nearest pixels in input array (nodes in grid)
  n.nw = mad24(y_prev, width_in, x_prev);
  n.ne = mad24(y_prev, width_in, x_next);
  n.se = mad24(y_next, width_in, x_next);
  n.sw = mad24(y_next, width_in, x_prev);

  // Spatial differences between nodes
  n.dy_next = (float)y_next - y_in;
  n.dy_prev = 1.0f - n.dy_next; // because next - prev = 1
  n.dx_next = (float)x_next - x_in;
  n.dx_prev = 1.0f - n.dx_next; // because next - prev = 1

  return n;
}


kernel void bilinear1(global const float *const in,
                      const int width_in,
                      const int height_in,
                      global float *const out,
                      const int width_out,
                      const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const dt_bilinear_nodes_t n =
    _bilinear_nodes(x, y, width_in, height_in, width_out, height_out);

  out[mad24(y, width_out, x)] =
      n.dy_prev * (in[n.sw] * n.dx_next + in[n.se] * n.dx_prev)
    + n.dy_next * (in[n.nw] * n.dx_next + in[n.ne] * n.dx_prev);
}


kernel void bilinear2(global const float2 *const in,
                      const int width_in,
                      const int height_in,
                      global float2 *const out,
                      const int width_out,
                      const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const dt_bilinear_nodes_t n =
    _bilinear_nodes(x, y, width_in, height_in, width_out, height_out);

  out[mad24(y, width_out, x)] =
      n.dy_prev * (in[n.sw] * n.dx_next + in[n.se] * n.dx_prev)
    + n.dy_next * (in[n.nw] * n.dx_next + in[n.ne] * n.dx_prev);
}


kernel void bilinear4(global const float4 *const in,
                      const int width_in,
                      const int height_in,
                      global float4 *const out,
                      const int width_out,
                      const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const dt_bilinear_nodes_t n =
    _bilinear_nodes(x, y, width_in, height_in, width_out, height_out);

  out[mad24(y, width_out, x)] =
      n.dy_prev * (in[n.sw] * n.dx_next + in[n.se] * n.dx_prev)
    + n.dy_next * (in[n.nw] * n.dx_next + in[n.ne] * n.dx_prev);
}
