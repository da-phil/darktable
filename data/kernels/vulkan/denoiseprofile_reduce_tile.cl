// Vulkan replacement for denoiseprofile.cl :: reduce_first + reduce_second.
//
// The original two-kernel reduction uses workgroup-shared memory and a
// tree-reduce within each workgroup. Our HAL doesn't expose local
// memory, so we fold both passes into a single tile-sum kernel: each
// work-item processes one TILE_SIZE × TILE_SIZE block of the detail
// buffer and writes one float4 of squared sums to dev_m. Host reads
// dev_m back and sums on CPU (the small dimension makes the readback
// cheap — for 5 wavelet scales × ceil(W/TS)*ceil(H/TS) tiles it's a
// few hundred KB total).
//
// Binding layout (2 storage buffers):
//   0: in   (float4)   — detail buffer
//   1: out  (float4)   — per-tile squared-sum array, length = ntiles
// Push constants: 12 B (width, height, tile_size).

#include "dt_vulkan_common.h"

kernel void denoiseprofile_reduce_tile(
    global const float4 *in_buf,
    global       float4 *out_buf,
    const int width, const int height, const int tile_size)
{
  const int tx = get_global_id(0);
  const int ty = get_global_id(1);
  const int ntiles_x = (width  + tile_size - 1) / tile_size;
  const int ntiles_y = (height + tile_size - 1) / tile_size;
  if(tx >= ntiles_x || ty >= ntiles_y) return;

  float4 sum = (float4)(0.0f);
  for(int j = 0; j < tile_size; j++)
    for(int i = 0; i < tile_size; i++)
    {
      const int x = tx * tile_size + i;
      const int y = ty * tile_size + j;
      if(x < width && y < height)
      {
        const float4 px = in_buf[y * width + x];
        sum += px * px;
      }
    }

  out_buf[ty * ntiles_x + tx] = sum;
}
