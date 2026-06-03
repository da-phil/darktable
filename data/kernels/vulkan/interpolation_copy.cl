// Vulkan port of basic.cl :: interpolation_copy.
//
// The 1:1 (scale == 1) expand path of the resampler: copies the
// input into the output at an offset (dx, dy), zero-filling output
// pixels whose source falls outside the input. Used when
// roi_out->scale == 1 but the input doesn't fully cover the output
// rectangle (the "expanded" crop case).
//
// Binding layout (2 storage buffers):
//   0: in   (float4, iwidth × iheight)
//   1: out  (float4, owidth × oheight)
// Push constants: 6 ints = 24 bytes (owidth, oheight, iwidth, iheight, dx, dy).

#include "dt_vulkan_common.h"

kernel void interpolation_copy(global const float4 *in,
                               global       float4 *out,
                               const int owidth,
                               const int oheight,
                               const int iwidth,
                               const int iheight,
                               const int dx,
                               const int dy)
{
  const int ocol = get_global_id(0);
  const int orow = get_global_id(1);
  if(ocol >= owidth || orow >= oheight) return;

  float4 pix = (float4)(0.0f);
  const int irow = orow + dy;
  const int icol = ocol + dx;
  if(irow >= 0 && irow < iheight && icol >= 0 && icol < iwidth)
    pix = in[irow * iwidth + icol];

  out[orow * owidth + ocol] = pix;
}
