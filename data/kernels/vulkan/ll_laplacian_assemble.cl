// Vulkan port of locallaplacian.cl :: laplacian_assemble.
//
// Coarse-to-fine pyramid reconstruction step: at each fine pixel,
// reads the coarse output via `expand_gaussian` (the bilinear-with-
// binomial-weights upsampler) and adds the laplacian = fine_g -
// expand(coarse_g) interpolated across two adjacent gamma curves.
//
// Image-shortcut port: all reads use sampler-clamp integer coords,
// which translate to flat buffer reads with explicit clamp. The
// coarse buffer dimensions (cw, ch) are passed via PC so the clamps
// know where to stop.
//
// Binding layout (15 storage buffers, all `float`):
//   0:  input        — fine gaussian at this pyramid level (pw×ph)
//   1:  output1      — coarse output (cw×ch)
//   2:  output0      — fine output (write target, pw×ph)
//   3-14: g0_l0, g0_l1, g1_l0, g1_l1, g2_l0, g2_l1,
//         g3_l0, g3_l1, g4_l0, g4_l1, g5_l0, g5_l1
//   (l0 at fine pw×ph, l1 at coarse cw×ch).
// Push constants: 4 ints = 16 bytes (pw, ph, cw, ch).

#include "dt_vulkan_common.h"

static inline float ll_expand_gaussian(global const float *coarse,
                                       const int i, const int j,
                                       const int cw, const int ch)
{
  const float w[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };
  const int cx = i / 2;
  const int cy = j / 2;
  float c = 0.0f;
  switch((i & 1) + 2 * (j & 1))
  {
    case 0:
      for(int ii = -1; ii <= 1; ii++)
        for(int jj = -1; jj <= 1; jj++)
        {
          const int xx = clamp(cx + ii, 0, cw - 1);
          const int yy = clamp(cy + jj, 0, ch - 1);
          c += coarse[yy * cw + xx] * w[2 * jj + 2] * w[2 * ii + 2];
        }
      break;
    case 1:
      for(int ii = 0; ii <= 1; ii++)
        for(int jj = -1; jj <= 1; jj++)
        {
          const int xx = clamp(cx + ii, 0, cw - 1);
          const int yy = clamp(cy + jj, 0, ch - 1);
          c += coarse[yy * cw + xx] * w[2 * jj + 2] * w[2 * ii + 1];
        }
      break;
    case 2:
      for(int ii = -1; ii <= 1; ii++)
        for(int jj = 0; jj <= 1; jj++)
        {
          const int xx = clamp(cx + ii, 0, cw - 1);
          const int yy = clamp(cy + jj, 0, ch - 1);
          c += coarse[yy * cw + xx] * w[2 * jj + 1] * w[2 * ii + 2];
        }
      break;
    default: // case 3
      for(int ii = 0; ii <= 1; ii++)
        for(int jj = 0; jj <= 1; jj++)
        {
          const int xx = clamp(cx + ii, 0, cw - 1);
          const int yy = clamp(cy + jj, 0, ch - 1);
          c += coarse[yy * cw + xx] * w[2 * jj + 1] * w[2 * ii + 1];
        }
      break;
  }
  return 4.0f * c;
}

static inline float ll_laplacian(global const float *coarse,
                                 global const float *fine,
                                 const int i, const int j,
                                 const int ci, const int cj,
                                 const int pw, const int ph,
                                 const int cw, const int ch)
{
  const float c = ll_expand_gaussian(coarse, ci, cj, cw, ch);
  return fine[clamp(j, 0, ph - 1) * pw + clamp(i, 0, pw - 1)] - c;
}

kernel void ll_laplacian_assemble(global const float *input,
                                  global const float *output1,
                                  global       float *output0,
                                  global const float *g0_l0, global const float *g0_l1,
                                  global const float *g1_l0, global const float *g1_l1,
                                  global const float *g2_l0, global const float *g2_l1,
                                  global const float *g3_l0, global const float *g3_l1,
                                  global const float *g4_l0, global const float *g4_l1,
                                  global const float *g5_l0, global const float *g5_l1,
                                  const int pw, const int ph,
                                  const int cw, const int ch)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= pw || y >= ph) return;

  int i = x, j = y;
  // Mirror the OpenCL kernel's boundary handling byte-for-byte.
  if(pw & 1) { if(x > pw - 2) i = pw - 2; }
  else       { if(x > pw - 3) i = pw - 3; }
  if(ph & 1) { if(y > ph - 2) j = ph - 2; }
  else       { if(y > ph - 3) j = ph - 3; }
  if(x <= 0) i = 1;
  if(y <= 0) j = 1;

  float pixel = ll_expand_gaussian(output1, i, j, cw, ch);

  const int num_gamma = 6;
  const float v = input[y * pw + x];
  int hi = 1;
  for(; hi < num_gamma - 1 && ((float)hi + 0.5f) / (float)num_gamma <= v; hi++);
  int lo = hi - 1;
  const float a = clamp(v * (float)num_gamma - ((float)lo + 0.5f), 0.0f, 1.0f);

  float l0 = 0.0f, l1 = 0.0f;
  switch(lo)
  {
    case 0:
      l0 = ll_laplacian(g0_l1, g0_l0, x, y, i, j, pw, ph, cw, ch);
      l1 = ll_laplacian(g1_l1, g1_l0, x, y, i, j, pw, ph, cw, ch);
      break;
    case 1:
      l0 = ll_laplacian(g1_l1, g1_l0, x, y, i, j, pw, ph, cw, ch);
      l1 = ll_laplacian(g2_l1, g2_l0, x, y, i, j, pw, ph, cw, ch);
      break;
    case 2:
      l0 = ll_laplacian(g2_l1, g2_l0, x, y, i, j, pw, ph, cw, ch);
      l1 = ll_laplacian(g3_l1, g3_l0, x, y, i, j, pw, ph, cw, ch);
      break;
    case 3:
      l0 = ll_laplacian(g3_l1, g3_l0, x, y, i, j, pw, ph, cw, ch);
      l1 = ll_laplacian(g4_l1, g4_l0, x, y, i, j, pw, ph, cw, ch);
      break;
    default:
      l0 = ll_laplacian(g4_l1, g4_l0, x, y, i, j, pw, ph, cw, ch);
      l1 = ll_laplacian(g5_l1, g5_l0, x, y, i, j, pw, ph, cw, ch);
      break;
  }
  pixel += l0 * (1.0f - a) + l1 * a;
  output0[y * pw + x] = pixel;
}
