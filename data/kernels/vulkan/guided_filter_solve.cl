// Vulkan port of guided_filter.cl :: guided_filter_solve.
//
// Per-pixel 3x3 linear solve (Cramer's rule) for the guided-filter
// coefficients a = Sigma^-1 * cov, b = mean_img - a . mean_guide.
// Falls back to a = 0, b = mean_img where the system is singular,
// exactly as the OpenCL kernel.
//
// 17 bindings (13 read + 4 write) — the reason DT_VULKAN_MAX_BINDINGS
// is 20. Binding order matches the OpenCL argument order:
//   0: img_mean
//   1..3: imgg_mean_r/g/b
//   4..6: cov_imgg_img_r/g/b
//   7..12: var_imgg_rr/rg/rb/gg/gb/bb
//   13..15: a_r/a_g/a_b   (write)
//   16: b                 (write)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void guided_filter_solve(global const float *img_mean,
                                global const float *imgg_mean_r,
                                global const float *imgg_mean_g,
                                global const float *imgg_mean_b,
                                global const float *cov_imgg_img_r,
                                global const float *cov_imgg_img_g,
                                global const float *cov_imgg_img_b,
                                global const float *var_imgg_rr,
                                global const float *var_imgg_rg,
                                global const float *var_imgg_rb,
                                global const float *var_imgg_gg,
                                global const float *var_imgg_gb,
                                global const float *var_imgg_bb,
                                global       float *a_r,
                                global       float *a_g,
                                global       float *a_b,
                                global       float *b,
                                const int width,
                                const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float Sigma_0_0 = var_imgg_rr[idx];
  const float Sigma_0_1 = var_imgg_rg[idx];
  const float Sigma_0_2 = var_imgg_rb[idx];
  const float Sigma_1_1 = var_imgg_gg[idx];
  const float Sigma_1_2 = var_imgg_gb[idx];
  const float Sigma_2_2 = var_imgg_bb[idx];
  const float cov0 = cov_imgg_img_r[idx];
  const float cov1 = cov_imgg_img_g[idx];
  const float cov2 = cov_imgg_img_b[idx];

  const float det0 = Sigma_0_0 * (Sigma_1_1 * Sigma_2_2 - Sigma_1_2 * Sigma_1_2)
                   - Sigma_0_1 * (Sigma_0_1 * Sigma_2_2 - Sigma_0_2 * Sigma_1_2)
                   + Sigma_0_2 * (Sigma_0_1 * Sigma_1_2 - Sigma_0_2 * Sigma_1_1);
  float a_r_ = 0.0f;
  float a_g_ = 0.0f;
  float a_b_ = 0.0f;
  float b_ = img_mean[idx];
  if(fabs(det0) > 4.f * FLT_EPSILON)
  {
    const float det1 = cov0 * (Sigma_1_1 * Sigma_2_2 - Sigma_1_2 * Sigma_1_2)
                     - Sigma_0_1 * (cov1 * Sigma_2_2 - cov2 * Sigma_1_2)
                     + Sigma_0_2 * (cov1 * Sigma_1_2 - cov2 * Sigma_1_1);
    const float det2 = Sigma_0_0 * (cov1 * Sigma_2_2 - cov2 * Sigma_1_2)
                     - cov0 * (Sigma_0_1 * Sigma_2_2 - Sigma_0_2 * Sigma_1_2)
                     + Sigma_0_2 * (Sigma_0_1 * cov2 - Sigma_0_2 * cov1);
    const float det3 = Sigma_0_0 * (Sigma_1_1 * cov2 - Sigma_1_2 * cov1)
                     - Sigma_0_1 * (Sigma_0_1 * cov2 - Sigma_0_2 * cov1)
                     + cov0 * (Sigma_0_1 * Sigma_1_2 - Sigma_0_2 * Sigma_1_1);
    a_r_ = det1 / det0;
    a_g_ = det2 / det0;
    a_b_ = det3 / det0;
    b_ = b_
       - a_r_ * imgg_mean_r[idx]
       - a_g_ * imgg_mean_g[idx]
       - a_b_ * imgg_mean_b[idx];
  }

  a_r[idx] = a_r_;
  a_g[idx] = a_g_;
  a_b[idx] = a_b_;
  b[idx]   = b_;
}
