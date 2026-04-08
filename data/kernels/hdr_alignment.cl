/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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

/* OpenCL kernels for HDR merge image alignment.
 *
 * These kernels accelerate the pixel-level operations used in the
 * multi-resolution ECC (Enhanced Correlation Coefficient) alignment
 * pipeline:
 *   - Image warping by a 3×3 homography
 *   - Sobel gradient computation
 *   - log(1 + x) dynamic-range compression
 *   - Signed Sobel gradient sum (gx + gy) for exposure-invariant ECC input
 *   - Mean-absolute-deviation gradient normalisation
 *   - Bayer mosaic to grayscale conversion
 *   - 2× box-filter downsampling
 *   - ECC weighted accumulation passes
 *
 * Gradient preprocessing pipeline (applied before ECC):
 *   1. Percentile normalisation of raw pixels [CPU-only due to histogram pass]
 *   2. hdr_align_log1p          – log(1 + x) dynamic range compression
 *   3. hdr_align_compute_gradients – Sobel gx, gy
 *   4. hdr_align_gradient_sobel_sum – combine into gx + gy (signed)
 *   5. hdr_align_normalize_mad  – g / (mean(|g|) + ε)
 */

/* ---------- Warp by projective homography ----------
 *
 * Backward-mapping warp with bilinear interpolation:
 *   sx = (H0*x + H1*y + H2) / (H6*x + H7*y + 1)
 *   sy = (H3*x + H4*y + H5) / (H6*x + H7*y + 1)
 *
 * Out-of-bounds pixels are set to 0 with mask = 0.
 * Valid pixels get mask = 1.
 */
kernel void
hdr_align_warp_homography(global const float *in,
                          global float *out,
                          global float *mask,
                          const int width,
                          const int height,
                          const float H0, const float H1, const float H2,
                          const float H3, const float H4, const float H5,
                          const float H6, const float H7)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t idx = (size_t)y * width + x;
  const float xf = (float)x;
  const float yf = (float)y;
  const float den = H6 * xf + H7 * yf + 1.0f;

  if(fabs(den) < 1e-8f)
  {
    out[idx] = 0.0f;
    mask[idx] = 0.0f;
    return;
  }

  const float sx = (H0 * xf + H1 * yf + H2) / den;
  const float sy = (H3 * xf + H4 * yf + H5) / den;

  // Bilinear interpolation with bounds check
  const int ix = (int)floor(sx);
  const int iy = (int)floor(sy);

  if(ix < 0 || ix >= width - 1 || iy < 0 || iy >= height - 1)
  {
    out[idx] = 0.0f;
    mask[idx] = 0.0f;
    return;
  }

  const float fx = sx - (float)ix;
  const float fy = sy - (float)iy;
  const size_t base = (size_t)iy * width + ix;

  const float v00 = in[base];
  const float v10 = in[base + 1];
  const float v01 = in[base + width];
  const float v11 = in[base + width + 1];

  out[idx] = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10)
           +         fy  * ((1.0f - fx) * v01 + fx * v11);
  mask[idx] = 1.0f;
}


/* ---------- Sobel gradient computation ----------
 *
 * Computes horizontal (gx) and vertical (gy) gradients using 3×3 Sobel
 * operators.  Border pixels are set to 0.
 */
kernel void
hdr_align_compute_gradients(global const float *in,
                            global float *gx,
                            global float *gy,
                            const int width,
                            const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t idx = (size_t)y * width + x;

  // Border pixels
  if(x == 0 || x >= width - 1 || y == 0 || y >= height - 1)
  {
    gx[idx] = 0.0f;
    gy[idx] = 0.0f;
    return;
  }

  // Sobel horizontal: [-1 0 1; -2 0 2; -1 0 1]
  const float dx = -in[(y - 1) * width + (x - 1)]
                   + in[(y - 1) * width + (x + 1)]
                   - 2.0f * in[y * width + (x - 1)]
                   + 2.0f * in[y * width + (x + 1)]
                   - in[(y + 1) * width + (x - 1)]
                   + in[(y + 1) * width + (x + 1)];

  // Sobel vertical: [-1 -2 -1; 0 0 0; 1 2 1]
  const float dy = -in[(y - 1) * width + (x - 1)]
                   - 2.0f * in[(y - 1) * width + x]
                   - in[(y - 1) * width + (x + 1)]
                   + in[(y + 1) * width + (x - 1)]
                   + 2.0f * in[(y + 1) * width + x]
                   + in[(y + 1) * width + (x + 1)];

  gx[idx] = dx * 0.125f;
  gy[idx] = dy * 0.125f;
}


/* ---------- log(1 + x) dynamic-range compression ----------
 *
 * Applied in-place after percentile normalisation and before Sobel gradient
 * computation.  Compresses highlights so that dark and bright exposures
 * produce comparable gradient magnitudes.
 */
kernel void
hdr_align_log1p(global float *img,
                const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;

  img[i] = log1p(img[i]);
}


/* ---------- Signed Sobel gradient sum (gx + gy) ----------
 *
 * Combines the horizontal (gx) and vertical (gy) Sobel gradients into a
 * single signed channel: out = gx + gy.  The result retains sign and
 * directional information from both axes.  This is the ECC feature image
 * used instead of the unsigned gradient magnitude.
 *
 * Both gx and gy must have been filled by hdr_align_compute_gradients first.
 * The combined result is written back to gx (in-place); gy is not modified.
 */
kernel void
hdr_align_gradient_sobel_sum(global float *gx,
                             global const float *gy,
                             const int npix)
{
  const int i = get_global_id(0);
  if(i >= npix) return;

  gx[i] = gx[i] + gy[i];
}


/* ---------- Mean-absolute-deviation gradient normalisation ----------
 *
 * Normalises a signed gradient image in-place:
 *   g[i] = g[i] / (mean(|g|) + ε)
 *
 * This prevents one image from dominating the ECC objective due to exposure
 * differences while preserving gradient sign.
 *
 * The mean absolute value must be computed on the host (a two-pass reduction)
 * and supplied as inv_scale = 1.0f / (mean_abs + 1e-7f).
 */
kernel void
hdr_align_normalize_mad(global float *g,
                        const int npix,
                        const float inv_scale)
{
  const int i = get_global_id(0);
  if(i >= npix) return;

  g[i] *= inv_scale;
}


/* ---------- CFA-aware stride-2 Sobel gradient (L0 full-resolution Bayer) ----------
 *
 * Computes the signed Sobel gradient sum (gx + gy) for a full-resolution Bayer
 * mosaic image.  Each output pixel uses only Sobel stencil neighbours from the
 * same CFA channel (stride-2 offsets), avoiding amplitude-mixing between the
 * G and R/B sublattices.
 *
 * Must be called AFTER per-sublattice normalisation (done on CPU via
 * _normalize_bayer_per_channel) so that all four sublattices have comparable
 * amplitude ranges.  Border pixels within 2 of the image edge are set to zero.
 */
kernel void
hdr_align_gradient_bayer_cfa_sobel(global const float *bayer,
                                   global float *out,
                                   const int width,
                                   const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const size_t idx = (size_t)y * width + x;
  if(x < 2 || x >= width - 2 || y < 2 || y >= height - 2)
  {
    out[idx] = 0.0f;
    return;
  }

  const float tl = bayer[(size_t)(y - 2) * width + (x - 2)];
  const float tc = bayer[(size_t)(y - 2) * width + x];
  const float tr = bayer[(size_t)(y - 2) * width + (x + 2)];
  const float ml = bayer[(size_t)y * width + (x - 2)];
  const float mr = bayer[(size_t)y * width + (x + 2)];
  const float bl = bayer[(size_t)(y + 2) * width + (x - 2)];
  const float bc = bayer[(size_t)(y + 2) * width + x];
  const float br = bayer[(size_t)(y + 2) * width + (x + 2)];

  const float gx = (-tl + tr - 2.0f * ml + 2.0f * mr - bl + br) * 0.125f;
  const float gy = (-tl - 2.0f * tc - tr + bl + 2.0f * bc + br) * 0.125f;
  out[idx] = gx + gy;
}

/* ---------- 2× box-filter downsample ---------- */
kernel void
hdr_align_downsample_2x(global const float *in,
                        global float *out,
                        const int in_w,
                        const int out_w,
                        const int out_h)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= out_w || y >= out_h) return;

  const int sx = x * 2;
  const int sy = y * 2;
  out[(size_t)y * out_w + x] = 0.25f * (in[(size_t)sy * in_w + sx]
                                        + in[(size_t)sy * in_w + sx + 1]
                                        + in[(size_t)(sy + 1) * in_w + sx]
                                        + in[(size_t)(sy + 1) * in_w + sx + 1]);
}


/* ---------- ECC Pass 1: Weighted means ----------
 *
 * Computes partial sums for a work-group tile.
 * Results are accumulated per work-group into a global partial-sums buffer.
 * The host performs the final reduction across work-groups.
 *
 * out_sums layout per work-group: [sum_r, sum_w, sum_weight, nvalid]
 */
kernel void
hdr_align_ecc_means(global const float *ref,
                    global const float *warped,
                    global const float *mask,
                    global float *out_sums,
                    const int width,
                    const int height,
                    const float cx,
                    const float cy,
                    const float inv_s,
                    const float edge_weight,
                    local float *local_sums)
{
  const int lid = get_local_id(0) + get_local_id(1) * get_local_size(0);
  const int lsize = get_local_size(0) * get_local_size(1);
  const int gid = get_group_id(0) + get_group_id(1) * get_num_groups(0);

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // 4 accumulators per work-item
  float s_r = 0.0f, s_w = 0.0f, s_wgt = 0.0f, s_n = 0.0f;

  if(x < width && y < height)
  {
    const size_t i = (size_t)y * width + x;
    if(mask[i] > 0.5f)
    {
      const float xn = ((float)x - cx) * inv_s;
      const float yn = ((float)y - cy) * inv_s;
      const float r2 = min(1.0f, 0.5f * (xn * xn + yn * yn));
      const float wgt = 1.0f + edge_weight * r2;
      s_r = wgt * ref[i];
      s_w = wgt * warped[i];
      s_wgt = wgt;
      s_n = 1.0f;
    }
  }

  // Store in local memory: interleaved [sum_r, sum_w, sum_weight, nvalid]
  local_sums[lid * 4 + 0] = s_r;
  local_sums[lid * 4 + 1] = s_w;
  local_sums[lid * 4 + 2] = s_wgt;
  local_sums[lid * 4 + 3] = s_n;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction within work-group
  for(int stride = lsize / 2; stride > 0; stride >>= 1)
  {
    if(lid < stride)
    {
      local_sums[lid * 4 + 0] += local_sums[(lid + stride) * 4 + 0];
      local_sums[lid * 4 + 1] += local_sums[(lid + stride) * 4 + 1];
      local_sums[lid * 4 + 2] += local_sums[(lid + stride) * 4 + 2];
      local_sums[lid * 4 + 3] += local_sums[(lid + stride) * 4 + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write work-group result
  if(lid == 0)
  {
    out_sums[gid * 4 + 0] = local_sums[0];
    out_sums[gid * 4 + 1] = local_sums[1];
    out_sums[gid * 4 + 2] = local_sums[2];
    out_sums[gid * 4 + 3] = local_sums[3];
  }
}


/* ---------- ECC Pass 2: Norms, Jacobian sums, sJw projection sums ----------
 *
 * Computes partial sums for a work-group tile.
 * out_sums layout per work-group (9 floats):
 *   [norm2_r, norm2_w, dot_rw, sum_J0, sum_J1, sum_J2,
 *    sJw0, sJw1, sJw2]
 *
 * The sJw terms (sJw[k] = Σ wgt·tw·J[k]) allow the host to derive
 * proj_coeff[k] = sJw[k] / norm2_w without a separate image pass,
 * matching the CPU _ecc_iteration merged-pass-2 optimization.
 * The mean_J correction term vanishes because Σ wgt·tw = 0 by
 * definition of mean_w.
 */
kernel void
hdr_align_ecc_norms(global const float *ref,
                    global const float *warped,
                    global const float *mask,
                    global const float *gx,
                    global const float *gy,
                    global float *out_sums,
                    const int width,
                    const int height,
                    const float cx,
                    const float cy,
                    const float inv_s,
                    const float s,
                    const float edge_weight,
                    const float mean_r,
                    const float mean_w,
                    const float cos_t,
                    const float sin_t,
                    local float *local_sums)
{
  const int lid = get_local_id(0) + get_local_id(1) * get_local_size(0);
  const int lsize = get_local_size(0) * get_local_size(1);
  const int gid = get_group_id(0) + get_group_id(1) * get_num_groups(0);

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float acc[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  if(x < width && y < height)
  {
    const size_t i = (size_t)y * width + x;
    if(mask[i] > 0.5f)
    {
      const float xn = ((float)x - cx) * inv_s;
      const float yn = ((float)y - cy) * inv_s;
      const float r2 = min(1.0f, 0.5f * (xn * xn + yn * yn));
      const float wgt = 1.0f + edge_weight * r2;
      const float gxi = gx[i];
      const float gyi = gy[i];

      const float J0 = s * (gxi * (-sin_t * xn + cos_t * yn)
                           + gyi * (-cos_t * xn - sin_t * yn));
      const float J1 = s * gxi;
      const float J2 = s * gyi;

      const float r = ref[i] - mean_r;
      const float tw = warped[i] - mean_w;

      acc[0] = wgt * r * r;       // norm2_r
      acc[1] = wgt * tw * tw;     // norm2_w
      acc[2] = wgt * r * tw;      // dot_rw
      acc[3] = wgt * J0;          // sum_J0
      acc[4] = wgt * J1;          // sum_J1
      acc[5] = wgt * J2;          // sum_J2
      acc[6] = wgt * tw * J0;     // sJw0 (projection numerator for J0)
      acc[7] = wgt * tw * J1;     // sJw1
      acc[8] = wgt * tw * J2;     // sJw2
    }
  }

  // Store in local memory
  for(int k = 0; k < 9; k++)
    local_sums[lid * 9 + k] = acc[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction
  for(int stride = lsize / 2; stride > 0; stride >>= 1)
  {
    if(lid < stride)
    {
      for(int k = 0; k < 9; k++)
        local_sums[lid * 9 + k] += local_sums[(lid + stride) * 9 + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    for(int k = 0; k < 9; k++)
      out_sums[gid * 9 + k] = local_sums[k];
  }
}


/* ---------- ECC Pass 3: Hessian + RHS assembly (given proj_coeff) ----------
 *
 * out_sums layout per work-group:
 *   [H00, H01, H02, H11, H12, H22, rhs0, rhs1, rhs2]  (9 floats)
 *
 * The host derives proj_coeff[k] = sJw[k] / norm2_w from the output of
 * hdr_align_ecc_norms, then passes proj0..proj2 as kernel arguments.
 * This eliminates the former intermediate hdr_align_ecc_hessian kernel.
 */
kernel void
hdr_align_ecc_hessian_final(global const float *ref,
                            global const float *warped,
                            global const float *mask,
                            global const float *gx,
                            global const float *gy,
                            global float *out_sums,
                            const int width,
                            const int height,
                            const float cx,
                            const float cy,
                            const float inv_s,
                            const float s,
                            const float edge_weight,
                            const float mean_r,
                            const float mean_w,
                            const float cos_t,
                            const float sin_t,
                            const float mean_J0,
                            const float mean_J1,
                            const float mean_J2,
                            const float proj0,
                            const float proj1,
                            const float proj2,
                            const float scale_rw,
                            const float rho,
                            local float *local_sums)
{
  const int lid = get_local_id(0) + get_local_id(1) * get_local_size(0);
  const int lsize = get_local_size(0) * get_local_size(1);
  const int gid = get_group_id(0) + get_group_id(1) * get_num_groups(0);

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // 9 accumulators: Hess upper-triangle [H00..H22] (6) + rhs[3]
  float acc[9];
  for(int k = 0; k < 9; k++) acc[k] = 0.0f;

  if(x < width && y < height)
  {
    const size_t i = (size_t)y * width + x;
    if(mask[i] > 0.5f)
    {
      const float xn = ((float)x - cx) * inv_s;
      const float yn = ((float)y - cy) * inv_s;
      const float r2 = min(1.0f, 0.5f * (xn * xn + yn * yn));
      const float wgt = 1.0f + edge_weight * r2;
      const float gxi = gx[i];
      const float gyi = gy[i];

      const float J0 = s * (gxi * (-sin_t * xn + cos_t * yn)
                           + gyi * (-cos_t * xn - sin_t * yn));
      const float J1 = s * gxi;
      const float J2 = s * gyi;

      const float tw = warped[i] - mean_w;
      const float r_val = ref[i] - mean_r;
      const float ei = scale_rw * r_val - rho * tw;

      const float Jp0 = (J0 - mean_J0) - proj0 * tw;
      const float Jp1 = (J1 - mean_J1) - proj1 * tw;
      const float Jp2 = (J2 - mean_J2) - proj2 * tw;

      // Upper triangle of symmetric Hessian
      acc[0] = wgt * Jp0 * Jp0;  // H00
      acc[1] = wgt * Jp0 * Jp1;  // H01
      acc[2] = wgt * Jp0 * Jp2;  // H02
      acc[3] = wgt * Jp1 * Jp1;  // H11
      acc[4] = wgt * Jp1 * Jp2;  // H12
      acc[5] = wgt * Jp2 * Jp2;  // H22

      // RHS
      acc[6] = wgt * Jp0 * ei;   // rhs0
      acc[7] = wgt * Jp1 * ei;   // rhs1
      acc[8] = wgt * Jp2 * ei;   // rhs2
    }
  }

  // Store in local memory
  for(int k = 0; k < 9; k++)
    local_sums[lid * 9 + k] = acc[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction
  for(int stride = lsize / 2; stride > 0; stride >>= 1)
  {
    if(lid < stride)
    {
      for(int k = 0; k < 9; k++)
        local_sums[lid * 9 + k] += local_sums[(lid + stride) * 9 + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
  {
    for(int k = 0; k < 9; k++)
      out_sums[gid * 9 + k] = local_sums[k];
  }
}
