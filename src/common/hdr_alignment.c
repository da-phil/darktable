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

#include "common/hdr_alignment.h"
#include "common/darktable.h"
#include "common/interpolation.h"
#include "common/math.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Maximum rotation search range in degrees (for coarse exhaustive search)
#define HDR_ALIGN_MAX_ANGLE_DEG 10.0f
// Coarse rotation step in degrees
#define HDR_ALIGN_COARSE_ANGLE_STEP 0.5f
// Maximum number of pyramid levels (enough for up to 32k images)
#define HDR_ALIGN_MAX_PYRAMID_LEVELS 12
// Target longest-side for the coarsest pyramid level
#define HDR_ALIGN_COARSEST_SIZE 64
// Minimum image dimension for alignment to make sense
#define HDR_ALIGN_MIN_DIM 64
// Maximum ECC iterations per pyramid level
#define HDR_ALIGN_ECC_MAX_ITER 50
// ECC convergence threshold (pixel-equivalent): stop when parameter update
// norm (|Δtx| + |Δty| + |Δθ|·corner_dist) is below this value.
#define HDR_ALIGN_ECC_EPSILON 1e-2f
// Extra weight given to outer image regions during ECC so edge/corner
// alignment has enough influence over the homography estimate.
#define HDR_ALIGN_ECC_EDGE_WEIGHT 3.0
// Minimum fraction of valid (in-bounds + unclipped) pixels for ECC to proceed
#define HDR_ALIGN_ECC_MIN_VALID_FRAC 0.3f
// Coarse search translation radius as fraction of the longest side of the
// coarsest pyramid level.  This must be large enough that the true
// displacement between HDR brackets is always within the search area.
// At a typical coarsest level of ~37×24 (scale factor ~128× from full-res),
// a fraction of 0.35 gives radius=12, covering ~1536 full-resolution pixels —
// enough for even large hand-held displacements.
#define HDR_ALIGN_COARSE_SEARCH_FRAC 0.35f
// Minimum image dimension for ECC to be numerically stable
#define HDR_ALIGN_ECC_MIN_DIM 48
// Number of consecutive iterations without improvement before declaring stall.
// Avoids wasting iterations at coarse pyramid levels where the update metric
// plateaus above the convergence threshold.
#define HDR_ALIGN_ECC_PATIENCE 5
// Maximum rotation change (degrees) that a single ECC pyramid level is allowed
// to introduce.  If ECC drifts the angle further than this it has wandered to
// a wrong local maximum; the result is reverted to the pre-level homography.
#define HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG 2.0f
// Maximum translation change (pixels at the current level) that a single ECC
// pyramid level may introduce.  Each level inherits a 2×-scaled estimate from
// its parent, so the parent's accuracy at the child's scale is ~2 px.
// Allowing up to 3 px gives sufficient room for genuine sub-pixel refinement
// while catching runaway ECC drift that would compound across levels.
// In well-aligned tripod shots the per-level change is typically <1 px;
// hand-held brackets occasionally reach ~2.5 px at coarser levels.
#define HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX 3.0f
// Number of free parameters in projective homography (h22 fixed to 1)
#define HDR_ALIGN_H_NPARAM 8
// Number of Jacobi smoothing iterations for the residual mesh.
#define HDR_ALIGN_MESH_SMOOTH_ITERS 12
// Smoothness weight used when regularising the residual mesh.
#define HDR_ALIGN_MESH_SMOOTH_LAMBDA 1.5f

// --- Adaptive DOF escalation parameters ---
// Weighted ECC score below which DOF escalation is attempted.
// A score of 1.0 means perfect correlation; lower values indicate the
// rigid model left residual misalignment.
#define HDR_ALIGN_ESCALATION_RHO_THRESHOLD 0.85f
// Minimum ρ improvement required to accept a higher-DOF result.
#define HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT 0.01f
// Maximum Hessian condition number for accepting a higher-DOF result.
// A poorly conditioned Hessian indicates the extra parameters are not
// well determined by the data.
#define HDR_ALIGN_ESCALATION_MAX_COND 1e6
// Maximum ECC iterations for the escalated (higher-DOF) solver.
#define HDR_ALIGN_ESCALATION_MAX_ITER 30
// ECC convergence threshold for the escalated solver.
#define HDR_ALIGN_ESCALATION_EPSILON 5e-3f

// Maximum area-scaling deviation allowed in an escalated homography.
// The determinant of the upper-left 2×2 equals the local area magnification;
// for HDR brackets from the same camera / lens this must be very close to 1.0.
// Values far from 1 indicate that the extra DOF are fitting exposure or
// vignetting gradients as geometric scaling, producing a wrong result.
#define HDR_ALIGN_ESCALATION_MAX_SCALE_DEVIATION 0.01f

// Coarse NCC score for identity (tx=0, ty=0, angle=0°) above which the
// images are considered already well-aligned and the full ECC pyramid is
// skipped.  Gradient-domain NCC values above 0.98 indicate near-perfect
// edge alignment at the coarsest level -- any ECC refinement risks
// introducing drift for no gain.
#define HDR_ALIGN_COARSE_IDENTITY_SKIP_NCC 0.98f

#define HDR_ALIGN_MESH_INDEX(row, col) ((row) * DT_HDR_ALIGN_MESH_COLS + (col))

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Downsample a Bayer mosaic image to grayscale by averaging 2x2 blocks.
 *  Output dimensions are (wd/2) x (ht/2).  Caller must free the result. */
static float *_mosaic_to_grayscale(const float *mosaic,
                                   const int wd,
                                   const int ht,
                                   int *out_w,
                                   int *out_h)
{
  const int gw = wd / 2;
  const int gh = ht / 2;
  float *gray = dt_alloc_align_float((size_t)gw * gh);
  if(!gray) return NULL;

  DT_OMP_FOR(collapse(2))
  for(int gy = 0; gy < gh; gy++)
    for(int gx = 0; gx < gw; gx++)
    {
      const int mx = gx * 2;
      const int my = gy * 2;
      gray[gy * gw + gx] = 0.25f * (mosaic[my * wd + mx]
                                     + mosaic[my * wd + mx + 1]
                                     + mosaic[(my + 1) * wd + mx]
                                     + mosaic[(my + 1) * wd + mx + 1]);
    }

  *out_w = gw;
  *out_h = gh;
  return gray;
}

/** Downsample an image by 2x using a box filter.
 *  Output dimensions are (w/2) x (h/2).  Caller must free the result. */
static float *_downsample_2x(const float *in,
                             const int w,
                             const int h,
                             int *out_w,
                             int *out_h)
{
  const int nw = w / 2;
  const int nh = h / 2;
  float *out = dt_alloc_align_float((size_t)nw * nh);
  if(!out) return NULL;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < nh; y++)
    for(int x = 0; x < nw; x++)
    {
      const int sx = x * 2;
      const int sy = y * 2;
      out[y * nw + x] = 0.25f * (in[sy * w + sx]
                                  + in[sy * w + sx + 1]
                                  + in[(sy + 1) * w + sx]
                                  + in[(sy + 1) * w + sx + 1]);
    }

  *out_w = nw;
  *out_h = nh;
  return out;
}

/** Warp an image by a Euclidean transform (translation + rotation around
 *  image centre).  Out-of-bounds pixels are set to 0 and the corresponding
 *  mask entry is set to 0.  Valid pixels get mask = 1.
 *  @p mask may be NULL if the caller doesn't need it.
 *  Caller must free returned image (and mask if non-NULL). */
static float *_warp_euclidean(const float *in,
                              const int w,
                              const int h,
                              const float tx,
                              const float ty,
                              const float angle,
                              float *mask)
{
  float *out = dt_alloc_align_float((size_t)w * h);
  if(!out) return NULL;

  const float cx = (w - 1) * 0.5f;
  const float cy = (h - 1) * 0.5f;
  // Inverse rotation for backward mapping
  const float cos_a = cosf(-angle);
  const float sin_a = sinf(-angle);

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      // Inverse warp: output(x,y) ← source at inverse-transformed location
      const float ox = (float)x - tx - cx;
      const float oy = (float)y - ty - cy;
      const float sx = cos_a * ox - sin_a * oy + cx;
      const float sy = sin_a * ox + cos_a * oy + cy;

      const int ix = (int)floorf(sx);
      const int iy = (int)floorf(sy);

      if(ix >= 0 && ix < w - 1 && iy >= 0 && iy < h - 1)
      {
        const float fx = sx - (float)ix;
        const float fy = sy - (float)iy;
        out[y * w + x] = (1.0f - fx) * (1.0f - fy) * in[iy * w + ix]
                        + fx * (1.0f - fy) * in[iy * w + ix + 1]
                        + (1.0f - fx) * fy * in[(iy + 1) * w + ix]
                        + fx * fy * in[(iy + 1) * w + ix + 1];
        if(mask) mask[y * w + x] = 1.0f;
      }
      else
      {
        out[y * w + x] = 0.0f;
        if(mask) mask[y * w + x] = 0.0f;
      }
    }

  return out;
}

/** Initialise a backward-mapping homography from Euclidean parameters
 *  compatible with _warp_euclidean semantics. */
static void _homography_from_euclidean(const float tx,
                                       const float ty,
                                       const float angle,
                                       const int w,
                                       const int h,
                                       float H[HDR_ALIGN_H_NPARAM])
{
  const float cx = (w - 1) * 0.5f;
  const float cy = (h - 1) * 0.5f;
  const float ca = cosf(angle);
  const float sa = sinf(angle);

  H[0] = ca;
  H[1] = sa;
  H[2] = -ca * (tx + cx) - sa * (ty + cy) + cx;
  H[3] = -sa;
  H[4] = ca;
  H[5] =  sa * (tx + cx) - ca * (ty + cy) + cy;
  H[6] = 0.0f;
  H[7] = 0.0f;
}

/** Scale a homography when moving from a coarser pyramid level to a finer
 *  level (coordinates are doubled). */
static void _homography_scale_to_finer(float H[HDR_ALIGN_H_NPARAM])
{
  H[2] *= 2.0f;
  H[5] *= 2.0f;
  H[6] *= 0.5f;
  H[7] *= 0.5f;
}

static void _homography_to_matrix(const float H[HDR_ALIGN_H_NPARAM],
                                  double M[3][3])
{
  M[0][0] = H[0]; M[0][1] = H[1]; M[0][2] = H[2];
  M[1][0] = H[3]; M[1][1] = H[4]; M[1][2] = H[5];
  M[2][0] = H[6]; M[2][1] = H[7]; M[2][2] = 1.0;
}

static void _homography_from_matrix(const double M[3][3],
                                    float H[HDR_ALIGN_H_NPARAM])
{
  const double s = fabs(M[2][2]) > 1e-16 ? 1.0 / M[2][2] : 1.0;
  H[0] = (float)(M[0][0] * s);
  H[1] = (float)(M[0][1] * s);
  H[2] = (float)(M[0][2] * s);
  H[3] = (float)(M[1][0] * s);
  H[4] = (float)(M[1][1] * s);
  H[5] = (float)(M[1][2] * s);
  H[6] = (float)(M[2][0] * s);
  H[7] = (float)(M[2][1] * s);
}

static void _mat3_mul(const double A[3][3],
                      const double B[3][3],
                      double C[3][3])
{
  for(int r = 0; r < 3; r++)
    for(int c = 0; c < 3; c++)
      C[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
}

/** Convert a homography between local half-resolution coordinates and full
 *  resolution coordinates using x_full = 2*x_local + off_x,
 *  y_full = 2*y_local + off_y. */
static void _homography_local_to_full(const float H_local[HDR_ALIGN_H_NPARAM],
                                      const float off_x,
                                      const float off_y,
                                      float H_full[HDR_ALIGN_H_NPARAM])
{
  const double A[3][3] = {
    { 2.0, 0.0, off_x },
    { 0.0, 2.0, off_y },
    { 0.0, 0.0, 1.0 }
  };
  const double Ainv[3][3] = {
    { 0.5, 0.0, -0.5 * off_x },
    { 0.0, 0.5, -0.5 * off_y },
    { 0.0, 0.0, 1.0 }
  };
  double Hm[3][3], tmp[3][3], out[3][3];
  _homography_to_matrix(H_local, Hm);
  _mat3_mul(A, Hm, tmp);
  _mat3_mul(tmp, Ainv, out);
  _homography_from_matrix(out, H_full);
}

static void _homography_pixel_to_normalized(const float H_pixel[HDR_ALIGN_H_NPARAM],
                                            const int w,
                                            const int h,
                                            float H_norm[HDR_ALIGN_H_NPARAM])
{
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;
  const double A[3][3] = {
    { s, 0.0, cx },
    { 0.0, s, cy },
    { 0.0, 0.0, 1.0 }
  };
  const double Ainv[3][3] = {
    { 1.0 / s, 0.0, -cx / s },
    { 0.0, 1.0 / s, -cy / s },
    { 0.0, 0.0, 1.0 }
  };
  double Hm[3][3], tmp[3][3], out[3][3];
  _homography_to_matrix(H_pixel, Hm);
  _mat3_mul(Ainv, Hm, tmp);
  _mat3_mul(tmp, A, out);
  _homography_from_matrix(out, H_norm);
}

static void _homography_normalized_to_pixel(const float H_norm[HDR_ALIGN_H_NPARAM],
                                            const int w,
                                            const int h,
                                            float H_pixel[HDR_ALIGN_H_NPARAM])
{
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;
  const double A[3][3] = {
    { s, 0.0, cx },
    { 0.0, s, cy },
    { 0.0, 0.0, 1.0 }
  };
  const double Ainv[3][3] = {
    { 1.0 / s, 0.0, -cx / s },
    { 0.0, 1.0 / s, -cy / s },
    { 0.0, 0.0, 1.0 }
  };
  double Hm[3][3], tmp[3][3], out[3][3];
  _homography_to_matrix(H_norm, Hm);
  _mat3_mul(A, Hm, tmp);
  _mat3_mul(tmp, Ainv, out);
  _homography_from_matrix(out, H_pixel);
}

static inline double _ecc_spatial_weight(const double xn, const double yn)
{
  const double r2 = MIN(1.0, 0.5 * (xn * xn + yn * yn));
  return 1.0 + HDR_ALIGN_ECC_EDGE_WEIGHT * r2;
}

/** Warp an image by a projective homography in backward-mapping form:
 *    sx = (H0*x + H1*y + H2) / (H6*x + H7*y + 1)
 *    sy = (H3*x + H4*y + H5) / (H6*x + H7*y + 1)
 *  Out-of-bounds pixels are set to 0 and mask is set to 0.
 *  Caller must free returned image. */
static float *_warp_homography(const float *in,
                               const int w,
                               const int h,
                               const float H[HDR_ALIGN_H_NPARAM],
                               float *mask)
{
  float *out = dt_alloc_align_float((size_t)w * h);
  if(!out) return NULL;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const float xf = (float)x;
      const float yf = (float)y;
      const float den = H[6] * xf + H[7] * yf + 1.0f;

      if(fabsf(den) < 1e-8f)
      {
        out[y * w + x] = 0.0f;
        if(mask) mask[y * w + x] = 0.0f;
        continue;
      }

      const float sx = (H[0] * xf + H[1] * yf + H[2]) / den;
      const float sy = (H[3] * xf + H[4] * yf + H[5]) / den;

      const int ix = (int)floorf(sx);
      const int iy = (int)floorf(sy);

      if(ix >= 0 && ix < w - 1 && iy >= 0 && iy < h - 1)
      {
        const float fx = sx - (float)ix;
        const float fy = sy - (float)iy;
        out[y * w + x] = (1.0f - fx) * (1.0f - fy) * in[iy * w + ix]
                        + fx * (1.0f - fy) * in[iy * w + ix + 1]
                        + (1.0f - fx) * fy * in[(iy + 1) * w + ix]
                        + fx * fy * in[(iy + 1) * w + ix + 1];
        if(mask) mask[y * w + x] = 1.0f;
      }
      else
      {
        out[y * w + x] = 0.0f;
        if(mask) mask[y * w + x] = 0.0f;
      }
    }

  return out;
}

/** Compute Sobel gradient images (∇x and ∇y).
 *  @p gx and @p gy must be pre-allocated with w*h floats.
 *  Border pixels are set to 0. */
static void _compute_gradients(const float *img,
                               const int w,
                               const int h,
                               float *gx,
                               float *gy)
{
  // Zero the border
  DT_OMP_FOR()
  for(int i = 0; i < w; i++)
  {
    gx[i] = gy[i] = 0.0f;                          // top row
    gx[(h - 1) * w + i] = gy[(h - 1) * w + i] = 0.0f;  // bottom row
  }
  DT_OMP_FOR()
  for(int i = 0; i < h; i++)
  {
    gx[i * w] = gy[i * w] = 0.0f;                  // left column
    gx[i * w + w - 1] = gy[i * w + w - 1] = 0.0f; // right column
  }

  // Sobel 3×3 interior
  DT_OMP_FOR(collapse(2))
  for(int y = 1; y < h - 1; y++)
    for(int x = 1; x < w - 1; x++)
    {
      const float tl = img[(y - 1) * w + (x - 1)];
      const float tc = img[(y - 1) * w + x];
      const float tr = img[(y - 1) * w + (x + 1)];
      const float ml = img[y * w + (x - 1)];
      const float mr = img[y * w + (x + 1)];
      const float bl = img[(y + 1) * w + (x - 1)];
      const float bc = img[(y + 1) * w + x];
      const float br = img[(y + 1) * w + (x + 1)];

      gx[y * w + x] = (-tl + tr - 2.0f * ml + 2.0f * mr - bl + br) / 8.0f;
      gy[y * w + x] = (-tl - 2.0f * tc - tr + bl + 2.0f * bc + br) / 8.0f;
    }
}

/** Normalise a float image to [0, 1] in-place.  This ensures that
 *  gradient magnitudes computed on dark and bright exposures have
 *  comparable numerical scale, preventing ill-conditioned ECC systems. */
static void _normalize_image_01(float *img, const size_t npix)
{
  float vmin = FLT_MAX, vmax = -FLT_MAX;
  for(size_t i = 0; i < npix; i++)
  {
    if(img[i] < vmin) vmin = img[i];
    if(img[i] > vmax) vmax = img[i];
  }
  const float range = vmax - vmin;
  if(range < 1e-12f) return;
  const float inv_range = 1.0f / range;
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    img[i] = (img[i] - vmin) * inv_range;
}

/** Check whether a homography is plausible for hand-held HDR brackets.
 *  Returns TRUE if the normalised H looks like a near-identity rotation
 *  with small translation and negligible perspective. */
static gboolean _homography_is_sane(const float H[HDR_ALIGN_H_NPARAM],
                                    const int w, const int h)
{
  float Hn[HDR_ALIGN_H_NPARAM];
  _homography_pixel_to_normalized(H, w, h, Hn);
  // Diagonal elements should be near 1 (cos of small angle)
  if(Hn[0] < 0.9f || Hn[0] > 1.1f) return FALSE;
  if(Hn[4] < 0.9f || Hn[4] > 1.1f) return FALSE;
  // Off-diagonal should be small (sin of small angle)
  if(fabsf(Hn[1]) > 0.20f) return FALSE;
  if(fabsf(Hn[3]) > 0.20f) return FALSE;
  // Perspective should be negligible
  if(fabsf(Hn[6]) > 0.01f) return FALSE;
  if(fabsf(Hn[7]) > 0.01f) return FALSE;
  // Rotation consistency: diagonals must match (both = cos θ),
  // off-diagonals must be anti-symmetric (sin θ and −sin θ).
  if(fabsf(Hn[0] - Hn[4]) > 0.03f) return FALSE;
  if(fabsf(Hn[1] + Hn[3]) > 0.03f) return FALSE;
  return TRUE;
}

/** Compute gradient magnitude image.  Removes the exposure-dependent
 *  DC component, making the feature image suitable for comparing HDR
 *  brackets with different exposure levels.  Caller must free result. */
static float *_gradient_magnitude(const float *img, const int w, const int h)
{
  const size_t npix = (size_t)w * h;
  float *gx = dt_alloc_align_float(npix);
  float *gy = dt_alloc_align_float(npix);
  if(!gx || !gy)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    return NULL;
  }

  _compute_gradients(img, w, h, gx, gy);

  // Re-use gx buffer for magnitude output
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    gx[i] = sqrtf(gx[i] * gx[i] + gy[i] * gy[i]);

  dt_free_align(gy);
  return gx;
}

/** Solve an 8x8 linear system A·x = b using Gauss-Jordan elimination with
 *  partial pivoting. Returns FALSE if the system is singular. */
static gboolean _solve_8x8(const double A[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM],
                           const double b[HDR_ALIGN_H_NPARAM],
                           double x[HDR_ALIGN_H_NPARAM])
{
  double aug[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM + 1];

  for(int r = 0; r < HDR_ALIGN_H_NPARAM; r++)
  {
    for(int c = 0; c < HDR_ALIGN_H_NPARAM; c++)
      aug[r][c] = A[r][c];
    aug[r][HDR_ALIGN_H_NPARAM] = b[r];
  }

  for(int col = 0; col < HDR_ALIGN_H_NPARAM; col++)
  {
    int piv = col;
    double max_abs = fabs(aug[col][col]);
    for(int r = col + 1; r < HDR_ALIGN_H_NPARAM; r++)
    {
      const double v = fabs(aug[r][col]);
      if(v > max_abs)
      {
        max_abs = v;
        piv = r;
      }
    }

    if(max_abs < 1e-16) return FALSE;

    if(piv != col)
    {
      for(int c = col; c <= HDR_ALIGN_H_NPARAM; c++)
      {
        const double tmp = aug[col][c];
        aug[col][c] = aug[piv][c];
        aug[piv][c] = tmp;
      }
    }

    const double pivot = aug[col][col];
    for(int c = col; c <= HDR_ALIGN_H_NPARAM; c++)
      aug[col][c] /= pivot;

    for(int r = 0; r < HDR_ALIGN_H_NPARAM; r++)
    {
      if(r == col) continue;
      const double f = aug[r][col];
      if(f == 0.0) continue;
      for(int c = col; c <= HDR_ALIGN_H_NPARAM; c++)
        aug[r][c] -= f * aug[col][c];
    }
  }

  for(int r = 0; r < HDR_ALIGN_H_NPARAM; r++)
    x[r] = aug[r][HDR_ALIGN_H_NPARAM];

  return TRUE;
}

/** Solve a dense n×n linear system A·x = b using Gauss-Jordan elimination
 *  with partial pivoting.  n must be <= HDR_ALIGN_H_NPARAM (8).
 *  Optionally returns a rough condition number estimate via the ratio
 *  of the largest to smallest absolute pivot values.
 *  Returns FALSE if the system is singular. */
static gboolean _solve_NxN(const int n,
                            const double A[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM],
                            const double b[HDR_ALIGN_H_NPARAM],
                            double x[HDR_ALIGN_H_NPARAM],
                            double *out_cond_est)
{
  double aug[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM + 1];
  double max_pivot = 0.0, min_pivot = 1e30;

  for(int r = 0; r < n; r++)
  {
    for(int c = 0; c < n; c++)
      aug[r][c] = A[r][c];
    aug[r][n] = b[r];
  }

  for(int col = 0; col < n; col++)
  {
    int piv = col;
    double max_abs = fabs(aug[col][col]);
    for(int r = col + 1; r < n; r++)
    {
      const double v = fabs(aug[r][col]);
      if(v > max_abs)
      {
        max_abs = v;
        piv = r;
      }
    }

    if(max_abs < 1e-16) return FALSE;

    if(piv != col)
    {
      for(int c = col; c <= n; c++)
      {
        const double tmp = aug[col][c];
        aug[col][c] = aug[piv][c];
        aug[piv][c] = tmp;
      }
    }

    const double pivot = aug[col][col];
    const double abs_pivot = fabs(pivot);
    if(abs_pivot > max_pivot) max_pivot = abs_pivot;
    if(abs_pivot < min_pivot) min_pivot = abs_pivot;

    for(int c = col; c <= n; c++) aug[col][c] /= pivot;

    for(int r = 0; r < n; r++)
    {
      if(r == col) continue;
      const double f = aug[r][col];
      if(f == 0.0) continue;
      for(int c = col; c <= n; c++)
        aug[r][c] -= f * aug[col][c];
    }
  }

  for(int r = 0; r < n; r++)
    x[r] = aug[r][n];

  if(out_cond_est)
    *out_cond_est = min_pivot > 1e-30 ? max_pivot / min_pivot : 1e30;

  return TRUE;
}

static void _compose_homographies(const float H_left[HDR_ALIGN_H_NPARAM],
                                  const float H_right[HDR_ALIGN_H_NPARAM],
                                  float H_out[HDR_ALIGN_H_NPARAM])
{
  double L[3][3], R[3][3], O[3][3];
  _homography_to_matrix(H_left, L);
  _homography_to_matrix(H_right, R);
  _mat3_mul(L, R, O);
  _homography_from_matrix(O, H_out);
}

static void _zero_mesh(float *mesh_dx, float *mesh_dy)
{
  for(int i = 0; i < DT_HDR_ALIGN_MESH_NODES; i++)
  {
    mesh_dx[i] = 0.0f;
    mesh_dy[i] = 0.0f;
  }
}

static float _mesh_max_abs(const float *mesh_dx, const float *mesh_dy)
{
  float max_abs = 0.0f;
  for(int i = 0; i < DT_HDR_ALIGN_MESH_NODES; i++)
  {
    max_abs = MAX(max_abs, fabsf(mesh_dx[i]));
    max_abs = MAX(max_abs, fabsf(mesh_dy[i]));
  }
  return max_abs;
}

static float _sample_mesh_grid(const float *mesh,
                               const float u,
                               const float v)
{
  const float gx = CLAMP(u, 0.0f, 1.0f) * (DT_HDR_ALIGN_MESH_COLS - 1);
  const float gy = CLAMP(v, 0.0f, 1.0f) * (DT_HDR_ALIGN_MESH_ROWS - 1);
  const int x0 = MIN((int)floorf(gx), DT_HDR_ALIGN_MESH_COLS - 2);
  const int y0 = MIN((int)floorf(gy), DT_HDR_ALIGN_MESH_ROWS - 2);
  const float fx = gx - (float)x0;
  const float fy = gy - (float)y0;

  const float v00 = mesh[HDR_ALIGN_MESH_INDEX(y0, x0)];
  const float v10 = mesh[HDR_ALIGN_MESH_INDEX(y0, x0 + 1)];
  const float v01 = mesh[HDR_ALIGN_MESH_INDEX(y0 + 1, x0)];
  const float v11 = mesh[HDR_ALIGN_MESH_INDEX(y0 + 1, x0 + 1)];

  return (1.0f - fx) * (1.0f - fy) * v00
       + fx * (1.0f - fy) * v10
       + (1.0f - fx) * fy * v01
       + fx * fy * v11;
}

static float _ncc_patch_shift(const float *ref,
                              const float *warped,
                              const int w,
                              const int h,
                              const int cx,
                              const int cy,
                              const int half_size,
                              const int dx,
                              const int dy)
{
  const int x0 = cx - half_size;
  const int x1 = cx + half_size;
  const int y0 = cy - half_size;
  const int y1 = cy + half_size;

  if(x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) return -2.0f;
  if(x0 + dx < 0 || y0 + dy < 0 || x1 + dx >= w || y1 + dy >= h) return -2.0f;

  const size_t n = (size_t)(x1 - x0 + 1) * (size_t)(y1 - y0 + 1);
  double sum_r = 0.0, sum_w = 0.0;

  for(int y = y0; y <= y1; y++)
    for(int x = x0; x <= x1; x++)
    {
      sum_r += (double)ref[y * w + x];
      sum_w += (double)warped[(y + dy) * w + (x + dx)];
    }

  const double mean_r = sum_r / (double)n;
  const double mean_w = sum_w / (double)n;

  double cross = 0.0, var_r = 0.0, var_w = 0.0;
  for(int y = y0; y <= y1; y++)
    for(int x = x0; x <= x1; x++)
    {
      const double r = (double)ref[y * w + x] - mean_r;
      const double ww = (double)warped[(y + dy) * w + (x + dx)] - mean_w;
      cross += r * ww;
      var_r += r * r;
      var_w += ww * ww;
    }

  const double denom = sqrt(var_r * var_w);
  if(denom < 1e-12) return -2.0f;
  return (float)(cross / denom);
}

static gboolean _find_best_patch_shift(const float *ref,
                                       const float *warped,
                                       const int w,
                                       const int h,
                                       const int cx,
                                       const int cy,
                                       const int half_size,
                                       const int search_radius,
                                       int *best_dx,
                                       int *best_dy,
                                       float *best_ncc,
                                       float *zero_ncc)
{
  *best_dx = 0;
  *best_dy = 0;
  *zero_ncc = _ncc_patch_shift(ref, warped, w, h, cx, cy, half_size, 0, 0);
  *best_ncc = *zero_ncc;

  if(*zero_ncc < -1.0f) return FALSE;

  for(int dy = -search_radius; dy <= search_radius; dy++)
    for(int dx = -search_radius; dx <= search_radius; dx++)
    {
      const float ncc = _ncc_patch_shift(ref, warped, w, h, cx, cy, half_size, dx, dy);
      if(ncc > *best_ncc)
      {
        *best_ncc = ncc;
        *best_dx = dx;
        *best_dy = dy;
      }
    }

  return TRUE;
}

static gboolean _accept_patch_shift(const int best_dx,
                                    const int best_dy,
                                    const int search_radius,
                                    const float best_ncc,
                                    const float zero_ncc)
{
  if(best_ncc < 0.1f || zero_ncc < -1.0f) return FALSE;

  const float improvement = best_ncc - zero_ncc;
  const gboolean boundary_hit = abs(best_dx) == search_radius || abs(best_dy) == search_radius;
  const gboolean tiny_shift = abs(best_dx) <= 1 && abs(best_dy) <= 1;

  if(tiny_shift) return TRUE;
  if(improvement < 0.03f) return FALSE;
  if(boundary_hit && improvement < 0.08f) return FALSE;
  return TRUE;
}

static gboolean _solve_homography_4pt(const float px[4],
                                      const float py[4],
                                      const float qx[4],
                                      const float qy[4],
                                      float H[HDR_ALIGN_H_NPARAM])
{
  double A[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM] = { { 0.0 } };
  double b[HDR_ALIGN_H_NPARAM] = { 0.0 };
  double x[HDR_ALIGN_H_NPARAM];

  for(int i = 0; i < 4; i++)
  {
    const int r0 = 2 * i;
    const int r1 = r0 + 1;
    const double pxi = px[i], pyi = py[i], qxi = qx[i], qyi = qy[i];

    A[r0][0] = pxi; A[r0][1] = pyi; A[r0][2] = 1.0;
    A[r0][6] = -qxi * pxi; A[r0][7] = -qxi * pyi;
    b[r0] = qxi;

    A[r1][3] = pxi; A[r1][4] = pyi; A[r1][5] = 1.0;
    A[r1][6] = -qyi * pxi; A[r1][7] = -qyi * pyi;
    b[r1] = qyi;
  }

  if(!_solve_8x8(A, b, x)) return FALSE;
  for(int i = 0; i < HDR_ALIGN_H_NPARAM; i++) H[i] = (float)x[i];
  return TRUE;
}

static void _corner_refine_level(const float *ref,
                                 const float *img,
                                 const int w,
                                 const int h,
                                 float H[HDR_ALIGN_H_NPARAM])
{
  if(w < 128 || h < 128) return;

  float *warped = _warp_homography(img, w, h, H, NULL);
  if(!warped) return;

  const int patch_half = CLAMP(MIN(w, h) / 18, 20, 72);
  const int search_radius = CLAMP(MIN(w, h) / 96, 3, 12);
  const int inset_x = MAX(w / 10, patch_half + search_radius + 2);
  const int inset_y = MAX(h / 10, patch_half + search_radius + 2);

  if(inset_x >= w - 1 - inset_x || inset_y >= h - 1 - inset_y)
  {
    dt_free_align(warped);
    return;
  }

  const float px[4] = {
    (float)inset_x,
    (float)(w - 1 - inset_x),
    (float)inset_x,
    (float)(w - 1 - inset_x)
  };
  const float py[4] = {
    (float)inset_y,
    (float)inset_y,
    (float)(h - 1 - inset_y),
    (float)(h - 1 - inset_y)
  };
  float qx[4], qy[4];
  float total_shift = 0.0f;

  for(int i = 0; i < 4; i++)
  {
    const int cxi = (int)px[i];
    const int cyi = (int)py[i];
    int best_dx = 0, best_dy = 0;
    float best_ncc = -2.0f, zero_ncc = -2.0f;

    if(!_find_best_patch_shift(ref, warped, w, h, cxi, cyi, patch_half,
                               search_radius, &best_dx, &best_dy,
                               &best_ncc, &zero_ncc)
       || !_accept_patch_shift(best_dx, best_dy, search_radius,
                               best_ncc, zero_ncc))
    {
      dt_free_align(warped);
      return;
    }

    qx[i] = px[i] + 0.75f * (float)best_dx;
    qy[i] = py[i] + 0.75f * (float)best_dy;
    total_shift += fabsf(qx[i] - px[i]) + fabsf(qy[i] - py[i]);
  }

  dt_free_align(warped);

  if(total_shift < 0.5f) return;

  float Hcorr[HDR_ALIGN_H_NPARAM];
  if(!_solve_homography_4pt(px, py, qx, qy, Hcorr)) return;

  float Hnew[HDR_ALIGN_H_NPARAM];
  _compose_homographies(H, Hcorr, Hnew);
  memcpy(H, Hnew, sizeof(float) * HDR_ALIGN_H_NPARAM);
}

static gboolean _estimate_mesh_residuals(const float *ref,
                                         const float *img,
                                         const int w,
                                         const int h,
                                         const float H[HDR_ALIGN_H_NPARAM],
                                         float out_dx[DT_HDR_ALIGN_MESH_NODES],
                                         float out_dy[DT_HDR_ALIGN_MESH_NODES])
{
  _zero_mesh(out_dx, out_dy);

  if(w < 128 || h < 128) return FALSE;

  float *warped = _warp_homography(img, w, h, H, NULL);
  if(!warped) return FALSE;

  const int patch_half = CLAMP(MIN(w, h) / 18, 20, 72);
  const int search_radius = CLAMP(MIN(w, h) / 80, 4, 16);
  const int inset_x = MAX(w / 10, patch_half + search_radius + 2);
  const int inset_y = MAX(h / 10, patch_half + search_radius + 2);

  if(inset_x >= w - 1 - inset_x || inset_y >= h - 1 - inset_y)
  {
    dt_free_align(warped);
    return FALSE;
  }

  const float x0 = (float)inset_x;
  const float x1 = (float)(w - 1 - inset_x);
  const float y0 = (float)inset_y;
  const float y1 = (float)(h - 1 - inset_y);
  float raw_dx[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float raw_dy[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float cur_dx[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float cur_dy[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float next_dx[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float next_dy[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  float data_weight[DT_HDR_ALIGN_MESH_NODES] = { 0.0f };
  int valid_count = 0;

  for(int row = 0; row < DT_HDR_ALIGN_MESH_ROWS; row++)
  {
    const float vy = DT_HDR_ALIGN_MESH_ROWS > 1 ? (float)row / (DT_HDR_ALIGN_MESH_ROWS - 1) : 0.0f;
    const int py = (int)lrintf(y0 + (y1 - y0) * vy);
    for(int col = 0; col < DT_HDR_ALIGN_MESH_COLS; col++)
    {
      const float vx = DT_HDR_ALIGN_MESH_COLS > 1 ? (float)col / (DT_HDR_ALIGN_MESH_COLS - 1) : 0.0f;
      const int px = (int)lrintf(x0 + (x1 - x0) * vx);
      const int idx = HDR_ALIGN_MESH_INDEX(row, col);
      int best_dx = 0, best_dy = 0;
      float best_ncc = -2.0f, zero_ncc = -2.0f;

      if(!_find_best_patch_shift(ref, warped, w, h, px, py, patch_half,
                                 search_radius, &best_dx, &best_dy,
                                 &best_ncc, &zero_ncc))
        continue;

      if(_accept_patch_shift(best_dx, best_dy, search_radius, best_ncc, zero_ncc))
      {
        raw_dx[idx] = 0.75f * (float)best_dx;
        raw_dy[idx] = 0.75f * (float)best_dy;
        cur_dx[idx] = raw_dx[idx];
        cur_dy[idx] = raw_dy[idx];
        data_weight[idx] = 1.0f + 8.0f * (best_ncc - zero_ncc) + 2.0f * MAX(0.0f, best_ncc - 0.3f);
        valid_count++;
      }
      else if(zero_ncc >= 0.1f)
      {
        // Low-texture or ambiguous patches should resist drift rather than
        // injecting arbitrary local motion into smooth areas such as skies.
        raw_dx[idx] = 0.0f;
        raw_dy[idx] = 0.0f;
        cur_dx[idx] = 0.0f;
        cur_dy[idx] = 0.0f;
        data_weight[idx] = 1.0f + 2.0f * MAX(0.0f, zero_ncc - 0.2f);
      }
    }
  }

  dt_free_align(warped);

  if(valid_count < 4) return FALSE;

  for(int iter = 0; iter < HDR_ALIGN_MESH_SMOOTH_ITERS; iter++)
  {
    for(int row = 0; row < DT_HDR_ALIGN_MESH_ROWS; row++)
      for(int col = 0; col < DT_HDR_ALIGN_MESH_COLS; col++)
      {
        const int idx = HDR_ALIGN_MESH_INDEX(row, col);
        double sum_dx = data_weight[idx] * raw_dx[idx];
        double sum_dy = data_weight[idx] * raw_dy[idx];
        double sum_w = data_weight[idx];

        if(col > 0)
        {
          const int nidx = HDR_ALIGN_MESH_INDEX(row, col - 1);
          sum_dx += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dx[nidx];
          sum_dy += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dy[nidx];
          sum_w += HDR_ALIGN_MESH_SMOOTH_LAMBDA;
        }
        if(col + 1 < DT_HDR_ALIGN_MESH_COLS)
        {
          const int nidx = HDR_ALIGN_MESH_INDEX(row, col + 1);
          sum_dx += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dx[nidx];
          sum_dy += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dy[nidx];
          sum_w += HDR_ALIGN_MESH_SMOOTH_LAMBDA;
        }
        if(row > 0)
        {
          const int nidx = HDR_ALIGN_MESH_INDEX(row - 1, col);
          sum_dx += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dx[nidx];
          sum_dy += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dy[nidx];
          sum_w += HDR_ALIGN_MESH_SMOOTH_LAMBDA;
        }
        if(row + 1 < DT_HDR_ALIGN_MESH_ROWS)
        {
          const int nidx = HDR_ALIGN_MESH_INDEX(row + 1, col);
          sum_dx += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dx[nidx];
          sum_dy += HDR_ALIGN_MESH_SMOOTH_LAMBDA * cur_dy[nidx];
          sum_w += HDR_ALIGN_MESH_SMOOTH_LAMBDA;
        }

        if(sum_w > 0.0)
        {
          next_dx[idx] = (float)(sum_dx / sum_w);
          next_dy[idx] = (float)(sum_dy / sum_w);
        }
        else
        {
          next_dx[idx] = 0.0f;
          next_dy[idx] = 0.0f;
        }
      }

    memcpy(cur_dx, next_dx, sizeof(cur_dx));
    memcpy(cur_dy, next_dy, sizeof(cur_dy));
  }

  memcpy(out_dx, cur_dx, sizeof(cur_dx));
  memcpy(out_dy, cur_dy, sizeof(cur_dy));
  return _mesh_max_abs(out_dx, out_dy) > 0.0f;
}

/** One iteration of forward-additive ECC for a native 3-DOF Euclidean model
 *  (rotation θ about the image centre + translation tx, ty).
 *
 *  Builds a 3×3 Hessian directly in the (dθ, dtx, dty) parameter space
 *  so no energy leaks into shear / scale / perspective DOFs.  This is
 *  the correct formulation for hand-held HDR brackets where the true
 *  camera motion is a planar rotation plus small translation. */
static float _ecc_iteration(const float *ref,
                            const float *img,
                            const int w,
                            const int h,
                            float H[HDR_ALIGN_H_NPARAM])
{
  const size_t npix = (size_t)w * h;
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;

  // Extract current angle from the normalised homography.
  float Hn[HDR_ALIGN_H_NPARAM];
  _homography_pixel_to_normalized(H, w, h, Hn);
  const double cos_t = (double)Hn[0];
  const double sin_t = (double)Hn[1];

  // Warp image by current H.
  float *mask = dt_alloc_align_float(npix);
  if(!mask) return -1.0f;

  float *warped = _warp_homography(img, w, h, H, mask);
  if(!warped)
  {
    dt_free_align(mask);
    return -1.0f;
  }

  float *gx = dt_alloc_align_float(npix);
  float *gy = dt_alloc_align_float(npix);
  if(!gx || !gy)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    dt_free_align(warped);
    dt_free_align(mask);
    return -1.0f;
  }
  _compute_gradients(warped, w, h, gx, gy);

  // --- Pass 1: weighted means ---
  double sum_r = 0.0, sum_w = 0.0;
  double sum_weight = 0.0;
  long nvalid = 0;
  DT_OMP_FOR(collapse(2) reduction(+:sum_r, sum_w, sum_weight, nvalid))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] > 0.5f)
      {
        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        sum_r += wgt * (double)ref[i];
        sum_w += wgt * (double)warped[i];
        sum_weight += wgt;
        nvalid++;
      }
    }

  if(nvalid < (long)(npix * HDR_ALIGN_ECC_MIN_VALID_FRAC))
  {
    dt_free_align(gx);
    dt_free_align(gy);
    dt_free_align(warped);
    dt_free_align(mask);
    return -1.0f;
  }

  const double mean_r = sum_r / sum_weight;
  const double mean_w = sum_w / sum_weight;

  // --- Pass 2: norms, mean Jacobian, and correlation coefficient ---
  // 3-DOF Jacobian: J = (J_θ, J_tx, J_ty)
  //   J_θ  = s·(gx·(−sin θ·xn + cos θ·yn) + gy·(−cos θ·xn − sin θ·yn))
  //   J_tx = s·gx
  //   J_ty = s·gy
  // We merge the old pass 2 (norms + sum_J) and old correlation coefficient
  // pass into a single parallel pass to reduce memory bandwidth overhead.
  double norm2_r = 0.0, norm2_w = 0.0;
  double sum_J0 = 0.0, sum_J1 = 0.0, sum_J2 = 0.0;
  double dot_rw = 0.0;

  DT_OMP_FOR(collapse(2) reduction(+:norm2_r, norm2_w, sum_J0, sum_J1, sum_J2, dot_rw))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] < 0.5f) continue;

      const double xn = ((double)x - cx) / s;
      const double yn = ((double)y - cy) / s;
      const double wgt = _ecc_spatial_weight(xn, yn);

      const double gxi = (double)gx[i];
      const double gyi = (double)gy[i];

      const double J0 = s * (gxi * (-sin_t * xn + cos_t * yn)
                            + gyi * (-cos_t * xn - sin_t * yn));
      const double J1 = s * gxi;
      const double J2 = s * gyi;

      const double r  = (double)ref[i] - mean_r;
      const double tw = (double)warped[i] - mean_w;

      norm2_r += wgt * r * r;
      norm2_w += wgt * tw * tw;
      dot_rw  += wgt * r * tw;

      sum_J0 += wgt * J0;
      sum_J1 += wgt * J1;
      sum_J2 += wgt * J2;
    }

  const double norm_r = sqrt(norm2_r);
  const double norm_w = sqrt(norm2_w);
  if(norm_r < 1e-12 || norm_w < 1e-12)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    dt_free_align(warped);
    dt_free_align(mask);
    return -1.0f;
  }

  const double rho = dot_rw / (norm_r * norm_w);

  // --- Pass 3: projection coefficient ---
  const double mean_J[3] = { sum_J0 / sum_weight,
                              sum_J1 / sum_weight,
                              sum_J2 / sum_weight };

  double proj0 = 0.0, proj1 = 0.0, proj2 = 0.0;
  DT_OMP_FOR(collapse(2) reduction(+:proj0, proj1, proj2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] < 0.5f) continue;

      const double xn = ((double)x - cx) / s;
      const double yn = ((double)y - cy) / s;
      const double wgt = _ecc_spatial_weight(xn, yn);
      const double gxi = (double)gx[i];
      const double gyi = (double)gy[i];

      const double J[3] = {
        s * (gxi * (-sin_t * xn + cos_t * yn)
           + gyi * (-cos_t * xn - sin_t * yn)),
        s * gxi,
        s * gyi
      };

      const double tw = (double)warped[i] - mean_w;
      proj0 += wgt * tw * (J[0] - mean_J[0]);
      proj1 += wgt * tw * (J[1] - mean_J[1]);
      proj2 += wgt * tw * (J[2] - mean_J[2]);
    }

  const double proj_coeff[3] = { proj0 / norm2_w,
                                  proj1 / norm2_w,
                                  proj2 / norm2_w };

  // --- Pass 4: 3×3 Hessian and RHS ---
  // Use scalar accumulators for OMP reduction (arrays not directly
  // reducible with the DT_OMP_FOR macro).
  double H00 = 0.0, H01 = 0.0, H02 = 0.0;
  double H11 = 0.0, H12 = 0.0, H22 = 0.0;
  double rhs0 = 0.0, rhs1 = 0.0, rhs2 = 0.0;
  const double scale_rw = norm_w / norm_r;

  DT_OMP_FOR(collapse(2) reduction(+:H00, H01, H02, H11, H12, H22, rhs0, rhs1, rhs2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] < 0.5f) continue;

      const double xn = ((double)x - cx) / s;
      const double yn = ((double)y - cy) / s;
      const double wgt = _ecc_spatial_weight(xn, yn);
      const double gxi = (double)gx[i];
      const double gyi = (double)gy[i];

      const double J[3] = {
        s * (gxi * (-sin_t * xn + cos_t * yn)
           + gyi * (-cos_t * xn - sin_t * yn)),
        s * gxi,
        s * gyi
      };

      const double tw = (double)warped[i] - mean_w;
      const double r  = (double)ref[i] - mean_r;
      const double ei = scale_rw * r - rho * tw;

      const double Jp0 = (J[0] - mean_J[0]) - proj_coeff[0] * tw;
      const double Jp1 = (J[1] - mean_J[1]) - proj_coeff[1] * tw;
      const double Jp2 = (J[2] - mean_J[2]) - proj_coeff[2] * tw;

      rhs0 += wgt * Jp0 * ei;
      rhs1 += wgt * Jp1 * ei;
      rhs2 += wgt * Jp2 * ei;

      H00 += wgt * Jp0 * Jp0;
      H01 += wgt * Jp0 * Jp1;
      H02 += wgt * Jp0 * Jp2;
      H11 += wgt * Jp1 * Jp1;
      H12 += wgt * Jp1 * Jp2;
      H22 += wgt * Jp2 * Jp2;
    }

  // Assemble symmetric Hessian and RHS from scalar accumulators.
  double Hess[3][3] = { { H00, H01, H02 },
                         { H01, H11, H12 },
                         { H02, H12, H22 } };
  double rhs_ecc[3] = { rhs0, rhs1, rhs2 };

  dt_free_align(gx);
  dt_free_align(gy);
  dt_free_align(warped);
  dt_free_align(mask);

  // Tikhonov regularization on the 3×3 Hessian.
  {
    const double trace = Hess[0][0] + Hess[1][1] + Hess[2][2];
    const double lambda = 0.01 * trace / 3.0;
    for(int k = 0; k < 3; k++)
      Hess[k][k] += lambda;
  }

  // Solve 3×3 using Gauss-Jordan with partial pivoting.
  double dp[3];
  {
    double aug[3][4];
    for(int r = 0; r < 3; r++)
    {
      for(int c = 0; c < 3; c++) aug[r][c] = Hess[r][c];
      aug[r][3] = rhs_ecc[r];
    }

    for(int col = 0; col < 3; col++)
    {
      int piv = col;
      double best = fabs(aug[col][col]);
      for(int r = col + 1; r < 3; r++)
      {
        const double v = fabs(aug[r][col]);
        if(v > best) { best = v; piv = r; }
      }
      if(best < 1e-16) return -1.0f;

      if(piv != col)
        for(int c = col; c <= 3; c++)
        {
          const double tmp = aug[col][c];
          aug[col][c] = aug[piv][c];
          aug[piv][c] = tmp;
        }

      const double pivot = aug[col][col];
      for(int c = col; c <= 3; c++) aug[col][c] /= pivot;

      for(int r = 0; r < 3; r++)
      {
        if(r == col) continue;
        const double f = aug[r][col];
        if(f == 0.0) continue;
        for(int c = col; c <= 3; c++) aug[r][c] -= f * aug[col][c];
      }
    }
    for(int r = 0; r < 3; r++) dp[r] = aug[r][3];
  }

  double dtheta = dp[0];
  double dtx    = dp[1];
  double dty    = dp[2];

  // Clamp: max ~0.57° rotation per step, max 10% translation per step.
  const double max_dtheta = 0.01;
  const double max_shift_n = 0.10;
  dtheta = CLAMP(dtheta, -max_dtheta, max_dtheta);
  dtx = CLAMP(dtx, -max_shift_n, max_shift_n);
  dty = CLAMP(dty, -max_shift_n, max_shift_n);

  // Apply Euclidean update exactly (no linearisation error).
  const double theta_old = atan2(sin_t, cos_t);
  const double theta_new = theta_old + dtheta;

  Hn[0] = (float)cos(theta_new);
  Hn[1] = (float)sin(theta_new);
  Hn[2] += (float)dtx;
  Hn[3] = -(float)sin(theta_new);
  Hn[4] = (float)cos(theta_new);
  Hn[5] += (float)dty;
  Hn[6] = 0.0f;
  Hn[7] = 0.0f;

  _homography_normalized_to_pixel(Hn, w, h, H);

  // Convergence metric: pixel-equivalent displacement.
  const double trans_pix = s * (fabs(dtx) + fabs(dty));
  const double angle_pix = s * fabs(dtheta);

  return (float)(trans_pix + angle_pix);
}

/** Run iterative ECC at a single pyramid level until convergence. */
static float _ecc_refine_level(const float *ref,
                               const float *img,
                               const int w,
                               const int h,
                               float H[HDR_ALIGN_H_NPARAM])
{
  float best_update = FLT_MAX;
  int stall_count = 0;

  // Keep a copy of H at the iteration that produced the best update metric
  // so we can restore it on stall, max-iterations, or failure exit.
  // Without this, H is left in the last-iteration state -- which by
  // definition had a *worse* update than the best -- and that drifted
  // homography poisons every finer pyramid level.
  float H_best[HDR_ALIGN_H_NPARAM];
  memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);

  for(int iter = 0; iter < HDR_ALIGN_ECC_MAX_ITER; iter++)
  {
    const float update = _ecc_iteration(ref, img, w, h, H);

    if(update < 0.0f)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   ECC failed at iteration %d", iter);
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      return -1.0f;
    }

    if(update < HDR_ALIGN_ECC_EPSILON)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   ECC converged at iteration %d (update=%.6f)",
               iter, update);
      return update;
    }

    // Plateau detection: if the update metric has not improved for several
    // consecutive iterations the optimiser is stuck above the convergence
    // threshold — continuing will not help.
    if(update < best_update)
    {
      best_update = update;
      stall_count = 0;
      memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);
    }
    else if(++stall_count >= HDR_ALIGN_ECC_PATIENCE)
    {
      // Restore H to the state that gave the best update so far.
      // The current H corresponds to a worse iteration and must not be
      // used as the starting point for the next pyramid level.
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   ECC stalled at iteration %d (update=%.6f, best=%.6f)",
               iter, update, best_update);
      return best_update;
    }
  }

  // Max iterations reached: restore best H seen so far.
  memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge]   ECC did not converge in %d iterations",
           HDR_ALIGN_ECC_MAX_ITER);
  return 0.0f;
}

// ---------------------------------------------------------------------------
// Adaptive DOF escalation: 3-DOF → 6-DOF → 8-DOF
// ---------------------------------------------------------------------------

/** Compute the weighted ECC score ρ for a given homography on gradient-magnitude
 *  images.  Returns the correlation coefficient in [-1, 1], or -2 on failure. */
static float _ecc_compute_rho(const float *ref,
                               const float *img,
                               const int w,
                               const int h,
                               const float H[HDR_ALIGN_H_NPARAM])
{
  const size_t npix = (size_t)w * h;
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;

  float *mask = dt_alloc_align_float(npix);
  if(!mask) return -2.0f;

  float *warped = _warp_homography(img, w, h, H, mask);
  if(!warped)
  {
    dt_free_align(mask);
    return -2.0f;
  }

  double sum_r = 0.0, sum_w = 0.0, sum_weight = 0.0;
  long nvalid = 0;

  DT_OMP_FOR(collapse(2) reduction(+:sum_r, sum_w, sum_weight, nvalid))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] > 0.5f)
      {
        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        sum_r += wgt * (double)ref[i];
        sum_w += wgt * (double)warped[i];
        sum_weight += wgt;
        nvalid++;
      }
    }

  if(nvalid < (long)(npix * HDR_ALIGN_ECC_MIN_VALID_FRAC) || sum_weight < 1e-12)
  {
    dt_free_align(warped);
    dt_free_align(mask);
    return -2.0f;
  }

  const double mean_r = sum_r / sum_weight;
  const double mean_w = sum_w / sum_weight;

  double norm2_r = 0.0, norm2_w = 0.0, dot_rw = 0.0;
  DT_OMP_FOR(collapse(2) reduction(+:norm2_r, norm2_w, dot_rw))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] > 0.5f)
      {
        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        const double r = (double)ref[i] - mean_r;
        const double tw = (double)warped[i] - mean_w;
        norm2_r += wgt * r * r;
        norm2_w += wgt * tw * tw;
        dot_rw += wgt * r * tw;
      }
    }

  dt_free_align(warped);
  dt_free_align(mask);

  const double denom = sqrt(norm2_r * norm2_w);
  if(denom < 1e-12) return -2.0f;
  return (float)(dot_rw / denom);
}

/** One iteration of forward-additive ECC for a higher-DOF model.
 *  @p ndof selects the model:
 *    6 → affine (H[0..5] free, H[6]=H[7]=0)
 *    8 → full projective (H[0..7] free)
 *
 *  Returns the pixel-equivalent parameter update norm, or -1 on failure.
 *  @p out_cond_est receives a rough condition number estimate of the Hessian
 *  (pass NULL if not needed). */
static float _ecc_iteration_higher_dof(const float *ref,
                                        const float *img,
                                        const int w,
                                        const int h,
                                        float H[HDR_ALIGN_H_NPARAM],
                                        const int ndof,
                                        double *out_cond_est)
{
  if(ndof != 6 && ndof != 8) return -1.0f;

  const size_t npix = (size_t)w * h;
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;

  float Hn[HDR_ALIGN_H_NPARAM];
  _homography_pixel_to_normalized(H, w, h, Hn);

  float *mask = dt_alloc_align_float(npix);
  if(!mask) return -1.0f;

  float *warped = _warp_homography(img, w, h, H, mask);
  if(!warped)
  {
    dt_free_align(mask);
    return -1.0f;
  }

  float *gx = dt_alloc_align_float(npix);
  float *gy = dt_alloc_align_float(npix);
  if(!gx || !gy)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    dt_free_align(warped);
    dt_free_align(mask);
    return -1.0f;
  }
  _compute_gradients(warped, w, h, gx, gy);

  // --- Pass 1: weighted means ---
  double sum_r = 0.0, sum_w = 0.0, sum_weight = 0.0;
  long nvalid = 0;
  DT_OMP_FOR(collapse(2) reduction(+:sum_r, sum_w, sum_weight, nvalid))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] > 0.5f)
      {
        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        sum_r += wgt * (double)ref[i];
        sum_w += wgt * (double)warped[i];
        sum_weight += wgt;
        nvalid++;
      }
    }

  if(nvalid < (long)(npix * HDR_ALIGN_ECC_MIN_VALID_FRAC))
  {
    dt_free_align(gx);  dt_free_align(gy);
    dt_free_align(warped);  dt_free_align(mask);
    return -1.0f;
  }

  const double mean_r = sum_r / sum_weight;
  const double mean_w = sum_w / sum_weight;

  // --- Pass 2: norms, mean Jacobian, and correlation coefficient ---
  // We merge old pass 2 and the correlation pass into one parallel sweep.
  // Use scalar accumulators for OMP reduction (up to 8 DOF).
  double norm2_r = 0.0, norm2_w = 0.0, dot_rw = 0.0;
  double sJ0 = 0.0, sJ1 = 0.0, sJ2 = 0.0, sJ3 = 0.0;
  double sJ4 = 0.0, sJ5 = 0.0, sJ6 = 0.0, sJ7 = 0.0;

  DT_OMP_FOR(collapse(2) reduction(+:norm2_r, norm2_w, dot_rw, sJ0, sJ1, sJ2, sJ3, sJ4, sJ5, sJ6, sJ7))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      if(mask[i] < 0.5f) continue;

      const double xn = ((double)x - cx) / s;
      const double yn = ((double)y - cy) / s;
      const double wgt = _ecc_spatial_weight(xn, yn);
      const double gxi = (double)gx[i];
      const double gyi = (double)gy[i];

      double J[HDR_ALIGN_H_NPARAM];
      if(ndof == 6)
      {
        J[0] = s * gxi * xn;
        J[1] = s * gxi * yn;
        J[2] = s * gxi;
        J[3] = s * gyi * xn;
        J[4] = s * gyi * yn;
        J[5] = s * gyi;
        J[6] = 0.0;
        J[7] = 0.0;
      }
      else
      {
        const double d = (double)Hn[6] * xn + (double)Hn[7] * yn + 1.0;
        const double id = (fabs(d) > 1e-12) ? 1.0 / d : 0.0;
        const double sxn = (double)Hn[0] * xn + (double)Hn[1] * yn + (double)Hn[2];
        const double syn = (double)Hn[3] * xn + (double)Hn[4] * yn + (double)Hn[5];
        J[0] = s * gxi * xn * id;
        J[1] = s * gxi * yn * id;
        J[2] = s * gxi * id;
        J[3] = s * gyi * xn * id;
        J[4] = s * gyi * yn * id;
        J[5] = s * gyi * id;
        J[6] = -s * (gxi * sxn + gyi * syn) * xn * id * id;
        J[7] = -s * (gxi * sxn + gyi * syn) * yn * id * id;
      }

      const double r  = (double)ref[i] - mean_r;
      const double tw = (double)warped[i] - mean_w;
      norm2_r += wgt * r * r;
      norm2_w += wgt * tw * tw;
      dot_rw  += wgt * r * tw;

      sJ0 += wgt * J[0];  sJ1 += wgt * J[1];
      sJ2 += wgt * J[2];  sJ3 += wgt * J[3];
      sJ4 += wgt * J[4];  sJ5 += wgt * J[5];
      sJ6 += wgt * J[6];  sJ7 += wgt * J[7];
    }

  const double norm_r = sqrt(norm2_r);
  const double norm_w = sqrt(norm2_w);
  if(norm_r < 1e-12 || norm_w < 1e-12)
  {
    dt_free_align(gx);  dt_free_align(gy);
    dt_free_align(warped);  dt_free_align(mask);
    return -1.0f;
  }

  const double rho = dot_rw / (norm_r * norm_w);
  const double sum_J[HDR_ALIGN_H_NPARAM] = { sJ0, sJ1, sJ2, sJ3, sJ4, sJ5, sJ6, sJ7 };

  // --- Pass 3: projection coefficient ---
  double mean_J[HDR_ALIGN_H_NPARAM];
  for(int k = 0; k < ndof; k++)
    mean_J[k] = sum_J[k] / sum_weight;

  double proj_coeff[HDR_ALIGN_H_NPARAM] = { 0 };
  {
    double pc0 = 0.0, pc1 = 0.0, pc2 = 0.0, pc3 = 0.0;
    double pc4 = 0.0, pc5 = 0.0, pc6 = 0.0, pc7 = 0.0;
    DT_OMP_FOR(collapse(2) reduction(+:pc0, pc1, pc2, pc3, pc4, pc5, pc6, pc7))
    for(int y = 0; y < h; y++)
      for(int x = 0; x < w; x++)
      {
        const size_t i = (size_t)y * w + x;
        if(mask[i] < 0.5f) continue;

        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        const double gxi = (double)gx[i];
        const double gyi = (double)gy[i];

        double J[HDR_ALIGN_H_NPARAM];
        if(ndof == 6)
        {
          J[0] = s * gxi * xn;  J[1] = s * gxi * yn;  J[2] = s * gxi;
          J[3] = s * gyi * xn;  J[4] = s * gyi * yn;  J[5] = s * gyi;
          J[6] = 0.0;  J[7] = 0.0;
        }
        else
        {
          const double d = (double)Hn[6] * xn + (double)Hn[7] * yn + 1.0;
          const double id = (fabs(d) > 1e-12) ? 1.0 / d : 0.0;
          const double sxn = (double)Hn[0] * xn + (double)Hn[1] * yn + (double)Hn[2];
          const double syn = (double)Hn[3] * xn + (double)Hn[4] * yn + (double)Hn[5];
          J[0] = s * gxi * xn * id;  J[1] = s * gxi * yn * id;  J[2] = s * gxi * id;
          J[3] = s * gyi * xn * id;  J[4] = s * gyi * yn * id;  J[5] = s * gyi * id;
          J[6] = -s * (gxi * sxn + gyi * syn) * xn * id * id;
          J[7] = -s * (gxi * sxn + gyi * syn) * yn * id * id;
        }

        const double tw = (double)warped[i] - mean_w;
        pc0 += wgt * tw * (J[0] - mean_J[0]);
        pc1 += wgt * tw * (J[1] - mean_J[1]);
        pc2 += wgt * tw * (J[2] - mean_J[2]);
        pc3 += wgt * tw * (J[3] - mean_J[3]);
        pc4 += wgt * tw * (J[4] - mean_J[4]);
        pc5 += wgt * tw * (J[5] - mean_J[5]);
        pc6 += wgt * tw * (J[6] - mean_J[6]);
        pc7 += wgt * tw * (J[7] - mean_J[7]);
      }
    proj_coeff[0] = pc0 / norm2_w;  proj_coeff[1] = pc1 / norm2_w;
    proj_coeff[2] = pc2 / norm2_w;  proj_coeff[3] = pc3 / norm2_w;
    proj_coeff[4] = pc4 / norm2_w;  proj_coeff[5] = pc5 / norm2_w;
    proj_coeff[6] = pc6 / norm2_w;  proj_coeff[7] = pc7 / norm2_w;
  }

  // --- Pass 4: n×n Hessian and RHS ---
  // For OMP reduction, we use 36 scalar accumulators for the upper triangle
  // of the symmetric 8×8 Hessian (H[a][b] with a<=b) plus 8 for RHS.
  // We store them in a flat array and reduce in a critical section per thread.
  double Hess[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM] = { { 0 } };
  double rhs_ecc[HDR_ALIGN_H_NPARAM] = { 0 };
  const double scale_rw = norm_w / norm_r;

  // For the Hessian, we use thread-local accumulation with a critical merge.
  // This avoids needing 36+ scalar reduction variables.
  DT_OMP_PRAGMA(parallel default(firstprivate) shared(Hess, rhs_ecc))
  {
    double Hess_local[HDR_ALIGN_H_NPARAM][HDR_ALIGN_H_NPARAM] = { { 0 } };
    double rhs_local[HDR_ALIGN_H_NPARAM] = { 0 };

    DT_OMP_PRAGMA(for collapse(2) schedule(static))
    for(int y = 0; y < h; y++)
      for(int x = 0; x < w; x++)
      {
        const size_t i = (size_t)y * w + x;
        if(mask[i] < 0.5f) continue;

        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        const double gxi = (double)gx[i];
        const double gyi = (double)gy[i];

        double J[HDR_ALIGN_H_NPARAM];
        if(ndof == 6)
        {
          J[0] = s * gxi * xn;  J[1] = s * gxi * yn;  J[2] = s * gxi;
          J[3] = s * gyi * xn;  J[4] = s * gyi * yn;  J[5] = s * gyi;
          J[6] = 0.0;  J[7] = 0.0;
        }
        else
        {
          const double d = (double)Hn[6] * xn + (double)Hn[7] * yn + 1.0;
          const double id = (fabs(d) > 1e-12) ? 1.0 / d : 0.0;
          const double sxn = (double)Hn[0] * xn + (double)Hn[1] * yn + (double)Hn[2];
          const double syn = (double)Hn[3] * xn + (double)Hn[4] * yn + (double)Hn[5];
          J[0] = s * gxi * xn * id;  J[1] = s * gxi * yn * id;  J[2] = s * gxi * id;
          J[3] = s * gyi * xn * id;  J[4] = s * gyi * yn * id;  J[5] = s * gyi * id;
          J[6] = -s * (gxi * sxn + gyi * syn) * xn * id * id;
          J[7] = -s * (gxi * sxn + gyi * syn) * yn * id * id;
        }

        const double tw = (double)warped[i] - mean_w;
        const double r  = (double)ref[i] - mean_r;
        const double ei = scale_rw * r - rho * tw;

        double Jp[HDR_ALIGN_H_NPARAM];
        for(int k = 0; k < ndof; k++)
          Jp[k] = (J[k] - mean_J[k]) - proj_coeff[k] * tw;

        for(int a = 0; a < ndof; a++)
        {
          rhs_local[a] += wgt * Jp[a] * ei;
          for(int b2 = a; b2 < ndof; b2++)
            Hess_local[a][b2] += wgt * Jp[a] * Jp[b2];
        }
      }

    DT_OMP_PRAGMA(critical)
    {
      for(int a = 0; a < ndof; a++)
      {
        rhs_ecc[a] += rhs_local[a];
        for(int b2 = a; b2 < ndof; b2++)
          Hess[a][b2] += Hess_local[a][b2];
      }
    }
  }

  // Fill lower triangle from upper triangle (symmetric).
  for(int a = 0; a < ndof; a++)
    for(int b2 = 0; b2 < a; b2++)
      Hess[a][b2] = Hess[b2][a];

  dt_free_align(gx);  dt_free_align(gy);
  dt_free_align(warped);  dt_free_align(mask);

  // Tikhonov regularization.
  {
    double trace = 0.0;
    for(int k = 0; k < ndof; k++) trace += Hess[k][k];
    const double lambda = 0.01 * trace / (double)ndof;
    for(int k = 0; k < ndof; k++)
      Hess[k][k] += lambda;
  }

  // Solve n×n system.
  double dp[HDR_ALIGN_H_NPARAM] = { 0 };
  double cond_est = 0.0;
  if(!_solve_NxN(ndof, Hess, rhs_ecc, dp, &cond_est))
    return -1.0f;

  if(out_cond_est)
    *out_cond_est = cond_est;

  // Clamp updates to prevent catastrophic steps.
  // Affine diagonal/off-diagonal: max 0.02 per step.
  // Translation: max 10% of half-diagonal per step.
  // Perspective: max 0.001 per step.
  const double max_affine = 0.02;
  const double max_shift_n = 0.10;
  const double max_persp = 0.001;

  dp[0] = CLAMP(dp[0], -max_affine, max_affine);
  dp[1] = CLAMP(dp[1], -max_affine, max_affine);
  dp[2] = CLAMP(dp[2], -max_shift_n, max_shift_n);
  dp[3] = CLAMP(dp[3], -max_affine, max_affine);
  dp[4] = CLAMP(dp[4], -max_affine, max_affine);
  dp[5] = CLAMP(dp[5], -max_shift_n, max_shift_n);
  if(ndof == 8)
  {
    dp[6] = CLAMP(dp[6], -max_persp, max_persp);
    dp[7] = CLAMP(dp[7], -max_persp, max_persp);
  }

  // Additive update in normalized coordinates.
  for(int k = 0; k < ndof; k++)
    Hn[k] += (float)dp[k];

  // Force model constraints.
  if(ndof == 6)
  {
    Hn[6] = 0.0f;
    Hn[7] = 0.0f;
  }

  _homography_normalized_to_pixel(Hn, w, h, H);

  // Convergence metric: pixel-equivalent displacement of the update.
  double update_metric = 0.0;
  for(int k = 0; k < ndof; k++)
    update_metric += fabs(dp[k]);
  update_metric *= s;

  return (float)update_metric;
}

/** Run iterative higher-DOF ECC at a single pyramid level.
 *  @p ndof must be 6 (affine) or 8 (projective).
 *  Returns the final ρ after refinement, or -2 on failure. */
static float _ecc_refine_level_higher_dof(const float *ref,
                                            const float *img,
                                            const int w,
                                            const int h,
                                            float H[HDR_ALIGN_H_NPARAM],
                                            const int ndof)
{
  for(int iter = 0; iter < HDR_ALIGN_ESCALATION_MAX_ITER; iter++)
  {
    double cond_est = 0.0;
    const float update = _ecc_iteration_higher_dof(ref, img, w, h, H,
                                                    ndof, &cond_est);

    if(update < 0.0f)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   %d-DOF ECC failed at iteration %d",
               ndof, iter);
      return -2.0f;
    }

    // Reject if the Hessian is poorly conditioned.
    if(cond_est > HDR_ALIGN_ESCALATION_MAX_COND)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   %d-DOF ECC rejected: Hessian cond=%.1e > %.1e",
               ndof, cond_est, HDR_ALIGN_ESCALATION_MAX_COND);
      return -2.0f;
    }

    if(update < HDR_ALIGN_ESCALATION_EPSILON)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge]   %d-DOF ECC converged at iteration %d (update=%.6f, cond=%.1e)",
               ndof, iter, update, cond_est);
      break;
    }
  }

  return _ecc_compute_rho(ref, img, w, h, H);
}

/** Sanity check for an escalated homography.  Looser than _homography_is_sane
 *  because the higher-DOF model intentionally adds scale / shear / perspective
 *  parameters, but the result must still be physically plausible. */
static gboolean _homography_is_sane_escalated(const float H[HDR_ALIGN_H_NPARAM],
                                               const int w, const int h,
                                               const int ndof)
{
  float Hn[HDR_ALIGN_H_NPARAM];
  _homography_pixel_to_normalized(H, w, h, Hn);

  // Diagonal elements should be near 1 (allows some scale).
  if(Hn[0] < 0.85f || Hn[0] > 1.15f) return FALSE;
  if(Hn[4] < 0.85f || Hn[4] > 1.15f) return FALSE;

  // Off-diagonal: allow more shear than the rigid check.
  if(fabsf(Hn[1]) > 0.25f) return FALSE;
  if(fabsf(Hn[3]) > 0.25f) return FALSE;

  // Area scaling: the determinant of the affine part (upper-left 2×2 of the
  // normalised matrix) equals the local area magnification at image centre.
  // For HDR brackets from the same camera/lens there is no physical reason
  // for area scaling; if the extra DOF introduce significant scaling they
  // are fitting exposure or vignetting gradients, not real geometry.
  const float det_2x2 = Hn[0] * Hn[4] - Hn[1] * Hn[3];
  if(fabsf(det_2x2 - 1.0f) > HDR_ALIGN_ESCALATION_MAX_SCALE_DEVIATION) return FALSE;

  if(ndof == 6)
  {
    // Affine: perspective must remain zero.
    if(fabsf(Hn[6]) > 1e-6f || fabsf(Hn[7]) > 1e-6f) return FALSE;
  }
  else
  {
    // Projective: allow small perspective.
    if(fabsf(Hn[6]) > 0.02f) return FALSE;
    if(fabsf(Hn[7]) > 0.02f) return FALSE;
  }

  // Check that the four image corners map to reasonable source positions.
  // The mapped corner should not be further than 15% of the image diagonal
  // from the original corner.
  const float corners[4][2] = { { 0, 0 }, { (float)(w-1), 0 },
                                 { 0, (float)(h-1) }, { (float)(w-1), (float)(h-1) } };
  const float max_shift = 0.15f * sqrtf((float)((w-1)*(w-1) + (h-1)*(h-1)));

  for(int i = 0; i < 4; i++)
  {
    const float xf = corners[i][0], yf = corners[i][1];
    const float d = H[6] * xf + H[7] * yf + 1.0f;
    if(fabsf(d) < 1e-6f) return FALSE;
    const float sx = (H[0] * xf + H[1] * yf + H[2]) / d;
    const float sy = (H[3] * xf + H[4] * yf + H[5]) / d;
    const float dx = sx - xf, dy = sy - yf;
    if(sqrtf(dx * dx + dy * dy) > max_shift) return FALSE;
  }

  return TRUE;
}

/** Attempt adaptive DOF escalation at the finest pyramid level.
 *  Measures the weighted ECC score of the current 3-DOF result and
 *  selectively tries higher-DOF models when the rigid fit is insufficient.
 *
 *  Escalation path: 3-DOF → 6-DOF affine → 8-DOF projective.
 *  Each step is gated by measurable improvement in ρ, a Hessian conditioning
 *  check, and a geometric sanity check.  Falls back to the input H if
 *  escalation fails.
 *
 *  Returns the best ρ achieved (either from the accepted DOF level or the
 *  original 3-DOF result).  Returns -2 on measurement failure. */
static float _try_dof_escalation(const float *ref_grad,
                                 const float *img_grad,
                                 const int w,
                                 const int h,
                                 float H[HDR_ALIGN_H_NPARAM])
{
  // Measure current (3-DOF) quality.
  const float rho_3dof = _ecc_compute_rho(ref_grad, img_grad, w, h, H);
  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] DOF escalation: 3-DOF ρ = %.5f (threshold = %.2f)",
           rho_3dof, HDR_ALIGN_ESCALATION_RHO_THRESHOLD);

  if(rho_3dof >= HDR_ALIGN_ESCALATION_RHO_THRESHOLD || rho_3dof < -1.0f)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] DOF escalation: not needed (ρ >= threshold)");
    return rho_3dof;
  }

  // Save the 3-DOF result so we can fall back.
  float H_3dof[HDR_ALIGN_H_NPARAM];
  memcpy(H_3dof, H, sizeof(H_3dof));

  // --- Stage 1: try 6-DOF affine ---
  float H_6dof[HDR_ALIGN_H_NPARAM];
  memcpy(H_6dof, H_3dof, sizeof(H_6dof));

  const float rho_6dof = _ecc_refine_level_higher_dof(ref_grad, img_grad,
                                                       w, h, H_6dof, 6);

  const float improvement_6 = rho_6dof - rho_3dof;
  const gboolean sane_6 = _homography_is_sane_escalated(H_6dof, w, h, 6);

  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] DOF escalation: 6-DOF ρ = %.5f (Δρ = %.5f, sane = %d)",
           rho_6dof, improvement_6, sane_6);

  gboolean accept_6 = sane_6
                       && rho_6dof > rho_3dof
                       && improvement_6 >= HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT;

  // --- Stage 2: try 8-DOF projective, starting from the better of 3-DOF/6-DOF ---
  float H_8dof[HDR_ALIGN_H_NPARAM];
  float rho_base_for_8 = rho_3dof;
  if(accept_6)
  {
    memcpy(H_8dof, H_6dof, sizeof(H_8dof));
    rho_base_for_8 = rho_6dof;
  }
  else
    memcpy(H_8dof, H_3dof, sizeof(H_8dof));

  const float rho_8dof = _ecc_refine_level_higher_dof(ref_grad, img_grad,
                                                       w, h, H_8dof, 8);

  const float improvement_8 = rho_8dof - rho_base_for_8;
  const gboolean sane_8 = _homography_is_sane_escalated(H_8dof, w, h, 8);

  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] DOF escalation: 8-DOF ρ = %.5f (Δρ vs base = %.5f, sane = %d)",
           rho_8dof, improvement_8, sane_8);

  const gboolean accept_8 = sane_8
                             && rho_8dof > rho_base_for_8
                             && improvement_8 >= HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT;

  // --- Accept best result ---
  if(accept_8)
  {
    memcpy(H, H_8dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] DOF escalation: accepted 8-DOF (ρ %.5f → %.5f)",
             rho_3dof, rho_8dof);
    return rho_8dof;
  }
  else if(accept_6)
  {
    memcpy(H, H_6dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] DOF escalation: accepted 6-DOF (ρ %.5f → %.5f)",
             rho_3dof, rho_6dof);
    return rho_6dof;
  }
  else
  {
    // Keep original 3-DOF result.
    memcpy(H, H_3dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] DOF escalation: no improvement, keeping 3-DOF");
    return rho_3dof;
  }
}

/** Simple NCC on the full image (no patches / crops).
 *  Used only for the coarse exhaustive search at the smallest pyramid level. */
static float _ncc_full(const float *ref,
                       const float *img,
                       const int w,
                       const int h,
                       const int dx,
                       const int dy)
{
  const int x0 = MAX(0, dx);
  const int y0 = MAX(0, dy);
  const int x1 = MIN(w, w + dx);
  const int y1 = MIN(h, h + dy);
  const int ow = x1 - x0;
  const int oh = y1 - y0;

  if(ow <= 4 || oh <= 4) return -2.0f;

  const size_t n = (size_t)ow * oh;
  double sum_r = 0.0, sum_i = 0.0;

  for(int y = y0; y < y1; y++)
    for(int x = x0; x < x1; x++)
    {
      sum_r += (double)ref[y * w + x];
      sum_i += (double)img[(y - dy) * w + (x - dx)];
    }

  const double mean_r = sum_r / (double)n;
  const double mean_i = sum_i / (double)n;

  double cross = 0.0, var_r = 0.0, var_i = 0.0;

  for(int y = y0; y < y1; y++)
    for(int x = x0; x < x1; x++)
    {
      const double r = (double)ref[y * w + x] - mean_r;
      const double i = (double)img[(y - dy) * w + (x - dx)] - mean_i;
      cross += r * i;
      var_r += r * r;
      var_i += i * i;
    }

  const double denom = sqrt(var_r * var_i);
  if(denom < 1e-12) return -2.0f;

  return (float)(cross / denom);
}

// ---------------------------------------------------------------------------
// Pyramid structure
// ---------------------------------------------------------------------------

typedef struct _pyramid_t
{
  float *data[HDR_ALIGN_MAX_PYRAMID_LEVELS];
  int width[HDR_ALIGN_MAX_PYRAMID_LEVELS];
  int height[HDR_ALIGN_MAX_PYRAMID_LEVELS];
  int nlevels;
} _pyramid_t;

static gboolean _build_pyramid(const float *base, const int bw, const int bh,
                                _pyramid_t *pyr)
{
  memset(pyr, 0, sizeof(_pyramid_t));

  // Level 0 is a copy of the base grayscale image
  pyr->data[0] = dt_alloc_align_float((size_t)bw * bh);
  if(!pyr->data[0]) return FALSE;
  memcpy(pyr->data[0], base, sizeof(float) * (size_t)bw * bh);
  pyr->width[0] = bw;
  pyr->height[0] = bh;
  pyr->nlevels = 1;

  // Build deeper levels by 2x downsampling until the coarsest side
  // is at or below HDR_ALIGN_COARSEST_SIZE
  for(int l = 1; l < HDR_ALIGN_MAX_PYRAMID_LEVELS; l++)
  {
    const int prev_max = MAX(pyr->width[l - 1], pyr->height[l - 1]);
    if(prev_max <= HDR_ALIGN_COARSEST_SIZE || prev_max < 16) break;

    int nw, nh;
    pyr->data[l] = _downsample_2x(pyr->data[l - 1],
                                   pyr->width[l - 1],
                                   pyr->height[l - 1],
                                   &nw, &nh);
    if(!pyr->data[l]) break;
    pyr->width[l] = nw;
    pyr->height[l] = nh;
    pyr->nlevels = l + 1;
  }

  return pyr->nlevels >= 2;
}

static void _free_pyramid(_pyramid_t *pyr)
{
  for(int l = 0; l < pyr->nlevels; l++)
  {
    dt_free_align(pyr->data[l]);
    pyr->data[l] = NULL;
  }
  pyr->nlevels = 0;
}

// ---------------------------------------------------------------------------
// Public API: alignment computation (multi-resolution ECC)
// ---------------------------------------------------------------------------

gboolean dt_hdr_align_compute(const float *ref_mosaic,
                              const float *img_mosaic,
                              const int wd,
                              const int ht,
                              dt_hdr_alignment_t *out_align)
{
  out_align->H[0] = 1.0f; out_align->H[1] = 0.0f; out_align->H[2] = 0.0f;
  out_align->H[3] = 0.0f; out_align->H[4] = 1.0f; out_align->H[5] = 0.0f;
  out_align->H[6] = 0.0f; out_align->H[7] = 0.0f;
  _zero_mesh(out_align->mesh_dx, out_align->mesh_dy);

  if(wd < HDR_ALIGN_MIN_DIM || ht < HDR_ALIGN_MIN_DIM) return FALSE;

  // Step 1: Convert Bayer mosaic to grayscale
  int gw_ref, gh_ref, gw_img, gh_img;
  float *gray_ref = _mosaic_to_grayscale(ref_mosaic, wd, ht, &gw_ref, &gh_ref);
  float *gray_img = _mosaic_to_grayscale(img_mosaic, wd, ht, &gw_img, &gh_img);
  if(!gray_ref || !gray_img) goto error;

  // Step 2: Build pyramids
  _pyramid_t pyr_ref, pyr_img;
  if(!_build_pyramid(gray_ref, gw_ref, gh_ref, &pyr_ref))
  {
    dt_free_align(gray_ref);
    dt_free_align(gray_img);
    return FALSE;
  }
  if(!_build_pyramid(gray_img, gw_img, gh_img, &pyr_img))
  {
    _free_pyramid(&pyr_ref);
    dt_free_align(gray_ref);
    dt_free_align(gray_img);
    return FALSE;
  }

  dt_free_align(gray_ref);
  dt_free_align(gray_img);
  gray_ref = gray_img = NULL;

  // Step 3: Coarse exhaustive search at the deepest (smallest) pyramid level.
  //         NCC is still appropriate here because the images are tiny
  //         (~64×48) and the ECC algorithm needs a reasonable starting
  //         point to converge.
  const int coarsest = pyr_ref.nlevels - 1;
  const int cw = pyr_ref.width[coarsest];
  const int ch = pyr_ref.height[coarsest];

  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] pyramid: %d levels, coarsest %dx%d, finest %dx%d",
           pyr_ref.nlevels, cw, ch, pyr_ref.width[0], pyr_ref.height[0]);

  float best_tx = 0.0f, best_ty = 0.0f, best_angle = 0.0f;
  float best_ncc = -2.0f;

  const int search_radius = (int)(MAX(cw, ch) * HDR_ALIGN_COARSE_SEARCH_FRAC);
  const float angle_step_rad = HDR_ALIGN_COARSE_ANGLE_STEP * (float)M_PI / 180.0f;
  const float max_angle_rad = HDR_ALIGN_MAX_ANGLE_DEG * (float)M_PI / 180.0f;

  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] coarse search: %dx%d, radius=%d, angle_step=%.1f°",
           cw, ch, search_radius, HDR_ALIGN_COARSE_ANGLE_STEP);

  // Compute gradient magnitude images for exposure-invariant coarse search.
  // Raw-pixel NCC can fail badly when the exposure difference between
  // brackets is very large (clipped highlights, noisy shadows).  Gradient
  // magnitude removes the intensity DC component so that edges -- which
  // appear at the same locations regardless of exposure -- drive the match.
  // This is the same preprocessing that the ECC refinement uses.
  const size_t cpix = (size_t)cw * ch;
  float *coarse_ref_grad = NULL;
  float *coarse_img_grad = NULL;
  {
    float *tmp_ref = dt_alloc_align_float(cpix);
    float *tmp_img = dt_alloc_align_float(cpix);
    if(tmp_ref && tmp_img)
    {
      memcpy(tmp_ref, pyr_ref.data[coarsest], sizeof(float) * cpix);
      memcpy(tmp_img, pyr_img.data[coarsest], sizeof(float) * cpix);
      _normalize_image_01(tmp_ref, cpix);
      _normalize_image_01(tmp_img, cpix);
      coarse_ref_grad = _gradient_magnitude(tmp_ref, cw, ch);
      coarse_img_grad = _gradient_magnitude(tmp_img, cw, ch);
    }
    dt_free_align(tmp_ref);
    dt_free_align(tmp_img);
  }
  // Fall back to raw pixels if either gradient computation failed.
  // Both must succeed — mixing gradient and raw images would be wrong.
  const gboolean use_grad = coarse_ref_grad && coarse_img_grad;
  const float *coarse_ref = use_grad ? coarse_ref_grad
                                     : pyr_ref.data[coarsest];
  const float *coarse_img = use_grad ? coarse_img_grad
                                     : pyr_img.data[coarsest];

  // Capture the identity NCC (tx=0, ty=0, angle=0°) before the full search.
  // This tells us how well-aligned the images already are without any
  // correction, providing an early-out and a baseline for quality checks.
  const float ncc_identity = _ncc_full(coarse_ref, coarse_img, cw, ch, 0, 0);

  for(float angle = -max_angle_rad; angle <= max_angle_rad;
      angle += angle_step_rad)
  {
    // Warp coarsest-level input by candidate angle (no translation yet)
    float *rotated = NULL;
    const float *candidate = coarse_img;

    if(fabsf(angle) > 1e-6f)
    {
      rotated = _warp_euclidean(coarse_img, cw, ch,
                                0.0f, 0.0f, angle, NULL);
      if(!rotated) continue;
      candidate = rotated;
    }

    for(int dy = -search_radius; dy <= search_radius; dy++)
      for(int dx = -search_radius; dx <= search_radius; dx++)
      {
        const float ncc = _ncc_full(coarse_ref, candidate,
                                    cw, ch, dx, dy);
        if(ncc > best_ncc)
        {
          best_ncc = ncc;
          best_tx = (float)dx;
          best_ty = (float)dy;
          best_angle = angle;
        }
      }

    dt_free_align(rotated);
  }

  dt_free_align(coarse_ref_grad);
  dt_free_align(coarse_img_grad);

  dt_print(DT_DEBUG_ALWAYS,
           "[hdr_merge] coarse result: tx=%.0f ty=%.0f angle=%.2f° ncc=%.4f"
           " (identity ncc=%.4f)",
           best_tx, best_ty, best_angle * 180.0f / (float)M_PI, best_ncc,
           ncc_identity);

  // Early-out: if the images are already extremely well-aligned (identity
  // NCC close to 1.0), the full ECC pyramid would risk introducing drift
  // for no benefit.  Return identity immediately.
  if(ncc_identity >= HDR_ALIGN_COARSE_IDENTITY_SKIP_NCC)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] identity ncc %.4f >= %.2f -- images already aligned,"
             " skipping ECC",
             ncc_identity, HDR_ALIGN_COARSE_IDENTITY_SKIP_NCC);
    out_align->H[0] = 1.0f; out_align->H[1] = 0.0f; out_align->H[2] = 0.0f;
    out_align->H[3] = 0.0f; out_align->H[4] = 1.0f; out_align->H[5] = 0.0f;
    out_align->H[6] = 0.0f; out_align->H[7] = 0.0f;
    _zero_mesh(out_align->mesh_dx, out_align->mesh_dy);

    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] final homography: H=[1.00000 0.00000 0.00;"
             " 0.00000 1.00000 0.00; 0.0000000 0.0000000 1],"
             " approx dx=0.00 dy=0.00 angle=0.0000°,"
             " mesh max=0.00 px, mesh center=(0.00, 0.00)");

    _free_pyramid(&pyr_ref);
    _free_pyramid(&pyr_img);
    return TRUE;
  }

  // Initialise projective model from coarse Euclidean estimate.
  float H_level[HDR_ALIGN_H_NPARAM];
  _homography_from_euclidean(best_tx, best_ty, best_angle, cw, ch, H_level);

  // Step 4: Multi-resolution ECC refinement.
  //
  // Starting from the coarsest level, run iterative ECC at each level
  // to refine the (tx, ty, angle) parameters to sub-pixel accuracy.
  // The ECC (Evangelidis & Psarakis 2008) maximises the Enhanced
  // Correlation Coefficient, which is invariant to affine photometric
  // transforms — exactly what we need for HDR brackets with different
  // exposures.  Parameters are scaled by 2× when transitioning to
  // each finer level.
  for(int l = coarsest; l >= 0; l--)
  {
    const int lw = pyr_ref.width[l];
    const int lh = pyr_ref.height[l];

    // Scale homography from parent level to current (finer) level.
    if(l < coarsest)
      _homography_scale_to_finer(H_level);

    // Skip ECC on very small levels — the Hessian is ill-conditioned
    // and the coarse NCC result is sufficient at these scales.
    if(lw < HDR_ALIGN_ECC_MIN_DIM || lh < HDR_ALIGN_ECC_MIN_DIM)
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge] ECC level %d (%dx%d): skipped (too small)",
               l, lw, lh);
      continue;
    }

    // Compute gradient magnitude images for exposure-invariant ECC.
    // Gradient magnitude removes the multiplicative exposure difference
    // that causes raw-pixel ECC to fail on HDR brackets.
    // Normalise each pyramid level to [0,1] first so that the gradient
    // magnitudes of dark and bright exposures are on a comparable scale.
    const size_t lpix = (size_t)lw * lh;
    float *ref_norm = dt_alloc_align_float(lpix);
    float *img_norm = dt_alloc_align_float(lpix);
    if(!ref_norm || !img_norm)
    {
      dt_free_align(ref_norm);
      dt_free_align(img_norm);
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge] ECC level %d: norm alloc failed", l);
      continue;
    }
    memcpy(ref_norm, pyr_ref.data[l], sizeof(float) * lpix);
    memcpy(img_norm, pyr_img.data[l], sizeof(float) * lpix);
    _normalize_image_01(ref_norm, lpix);
    _normalize_image_01(img_norm, lpix);

    float *ref_grad = _gradient_magnitude(ref_norm, lw, lh);
    float *img_grad = _gradient_magnitude(img_norm, lw, lh);
    dt_free_align(ref_norm);
    dt_free_align(img_norm);
    if(!ref_grad || !img_grad)
    {
      dt_free_align(ref_grad);
      dt_free_align(img_grad);
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge] ECC level %d: gradient alloc failed", l);
      continue;
    }

    dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] ECC level %d (%dx%d): initial H=[%.4f %.4f %.2f; %.4f %.4f %.2f; %.6f %.6f 1]",
             l, lw, lh,
             H_level[0], H_level[1], H_level[2],
             H_level[3], H_level[4], H_level[5],
             H_level[6], H_level[7]);

    // Save pre-ECC homography so we can revert if ECC diverges.
    float H_backup[HDR_ALIGN_H_NPARAM];
    memcpy(H_backup, H_level, sizeof(H_backup));

    _ecc_refine_level(ref_grad, img_grad, lw, lh, H_level);

    // Sanity check: revert if ECC produced a physically implausible
    // homography (e.g. large scale change or perspective).
    if(!_homography_is_sane(H_level, lw, lh))
    {
      dt_print(DT_DEBUG_ALWAYS,
               "[hdr_merge] ECC level %d: result failed sanity check, reverting",
               l);
      memcpy(H_level, H_backup, sizeof(H_level));
    }

    // Angle drift guard: if ECC changed the rotation by more than the
    // per-level tolerance it wandered to a wrong local maximum.  Revert to
    // the pre-level homography so the error does not cascade to finer levels.
    {
      float delta = atan2f(H_level[3], H_level[0])
                  - atan2f(H_backup[3], H_backup[0]);
      if(delta >  (float)M_PI) delta -= 2.0f * (float)M_PI;
      if(delta < -(float)M_PI) delta += 2.0f * (float)M_PI;
      const float max_delta = HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG * (float)M_PI / 180.0f;
      if(fabsf(delta) > max_delta)
      {
        dt_print(DT_DEBUG_ALWAYS,
                 "[hdr_merge] ECC level %d: angle drift %.2f° > limit %.1f°, reverting",
                 l, delta * 180.0f / (float)M_PI, HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG);
        memcpy(H_level, H_backup, sizeof(H_level));
      }
    }

    // Translation drift guard: if ECC shifted the translation by more than
    // the per-level tolerance, the optimiser has wandered.  Each level
    // inherits a 2×-scaled estimate from its parent, so the expected
    // correction is at most a few pixels.  Large shifts at any single level
    // indicate a wrong local minimum, and the error doubles at every finer
    // level, producing catastrophically wrong final alignments.
    {
      const float dtx = H_level[2] - H_backup[2];
      const float dty = H_level[5] - H_backup[5];
      if(fabsf(dtx) > HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX
         || fabsf(dty) > HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX)
      {
        dt_print(DT_DEBUG_ALWAYS,
                 "[hdr_merge] ECC level %d: translation drift (%.2f, %.2f) px"
                 " > limit %.1f px, reverting",
                 l, dtx, dty, HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX);
        memcpy(H_level, H_backup, sizeof(H_level));
      }
    }

    // Adaptive DOF escalation at the finest level: if the 3-DOF rigid fit
    // left significant residual misalignment (low ρ), attempt 6-DOF affine
    // and then 8-DOF projective refinement gated by improvement and sanity.
    if(l == 0)
    {
      const float rho_best = _try_dof_escalation(ref_grad, img_grad, lw, lh, H_level);

      // Identity comparison: always check whether the identity transform
      // (no alignment) correlates at least as well as the computed H.
      // A wrong H — even one with moderate ρ — is worse than no correction
      // if the images are already well-aligned or the alignment wandered
      // to a wrong local minimum.  This catches both low-ρ catastrophic
      // failures and medium-ρ wrong solutions.
      {
        const float H_id[HDR_ALIGN_H_NPARAM]
          = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
        const float rho_id = _ecc_compute_rho(ref_grad, img_grad, lw, lh, H_id);
        dt_print(DT_DEBUG_ALWAYS,
                 "[hdr_merge] identity check: ρ_aligned=%.5f ρ_identity=%.5f",
                 rho_best, rho_id);
        if(rho_id >= rho_best)
        {
          dt_print(DT_DEBUG_ALWAYS,
                   "[hdr_merge] identity ρ >= aligned ρ -- reverting to identity");
          memcpy(H_level, H_id, sizeof(float) * HDR_ALIGN_H_NPARAM);
        }
      }
    }

    if(l <= 1)
      _corner_refine_level(ref_grad, img_grad, lw, lh, H_level);

    dt_free_align(ref_grad);
    dt_free_align(img_grad);

    dt_print(DT_DEBUG_ALWAYS,
            "[hdr_merge] ECC level %d result: H=[%.4f %.4f %.2f; %.4f %.4f %.2f; %.6f %.6f 1]",
            l,
            H_level[0], H_level[1], H_level[2],
            H_level[3], H_level[4], H_level[5],
            H_level[6], H_level[7]);
  }

      // Level 0 is half-resolution grayscale built from 2x2 Bayer blocks.
      // Each grayscale sample corresponds to the centre of its block at
      // (2*x + 0.5, 2*y + 0.5) in full-resolution coordinates.
      _homography_local_to_full(H_level, 0.5f, 0.5f, out_align->H);

      float *ref_grad0 = NULL;
      float *img_grad0 = NULL;
      {
        const size_t npix0 = (size_t)pyr_ref.width[0] * pyr_ref.height[0];
        float *rn = dt_alloc_align_float(npix0);
        float *in = dt_alloc_align_float(npix0);
        if(rn && in)
        {
          memcpy(rn, pyr_ref.data[0], sizeof(float) * npix0);
          memcpy(in, pyr_img.data[0], sizeof(float) * npix0);
          _normalize_image_01(rn, npix0);
          _normalize_image_01(in, npix0);
          ref_grad0 = _gradient_magnitude(rn, pyr_ref.width[0], pyr_ref.height[0]);
          img_grad0 = _gradient_magnitude(in, pyr_img.width[0], pyr_img.height[0]);
        }
        dt_free_align(rn);
        dt_free_align(in);
      }
      if(ref_grad0 && img_grad0)
      {
        float mesh_dx_half[DT_HDR_ALIGN_MESH_NODES];
        float mesh_dy_half[DT_HDR_ALIGN_MESH_NODES];
        if(_estimate_mesh_residuals(ref_grad0, img_grad0,
                                    pyr_ref.width[0], pyr_ref.height[0],
                                    H_level, mesh_dx_half, mesh_dy_half))
        {
          for(int i = 0; i < DT_HDR_ALIGN_MESH_NODES; i++)
          {
            out_align->mesh_dx[i] = mesh_dx_half[i] * 2.0f;
            out_align->mesh_dy[i] = mesh_dy_half[i] * 2.0f;
          }
        }
      }
      dt_free_align(ref_grad0);
      dt_free_align(img_grad0);

      const float approx_dx = -out_align->H[2];
      const float approx_dy = -out_align->H[5];
      const float approx_angle = atan2f(out_align->H[1], out_align->H[0]);
        const float mesh_max = _mesh_max_abs(out_align->mesh_dx, out_align->mesh_dy);

  dt_print(DT_DEBUG_ALWAYS,
             "[hdr_merge] final homography: H=[%.5f %.5f %.2f; %.5f %.5f %.2f; %.7f %.7f 1], approx dx=%.2f dy=%.2f angle=%.4f°, mesh max=%.2f px, mesh center=(%.2f, %.2f)",
               out_align->H[0], out_align->H[1], out_align->H[2],
               out_align->H[3], out_align->H[4], out_align->H[5],
               out_align->H[6], out_align->H[7],
           approx_dx, approx_dy, approx_angle * 180.0f / (float)M_PI,
           mesh_max,
           out_align->mesh_dx[HDR_ALIGN_MESH_INDEX(1, 1)],
           out_align->mesh_dy[HDR_ALIGN_MESH_INDEX(1, 1)]);

  _free_pyramid(&pyr_ref);
  _free_pyramid(&pyr_img);
  return TRUE;

error:
  dt_free_align(gray_ref);
  dt_free_align(gray_img);
  out_align->H[0] = 1.0f; out_align->H[1] = 0.0f; out_align->H[2] = 0.0f;
  out_align->H[3] = 0.0f; out_align->H[4] = 1.0f; out_align->H[5] = 0.0f;
  out_align->H[6] = 0.0f; out_align->H[7] = 0.0f;
  _zero_mesh(out_align->mesh_dx, out_align->mesh_dy);
  return FALSE;
}

// ---------------------------------------------------------------------------
// CFA-aware warping
// ---------------------------------------------------------------------------

/** Decompose a Bayer mosaic into 4 half-resolution color planes.
 *  planes[0..3] correspond to the 4 positions in the 2x2 Bayer block
 *  in raster order (row0col0, row0col1, row1col0, row1col1).
 *  Each plane has dimensions (wd/2) x (ht/2). */
static void _bayer_decompose(const float *mosaic,
                             float *planes[4],
                             const int wd,
                             const int ht)
{
  const int pw = wd / 2;
  const int ph = ht / 2;

  DT_OMP_FOR(collapse(2))
  for(int py = 0; py < ph; py++)
    for(int px = 0; px < pw; px++)
    {
      const int mx = px * 2;
      const int my = py * 2;
      planes[0][py * pw + px] = mosaic[my * wd + mx];             // row0, col0
      planes[1][py * pw + px] = mosaic[my * wd + mx + 1];         // row0, col1
      planes[2][py * pw + px] = mosaic[(my + 1) * wd + mx];       // row1, col0
      planes[3][py * pw + px] = mosaic[(my + 1) * wd + mx + 1];   // row1, col1
    }
}

/** Recompose 4 half-resolution color planes back into a Bayer mosaic.
 *  Inverse of _bayer_decompose. */
static void _bayer_recompose(const float *planes[4],
                             float *mosaic,
                             const int wd,
                             const int ht)
{
  const int pw = wd / 2;
  const int ph = ht / 2;

  DT_OMP_FOR(collapse(2))
  for(int py = 0; py < ph; py++)
    for(int px = 0; px < pw; px++)
    {
      const int mx = px * 2;
      const int my = py * 2;
      mosaic[my * wd + mx]             = planes[0][py * pw + px]; // row0, col0
      mosaic[my * wd + mx + 1]         = planes[1][py * pw + px]; // row0, col1
      mosaic[(my + 1) * wd + mx]       = planes[2][py * pw + px]; // row1, col0
      mosaic[(my + 1) * wd + mx + 1]   = planes[3][py * pw + px]; // row1, col1
    }
}

/** Warp a single half-resolution color plane using the full-resolution
 *  residual mesh plus the backward-mapping homography. */
static void _warp_plane(const float *in,
                        float *out,
                        const int pw,
                        const int ph,
                        const int wd,
                        const int ht,
                        const float off_x,
                        const float off_y,
                        const dt_hdr_alignment_t *align)
{
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < ph; y++)
    for(int x = 0; x < pw; x++)
    {
      const float xf = 2.0f * (float)x + off_x;
      const float yf = 2.0f * (float)y + off_y;
      const float u = wd > 1 ? xf / (float)(wd - 1) : 0.0f;
      const float v = ht > 1 ? yf / (float)(ht - 1) : 0.0f;
      const float mesh_dx = _sample_mesh_grid(align->mesh_dx, u, v);
      const float mesh_dy = _sample_mesh_grid(align->mesh_dy, u, v);

      const float xw = xf + mesh_dx;
      const float yw = yf + mesh_dy;
      const float den = align->H[6] * xw + align->H[7] * yw + 1.0f;

      if(fabsf(den) < 1e-8f)
      {
        out[y * pw + x] = -1.0f;
        continue;
      }

      const float sx_full = (align->H[0] * xw + align->H[1] * yw + align->H[2]) / den;
      const float sy_full = (align->H[3] * xw + align->H[4] * yw + align->H[5]) / den;
      const float sx = 0.5f * (sx_full - off_x);
      const float sy = 0.5f * (sy_full - off_y);

      // bilinear interpolation (we avoid pulling in dt_interpolation
      // here to keep it simple and avoid the samplestride/linestride
      // byte-count interface for this trivial single-channel case)
      const int ix = (int)floorf(sx);
      const int iy = (int)floorf(sy);

      if(ix >= 0 && ix < pw - 1 && iy >= 0 && iy < ph - 1)
      {
        const float fx = sx - (float)ix;
        const float fy = sy - (float)iy;
        out[y * pw + x] = (1.0f - fx) * (1.0f - fy) * in[iy * pw + ix]
                         + fx * (1.0f - fy) * in[iy * pw + ix + 1]
                         + (1.0f - fx) * fy * in[(iy + 1) * pw + ix]
                         + fx * fy * in[(iy + 1) * pw + ix + 1];
      }
      else
      {
        // Mark out-of-bounds pixels with a negative sentinel so the
        // merge code can distinguish them from legitimate zero values
        out[y * pw + x] = -1.0f;
      }
    }
}

// ---------------------------------------------------------------------------
// Public API: apply alignment
// ---------------------------------------------------------------------------

void dt_hdr_align_apply(const float *in_mosaic,
                        float *out_mosaic,
                        const int wd,
                        const int ht,
                        const uint32_t filters,
                        const dt_hdr_alignment_t *align)
{
  const int pw = wd / 2;
  const int ph = ht / 2;
  const size_t plane_size = (size_t)pw * ph;

  // Allocate 4 input planes + 4 output planes
  float *in_planes[4];
  float *out_planes[4];
  for(int i = 0; i < 4; i++)
  {
    in_planes[i] = dt_alloc_align_float(plane_size);
    out_planes[i] = dt_alloc_align_float(plane_size);
    if(!in_planes[i] || !out_planes[i])
    {
      // cleanup on allocation failure
      for(int j = 0; j <= i; j++)
      {
        dt_free_align(in_planes[j]);
        dt_free_align(out_planes[j]);
      }
      // on failure, zero out the output (will get zero weight in merge)
      memset(out_mosaic, 0, sizeof(float) * (size_t)wd * ht);
      return;
    }
  }

  // Decompose into 4 Bayer sub-planes
  _bayer_decompose(in_mosaic, in_planes, wd, ht);

  // Warp each plane independently in full-resolution output coordinates.
  static const float plane_off_x[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
  static const float plane_off_y[4] = { 0.0f, 0.0f, 1.0f, 1.0f };

  for(int i = 0; i < 4; i++)
    _warp_plane(in_planes[i], out_planes[i], pw, ph, wd, ht,
                plane_off_x[i], plane_off_y[i], align);

  // Recompose back into mosaic
  _bayer_recompose((const float **)out_planes, out_mosaic, wd, ht);

  for(int i = 0; i < 4; i++)
  {
    dt_free_align(in_planes[i]);
    dt_free_align(out_planes[i]);
  }
}

// ---------------------------------------------------------------------------
// OpenCL global init / cleanup
// ---------------------------------------------------------------------------

#ifdef HAVE_OPENCL

#include "common/opencl.h"

dt_hdr_alignment_cl_global_t *dt_hdr_alignment_init_cl_global(void)
{
  dt_hdr_alignment_cl_global_t *g = malloc(sizeof(*g));
  if(!g) return NULL;

  const int program = 41; // hdr_alignment.cl, from programs.conf

  g->kernel_warp_homography  = dt_opencl_create_kernel(program, "hdr_align_warp_homography");
  g->kernel_compute_gradients = dt_opencl_create_kernel(program, "hdr_align_compute_gradients");
  g->kernel_gradient_magnitude = dt_opencl_create_kernel(program, "hdr_align_gradient_magnitude");
  g->kernel_normalize_01     = dt_opencl_create_kernel(program, "hdr_align_normalize_01");
  g->kernel_mosaic_to_gray   = dt_opencl_create_kernel(program, "hdr_align_mosaic_to_gray");
  g->kernel_downsample_2x    = dt_opencl_create_kernel(program, "hdr_align_downsample_2x");
  g->kernel_ecc_means        = dt_opencl_create_kernel(program, "hdr_align_ecc_means");
  g->kernel_ecc_norms        = dt_opencl_create_kernel(program, "hdr_align_ecc_norms");
  g->kernel_ecc_hessian      = dt_opencl_create_kernel(program, "hdr_align_ecc_hessian");
  g->kernel_ecc_hessian_final = dt_opencl_create_kernel(program, "hdr_align_ecc_hessian_final");

  return g;
}

void dt_hdr_alignment_free_cl_global(dt_hdr_alignment_cl_global_t *g)
{
  if(!g) return;
  dt_opencl_free_kernel(g->kernel_warp_homography);
  dt_opencl_free_kernel(g->kernel_compute_gradients);
  dt_opencl_free_kernel(g->kernel_gradient_magnitude);
  dt_opencl_free_kernel(g->kernel_normalize_01);
  dt_opencl_free_kernel(g->kernel_mosaic_to_gray);
  dt_opencl_free_kernel(g->kernel_downsample_2x);
  dt_opencl_free_kernel(g->kernel_ecc_means);
  dt_opencl_free_kernel(g->kernel_ecc_norms);
  dt_opencl_free_kernel(g->kernel_ecc_hessian);
  dt_opencl_free_kernel(g->kernel_ecc_hessian_final);
  free(g);
}

#endif /* HAVE_OPENCL */

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
