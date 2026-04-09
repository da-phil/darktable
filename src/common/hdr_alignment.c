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
#include "common/gaussian.h"
#include "common/interpolation.h"
#include "common/math.h"
#include "common/pfm.h"

#ifdef HAVE_OPENCL
#include "common/opencl.h"
#endif

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// CFA mode selection for pyramid level 0 (finest level)
// ---------------------------------------------------------------------------
//
// HDR_ALIGN_L0_MODE selects how the Bayer mosaic is converted into
// the alignment pyramid's finest level:
//
//   HDR_ALIGN_L0_FULL_CFA (0):
//     The pyramid is built from full-resolution Bayer data.  L0 retains
//     the CFA pattern and uses per-sublattice 99th-percentile normalisation
//     followed by a stride-2 Sobel filter so that each CFA channel's
//     gradients are computed from same-channel neighbours only.  This
//     produces the sharpest possible gradients and isolates saturated
//     pixels to their own channel's edge response.
//
//   HDR_ALIGN_L0_AVG_BAYER (1):
//     L0 is converted to half-resolution grayscale by averaging each 2×2
//     Bayer block.  Standard global percentile normalisation + stride-1
//     Sobel is used.  This is the safe fallback: it halves the L0 pixel
//     count and avoids CFA-related artefacts, but averages different
//     colour channels which can spread a single saturated pixel to a
//     whole 2×2 block.
//
//   HDR_ALIGN_L0_GREEN_ONLY (2):
//     L0 is built from the green channel only: the two green pixels in
//     each 2×2 Bayer block are averaged into one half-resolution sample.
//     Green has the best SNR (highest quantum efficiency, 2 samples per
//     block) and introduces no demosaicing artefacts.  The resulting
//     image is photometrically consistent across exposures because the
//     spectral response is uniform.  Standard global percentile norm +
//     stride-1 Sobel is used (same as AVG_BAYER).
//
// At L1 and above, all modes are identical because the 2× box-filter
// downsample inherently averages Bayer blocks into grayscale.
//
// The restructured gradient pipeline (Gaussian blur → Sobel gx,gy →
// magnitude → percentile+power+threshold → mask → gx+gy → MAD norm →
// apply mask) runs identically regardless of mode; only the input
// preparation and Sobel stride differ.

#define HDR_ALIGN_L0_FULL_CFA   0
#define HDR_ALIGN_L0_AVG_BAYER  1
#define HDR_ALIGN_L0_GREEN_ONLY 2

// Select the active L0 mode here:
#define HDR_ALIGN_L0_MODE HDR_ALIGN_L0_GREEN_ONLY

// Maximum rotation search range in degrees (for coarse exhaustive search)
#define HDR_ALIGN_MAX_ANGLE_DEG 10.0f
// Coarse rotation step in degrees
#define HDR_ALIGN_COARSE_ANGLE_STEP 0.5f
// Maximum number of pyramid levels (enough for up to 32k images)
#define HDR_ALIGN_MAX_PYRAMID_LEVELS 12
// Target longest-side for the coarsest pyramid level.
// The coarse NCC search runs at the coarsest pyramid level.  A too-small
// coarsest image (37×24 = 888 px) cannot reliably discriminate between the
// true alignment and false NCC maxima when scale, rotation and translation
// vary simultaneously.  Setting the threshold to 128 makes the coarsest level
// 74×49 (3626 px, 4× more pixels) which gives a sharper NCC landscape.
#define HDR_ALIGN_COARSEST_SIZE 128
// Minimum image dimension for alignment to make sense
#define HDR_ALIGN_MIN_DIM 64
// Maximum ECC iterations per pyramid level
#define HDR_ALIGN_ECC_MAX_ITER 50
// ECC convergence threshold (pixel-equivalent): stop when parameter update
// norm (|Δtx| + |Δty| + |Δθ|·corner_dist) is below this value.
#define HDR_ALIGN_ECC_EPSILON 1e-3f
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
// Minimum size of the shortest edge (pixels) for an ECC pyramid level to be
// processed.  Levels below this threshold are skipped: at very small
// resolutions the gradient images carry insufficient spatial information to
// produce a reliable alignment update, and the Hessian becomes ill-conditioned.
// 128 pixels on the shortest edge is the practical lower bound — anything
// coarser than that does not benefit the ECC refinement.
#define HDR_ALIGN_ECC_MIN_DIM 64
// Number of consecutive iterations without improvement before declaring stall.
// Avoids wasting iterations at coarse pyramid levels where the update metric
// plateaus above the convergence threshold.
#define HDR_ALIGN_ECC_PATIENCE 5
// Maximum rotation change (degrees) that a single ECC pyramid level is allowed
// to introduce.  If ECC drifts the angle further than this it has wandered to
// a wrong local maximum; the result is reverted to the pre-level homography.
// This is the base value at the coarsest ECC level; it is halved for each
// finer level (see _adaptive_max_angle_delta and _adaptive_max_trans_delta).
#define HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_BASE 10.0f
// Minimum per-level angle drift limit (degrees) after adaptive halving.
#define HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_MIN  1.0f
// Maximum translation change (pixels at the current level) that a single ECC
// pyramid level may introduce.  Each level inherits a 2×-scaled estimate from
// its parent, so the parent's accuracy at the child's scale is ~2 px.
// This is the base value at the coarsest ECC level; it is halved for each
// finer level (same schedule as angle).
#define HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX_BASE 10.0f
// Minimum per-level translation drift limit after adaptive halving.
#define HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX_MIN  2.0f
// Number of free parameters in projective homography (h22 fixed to 1)
#define HDR_ALIGN_H_NPARAM 8

// --- Spatial pre-filtering parameters ---
// Gaussian blur sigma (in pixels) applied to each pyramid level before
// gradient extraction.  Suppresses sensor noise and Bayer-residual
// high-frequency artefacts that would otherwise create spurious gradient
// energy and degrade ECC convergence.  At coarse pyramid levels (shortest
// edge < HDR_ALIGN_PREFILTER_MIN_DIM) the blur is skipped because the 2×
// box-filter downsampling already provides sufficient anti-aliasing.
#define HDR_ALIGN_PREFILTER_SIGMA 3.0f
// Minimum shortest-edge dimension for applying the Gaussian pre-filter.
// Below this size the image is too small for a σ=3 blur to be meaningful.
#define HDR_ALIGN_PREFILTER_MIN_DIM 64

// --- Gradient magnitude normalization parameters ---
// Power exponent applied to the percentile-normalised gradient magnitude
// before mask construction.  Values < 1 compress strong gradients and
// lift weak ones, giving more uniform weight to texture-rich and smooth
// regions alike.  0.5 (square root) is a good default.
#define HDR_ALIGN_GRAD_MAG_POWER 0.5f
// Threshold applied to the [0,1]-normalised gradient magnitude.
// Pixels below this fraction of the 99th-percentile magnitude are
// considered too featureless for reliable ECC alignment and are masked out.
#define HDR_ALIGN_GRAD_MAG_THRESHOLD 0.02f
// Number of Jacobi smoothing iterations for the residual mesh.
#define HDR_ALIGN_MESH_SMOOTH_ITERS 12
// Smoothness weight used when regularising the residual mesh.
#define HDR_ALIGN_MESH_SMOOTH_LAMBDA 1.5f

// --- Adaptive DOF escalation parameters ---
// Minimum ρ improvement required to accept a 6-DOF result over 3-DOF.
#define HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT 0.01f
// Minimum ρ improvement required to accept 8-DOF over 6-DOF.
// Lower than the 3→6 threshold because 8-DOF adds only 2 extra parameters
// (perspective) over 6-DOF, so the expected noise gain is smaller.
#define HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT_8DOF 0.005f
// Maximum Hessian condition number for accepting a higher-DOF result.
// A poorly conditioned Hessian indicates the extra parameters are not
// well determined by the data.
#define HDR_ALIGN_ESCALATION_MAX_COND 1e6
// Coarse NCC scale search range and step (isotropic scale).
// Lens breathing, aperture changes and non-equal zoom can introduce
// significant scale differences between HDR brackets.  The scale search
// ensures the coarse translation/rotation search is in the correct basin
// even when the images differ in size by up to 45%.
#define HDR_ALIGN_COARSE_SCALE_MIN  0.70f
#define HDR_ALIGN_COARSE_SCALE_MAX  1.45f
#define HDR_ALIGN_COARSE_SCALE_STEP 0.05f
// Maximum ECC iterations for the escalated (higher-DOF) solver.
// 50 iterations are needed when the rigid pyramid left a large residual
// scale error (~15%) that the 6-DOF solver must bridge incrementally.
#define HDR_ALIGN_ESCALATION_MAX_ITER 50
// ECC convergence threshold for the escalated solver.
#define HDR_ALIGN_ESCALATION_EPSILON 5e-3f

// Maximum geometric-mean scale deviation allowed in an escalated homography.
// sqrt(det(A)) is the per-axis scale factor; we allow up to ±50% deviation
// from 1.0 to accommodate lens breathing, aperture-induced scale change, and
// different focal lengths.  Values beyond this range indicate the extra DOF
// are fitting non-geometric variation (exposure gradients, vignetting) rather
// than real image geometry.
#define HDR_ALIGN_ESCALATION_MAX_SCALE_DEVIATION 0.50f

// Maximum shear magnitude allowed in an escalated homography.
// Shear is defined as (h12 + h21) / 2 — the symmetric part of the 2×2
// submatrix.  Physical camera movements produce negligible shear: pure
// rotation, translation, zoom, and moderate perspective all produce
// shear < 0.004.  The threshold is set 50% above the empirical physical
// maximum to provide a safety margin for unusual lens/camera combinations
// and extreme perspective, while still reliably catching non-geometric
// patterns such as exposure gradients, vignetting, or moving scene
// content (e.g. ocean waves, shear typically > 0.008) that the optimizer
// can fit as spurious affine deformation.
#define HDR_ALIGN_ESCALATION_MAX_SHEAR 0.006f

// Minimum shortest-edge dimension (pixels) for DOF escalation to trigger.
// The escalation runs at the first (coarsest) pyramid level whose shortest
// edge is at least this value.  At smaller resolutions the gradient image
// does not contain enough spatial information to constrain 6-/8-DOF models
// reliably.  Once escalated, the higher-DOF model is kept for all remaining
// finer levels, so the expensive escalation runs on a relatively small image
// while finer levels benefit from the better motion model.
// 512 pixels is chosen as a balance: large enough for perspective / scale
// to be resolvable (≈ 60 gradient Sobel wavelengths), small enough to
// avoid the full-resolution cost of the previous L0-only regime.
#define HDR_ALIGN_ESCALATION_MIN_DIM 512

#define HDR_ALIGN_MESH_INDEX(row, col) ((row) * DT_HDR_ALIGN_MESH_COLS + (col))

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Prepare a Bayer mosaic for alignment input.
 *  Subtracts @p black_level from every pixel, clamps to ≥ 0, and multiplies
 *  by @p inv_exposure (= 1 / (relative_exposure × relative_iso)) to bring
 *  the candidate image to the same effective photon-count scale as the
 *  reference image.  For the reference image pass black_level = 0 and
 *  inv_exposure = 1.0f.
 *
 *  The output is a full-resolution (wd × ht) buffer that retains the Bayer
 *  CFA structure.  At pyramid level 0, _normalize_bayer_per_channel() is
 *  applied afterwards to independently normalise each CFA sublattice before
 *  computing CFA-aware gradients.  Higher pyramid levels are built by 2×
 *  box-filter downsampling from L0; each downsampling step averages a 2×2
 *  Bayer block into one grayscale-equivalent value, so L1 and above are
 *  effectively grayscale.
 *  Caller must free the result. */
static float *_normalize_bayer(const float *mosaic,
                               const int wd,
                               const int ht,
                               const float black_level,
                               const float inv_exposure)
{
  const size_t npix = (size_t)wd * ht;
  float *out = dt_alloc_align_float(npix);
  if(!out) return NULL;

  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    out[i] = fmaxf(mosaic[i] - black_level, 0.0f) * inv_exposure;

  return out;
}

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_AVG_BAYER
/** Convert a full-resolution Bayer mosaic to half-resolution grayscale
 *  by averaging each 2×2 Bayer block.  Output dimensions are (wd/2)×(ht/2).
 *  This produces a photometrically uniform single-channel image suitable
 *  for standard stride-1 Sobel filtering at all pyramid levels.
 *  Caller must free the result. */
static float *_mosaic_to_grayscale(const float *mosaic,
                                   const int wd,
                                   const int ht,
                                   int *out_w,
                                   int *out_h)
{
  const int gw = wd / 2;
  const int gh = ht / 2;
  float *out = dt_alloc_align_float((size_t)gw * gh);
  if(!out) return NULL;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < gh; y++)
    for(int x = 0; x < gw; x++)
    {
      const int mx = x * 2;
      const int my = y * 2;
      out[y * gw + x] = 0.25f * (mosaic[my * wd + mx]
                                  + mosaic[my * wd + mx + 1]
                                  + mosaic[(my + 1) * wd + mx]
                                  + mosaic[(my + 1) * wd + mx + 1]);
    }

  *out_w = gw;
  *out_h = gh;
  return out;
}
#endif /* HDR_ALIGN_L0_AVG_BAYER */

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_GREEN_ONLY
/** Convert a full-resolution Bayer mosaic to half-resolution grayscale
 *  using only the green channel.  In a standard RGGB Bayer pattern the
 *  green pixels sit at positions (0,1) and (1,0) within each 2×2 block.
 *  Averaging these two samples gives the best-SNR single-channel
 *  representation because green has the highest quantum efficiency and
 *  there are two samples per block.
 *
 *  NOTE: This assumes an RGGB Bayer pattern (the standard darktable
 *  convention for raw mosaic data after rawprepare).  For GRBG, BGGR or
 *  GBRG patterns the green pixel offsets would differ, but darktable's
 *  raw pipeline normalises all patterns to RGGB before this stage.
 *
 *  Unlike _mosaic_to_grayscale (which averages all four RGGB pixels), this
 *  avoids mixing dissimilar spectral channels: a saturated red pixel cannot
 *  contaminate the green-derived luminance.  The result is photometrically
 *  consistent across exposures and free of demosaicing artefacts.
 *
 *  Output dimensions are (wd/2) × (ht/2).  Caller must free the result. */
static float *_mosaic_to_green_only(const float *mosaic,
                                    const int wd,
                                    const int ht,
                                    int *out_w,
                                    int *out_h)
{
  const int gw = wd / 2;
  const int gh = ht / 2;
  float *out = dt_alloc_align_float((size_t)gw * gh);
  if(!out) return NULL;

  // In an RGGB Bayer pattern the two green pixels in each 2×2 block
  // are at offsets (row=0,col=1) = Gr and (row=1,col=0) = Gb.
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < gh; y++)
    for(int x = 0; x < gw; x++)
    {
      const int mx = x * 2;
      const int my = y * 2;
      const float gr = mosaic[my * wd + mx + 1];        // row 0, col 1
      const float gb = mosaic[(my + 1) * wd + mx];      // row 1, col 0
      out[y * gw + x] = 0.5f * (gr + gb);
    }

  *out_w = gw;
  *out_h = gh;
  return out;
}
#endif /* HDR_ALIGN_L0_GREEN_ONLY */

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
/** Normalise each CFA sublattice of a Bayer mosaic to [0, 1] independently
 *  using a 99th-percentile stretch.
 *
 *  Without per-sublattice normalisation, the G channels (which capture
 *  roughly twice as much light as R or B in a typical Bayer sensor) have
 *  approximately twice the gradient amplitude of R/B channels after a
 *  stride-2 Sobel filter.  This Bayer-frequency amplitude pattern reduces
 *  the ECC correlation coefficient (ρ) to ≈ 0.28 regardless of alignment
 *  quality, because the pattern appears in both images and interferes with
 *  the signal of actual image edges.
 *
 *  By processing the four sublattices ((cx,cy) ∈ {(0,0),(0,1),(1,0),(1,1)})
 *  separately with their own 99th-percentile white-point, all channels fill
 *  the same [0, 1] range and the gradient amplitudes are equalised.  This
 *  is the key prerequisite for reliable ECC at L0.
 *
 *  NOTE: This must only be called on the full-resolution Bayer L0 data.
 *  At L1 and above the pyramid data is already grayscale-equivalent (2×2
 *  box-filter downsampling averages Bayer blocks), so _normalize_image_percentile
 *  suffices. */
static void _normalize_bayer_per_channel(float *bayer, const int wd, const int ht)
{
  for(int cy = 0; cy < 2; cy++)
    for(int cx = 0; cx < 2; cx++)
    {
      // Count pixels in this sublattice
      const int sw = (wd - cx + 1) / 2;
      const int sh = (ht - cy + 1) / 2;
      const size_t spix = (size_t)sw * sh;

      // Find maximum value in this sublattice
      float vmax = 0.0f;
      for(int y = cy; y < ht; y += 2)
        for(int x = cx; x < wd; x += 2)
          if(bayer[(size_t)y * wd + x] > vmax) vmax = bayer[(size_t)y * wd + x];

      if(vmax < 1e-12f) continue;

      // Build histogram for 99th-percentile white-point
      enum { NBINS = 4096 };
      size_t hist[NBINS] = { 0 };
      const float inv_vmax = (float)(NBINS - 1) / vmax;
      for(int y = cy; y < ht; y += 2)
        for(int x = cx; x < wd; x += 2)
        {
          const int bin = (int)(bayer[(size_t)y * wd + x] * inv_vmax);
          hist[CLAMP(bin, 0, NBINS - 1)]++;
        }

      const size_t target = (size_t)((double)spix * 0.99);
      size_t cumul = 0;
      int p99_bin = NBINS - 1;
      for(int b = 0; b < NBINS; b++)
      {
        cumul += hist[b];
        if(cumul >= target) { p99_bin = b; break; }
      }
      const float wp = (float)(p99_bin + 1) / (float)NBINS * vmax;
      if(wp < 1e-12f) continue;
      const float inv_wp = 1.0f / wp;

      DT_OMP_FOR(collapse(2))
      for(int y = cy; y < ht; y += 2)
        for(int x = cx; x < wd; x += 2)
          bayer[(size_t)y * wd + x] = CLAMP(bayer[(size_t)y * wd + x] * inv_wp, 0.0f, 1.0f);
    }
}

/** Compute the signed Sobel gradient sum (gx + gy) for a full-resolution
 *  Bayer mosaic image using CFA-aware stride-2 sampling.
 *
 *  At each output pixel (x, y), the Sobel kernel reads only from pixels that
 *  belong to the same CFA channel (i.e., at stride-2 offsets).  This avoids
 *  mixing different-channel values and preserves the edge structure within
 *  each sublattice.
 *
 *  This function must be called AFTER _normalize_bayer_per_channel() so that
 *  all four sublattices have comparable amplitude ranges.  Without that
 *  pre-normalisation, the G/R/B amplitude difference would corrupt ECC.
 *
 *  Border pixels within 2 pixels of the image edge are set to zero because
 *  the stride-2 stencil reads 2 pixels beyond the current position.
 *
 *  Caller must free the result. */
static float *_gradient_bayer_cfa_sobel(const float *bayer,
                                        const int wd,
                                        const int ht)
{
  const size_t npix = (size_t)wd * ht;
  float *out = dt_alloc_align_float(npix);
  if(!out) return NULL;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < ht; y++)
    for(int x = 0; x < wd; x++)
    {
      if(y < 2 || y >= ht - 2 || x < 2 || x >= wd - 2)
      {
        out[(size_t)y * wd + x] = 0.0f;
        continue;
      }
      // stride-2 Sobel: all 9 stencil points share the same CFA channel
      const float tl = bayer[(size_t)(y - 2) * wd + (x - 2)];
      const float tc = bayer[(size_t)(y - 2) * wd + x];
      const float tr = bayer[(size_t)(y - 2) * wd + (x + 2)];
      const float ml = bayer[(size_t)y * wd + (x - 2)];
      const float mr = bayer[(size_t)y * wd + (x + 2)];
      const float bl = bayer[(size_t)(y + 2) * wd + (x - 2)];
      const float bc = bayer[(size_t)(y + 2) * wd + x];
      const float br = bayer[(size_t)(y + 2) * wd + (x + 2)];
      const float gx = (-tl + tr - 2.0f * ml + 2.0f * mr - bl + br) * 0.125f;
      const float gy = (-tl - 2.0f * tc - tr + bl + 2.0f * bc + br) * 0.125f;
      out[(size_t)y * wd + x] = gx + gy;
    }
  return out;
}
#endif /* HDR_ALIGN_L0_FULL_CFA */

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

/** Warp an image by isotropic scale×rotation about the image centre.
 *  Translation is excluded so that the NCC slide search can handle it.
 *  Caller must free the returned buffer. */
static float *_warp_similarity_nocrop(const float *in,
                                      const int w,
                                      const int h,
                                      const float angle,
                                      const float scale)
{
  float *out = dt_alloc_align_float((size_t)w * h);
  if(!out) return NULL;

  const float cx = (w - 1) * 0.5f;
  const float cy = (h - 1) * 0.5f;
  // Inverse rotation for backward mapping; inverse scale = 1/scale
  const float cos_a = cosf(-angle);
  const float sin_a = sinf(-angle);
  const float inv_s = 1.0f / scale;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      // Shift to centre, apply inverse scale, then inverse rotation.
      const float ox = ((float)x - cx) * inv_s;
      const float oy = ((float)y - cy) * inv_s;
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
      }
      else
      {
        out[y * w + x] = 0.0f;
      }
    }

  return out;
}

/** Initialise a backward-mapping similarity homography (scale + rotation +
 *  translation) about the image centre.  When scale=1 this reduces exactly
 *  to _homography_from_euclidean.  The backward mapping divides the centred
 *  offset by scale before applying the inverse rotation, so a warped pixel
 *  at distance r from the centre in the destination came from distance r/scale
 *  in the source — matching the forward model where img features appear
 *  scale× larger than in ref. */
static void _homography_from_similarity(const float tx,
                                        const float ty,
                                        const float angle,
                                        const float scale,
                                        const int w,
                                        const int h,
                                        float H[HDR_ALIGN_H_NPARAM])
{
  const float cx = (w - 1) * 0.5f;
  const float cy = (h - 1) * 0.5f;
  const float ca = cosf(angle);
  const float sa = sinf(angle);
  const float inv_s = 1.0f / scale;

  H[0] =  ca * inv_s;
  H[1] =  sa * inv_s;
  H[2] = -ca * inv_s * (tx + cx) - sa * inv_s * (ty + cy) + cx;
  H[3] = -sa * inv_s;
  H[4] =  ca * inv_s;
  H[5] =  sa * inv_s * (tx + cx) - ca * inv_s * (ty + cy) + cy;
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

#if HDR_ALIGN_L0_MODE != HDR_ALIGN_L0_FULL_CFA
/** Scale a homography from half-resolution local coordinates to
 *  full-resolution Bayer coordinates.  Used by AVG_BAYER and GREEN_ONLY
 *  modes where L0 is half-resolution.  The rotation/scale/shear components
 *  (H[0],H[1],H[3],H[4]) are dimensionless and unchanged.  The
 *  translation (H[2],H[5]) and perspective (H[6],H[7]) scale exactly
 *  like _homography_scale_to_finer(). */
static void _homography_local_to_full(const float H_local[HDR_ALIGN_H_NPARAM],
                                      float H_full[HDR_ALIGN_H_NPARAM])
{
  memcpy(H_full, H_local, sizeof(float) * HDR_ALIGN_H_NPARAM);
  H_full[2] *= 2.0f;  // tx
  H_full[5] *= 2.0f;  // ty
  H_full[6] *= 0.5f;  // p1
  H_full[7] *= 0.5f;  // p2
}
#endif /* HDR_ALIGN_L0_MODE != HDR_ALIGN_L0_FULL_CFA */

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

/** Normalise a non-negative image to [0, 1] in-place using a
 *  99th-percentile stretch.
 *
 *  Many image distributions are right-skewed.  A naive global min-max stretch
 *  would let a few bright outliers compress the useful range.
 *
 *  This function clips the top 1 % of pixel values to 1.0 and stretches the
 *  remaining range linearly to [0, 1].  When used on raw pixel images before
 *  gradient computation, it brings dark and bright exposures to a comparable
 *  scale and reduces the influence of saturated highlights. */
static void _normalize_image_percentile(float *img, const size_t npix)
{
  if(!img || npix == 0) return;

  // Find the global maximum to set histogram range.
  float vmax = 0.0f;
  for(size_t i = 0; i < npix; i++)
    if(img[i] > vmax) vmax = img[i];

  if(vmax < 1e-12f) return;

  // Build a 4096-bin histogram over [0, vmax].
  enum { NBINS = 4096 };
  size_t hist[NBINS] = { 0 };
  const float inv_vmax = (float)(NBINS - 1) / vmax;
  for(size_t i = 0; i < npix; i++)
  {
    const int bin = (int)(img[i] * inv_vmax);
    hist[CLAMP(bin, 0, NBINS - 1)]++;
  }

  // Walk the histogram to find the 99th-percentile bin as white point.
  const size_t target = (size_t)((double)npix * 0.99);
  size_t cumul = 0;
  int p99_bin = NBINS - 1;
  for(int b = 0; b < NBINS; b++)
  {
    cumul += hist[b];
    if(cumul >= target) { p99_bin = b; break; }
  }
  const float white_point = (float)(p99_bin + 1) / (float)NBINS * vmax;
  if(white_point < 1e-12f) return;

  // Stretch [0, white_point] → [0, 1], clamp values above white_point.
  const float inv_wp = 1.0f / white_point;
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    img[i] = CLAMP(img[i] * inv_wp, 0.0f, 1.0f);
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

/** Compute unsigned Sobel gradient magnitude sqrt(gx² + gy²) image.
 *  Used for quality-score computation (ρ) where sign cancellation in the
 *  signed sum would drive PCC to near-zero for natural images with diverse
 *  edge orientations, even when the images are well-aligned.  Caller must
 *  free the result. */
static float *_gradient_sobel_magnitude(const float *img, const int w, const int h)
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

  // Compute magnitude in-place into gx.
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    gx[i] = sqrtf(gx[i] * gx[i] + gy[i] * gy[i]);

  dt_free_align(gy);
  return gx;
}


/** Mask a gradient image by zeroing out pixels whose corresponding
 *  [0,1]-normalised intensity is outside [lo, hi].  Pixels near the
 *  sensor black level (underexposed) or near saturation (overexposed)
 *  produce unreliable gradients that would mislead ECC alignment.
 *  By setting those gradient values to zero they do not contribute to
 *  the correlation, so ECC optimises over valid regions only. */
static void _mask_gradient_by_intensity(float *grad,
                                        const float *mask,
                                        const size_t npix)
{
  if(!grad || !mask) return;
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    grad[i] *= mask[i];
}

// Thresholds for the gradient validity mask.  After percentile
// normalisation the image is in [0, 1].  Values below MASK_LO are
// underexposed (essentially black) and values above MASK_HI are
// saturated / clipped.  Both produce unreliable gradients.
#define HDR_ALIGN_GRADIENT_MASK_LO 0.01f
#define HDR_ALIGN_GRADIENT_MASK_HI 0.99f

/** Normalise a signed gradient image in-place by its mean absolute value.
 *  g[i] = g[i] / (mean(|g|) + ε)
 *  This prevents one image from dominating due to exposure differences,
 *  while preserving the sign of gradients. */
static void _normalize_gradient_mad(float *g, const size_t npix)
{
  if(!g || npix == 0) return;

  double sum_abs = 0.0;
  DT_OMP_FOR(reduction(+:sum_abs))
  for(size_t i = 0; i < npix; i++)
    sum_abs += (double)fabsf(g[i]);

  const float mean_abs = (float)(sum_abs / (double)npix);
  const float inv_scale = 1.0f / (mean_abs + 1e-7f);

  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    g[i] *= inv_scale;
}

/** Apply in-place Gaussian pre-filter to a single-channel image.
 *  Uses darktable's existing IIR Gaussian implementation for efficiency.
 *  Skipped when the shortest dimension is below HDR_ALIGN_PREFILTER_MIN_DIM
 *  (at that size the 2× downsampling already provides sufficient smoothing). */
static void _gaussian_prefilter(float *img, const int w, const int h,
                                const float sigma)
{
  if(!img || sigma <= 0.0f) return;
  if(MIN(w, h) < HDR_ALIGN_PREFILTER_MIN_DIM) return;

  dt_gaussian_mean_blur(img, w, h, 1, sigma);
}

/** Normalise gradient magnitude to [0,1] using a 99th-percentile stretch,
 *  then apply power scaling and a low-end threshold.
 *
 *  Steps:
 *    1. Find the 99th-percentile of the magnitude → white_point
 *    2. Divide every pixel by white_point, clamp to [0,1]
 *    3. Apply power scaling: m[i] = m[i]^power  (compresses strong gradients,
 *       lifts weak ones, giving more uniform weight)
 *    4. Apply threshold: m[i] < threshold → 0  (featureless regions excluded)
 *
 *  This replaces the previous raw-intensity percentile normalisation and
 *  moves the normalisation to operate on gradient magnitude instead,
 *  which is more directly related to alignment feature quality. */
static void _normalize_magnitude_percentile_power(float *mag,
                                                   const size_t npix,
                                                   const float power,
                                                   const float threshold)
{
  if(!mag || npix == 0) return;

  // Find the global maximum to set histogram range.
  float vmax = 0.0f;
  for(size_t i = 0; i < npix; i++)
    if(mag[i] > vmax) vmax = mag[i];

  if(vmax < 1e-12f) return;

  // Build a 4096-bin histogram over [0, vmax].
  enum { NBINS = 4096 };
  size_t hist[NBINS] = { 0 };
  const float inv_vmax = (float)(NBINS - 1) / vmax;
  for(size_t i = 0; i < npix; i++)
  {
    const int bin = (int)(mag[i] * inv_vmax);
    hist[CLAMP(bin, 0, NBINS - 1)]++;
  }

  // Walk the histogram to find the 99th-percentile bin as white point.
  const size_t target = (size_t)((double)npix * 0.99);
  size_t cumul = 0;
  int p99_bin = NBINS - 1;
  for(int b = 0; b < NBINS; b++)
  {
    cumul += hist[b];
    if(cumul >= target) { p99_bin = b; break; }
  }
  const float white_point = (float)(p99_bin + 1) / (float)NBINS * vmax;
  if(white_point < 1e-12f) return;

  // Stretch [0, white_point] → [0, 1], apply power scaling, then threshold.
  const float inv_wp = 1.0f / white_point;
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
  {
    float v = CLAMP(mag[i] * inv_wp, 0.0f, 1.0f);
    if(power != 1.0f) v = powf(v, power);
    mag[i] = (v >= threshold) ? v : 0.0f;
  }
}

/** Build a gradient-magnitude-based mask for ECC input images.
 *
 *  The mask is 1 where the normalised+power-scaled gradient magnitude is
 *  above zero (i.e., passed the threshold in _normalize_magnitude_percentile_power)
 *  AND the raw-intensity pixel is not saturated/underexposed.
 *
 *  @p mag_norm  Percentile-normalised, power-scaled gradient magnitude
 *               (modified in-place by _normalize_magnitude_percentile_power).
 *  @p raw_norm  Raw pixel values normalised to [0,1] by percentile stretch
 *               (used to detect saturated/underexposed pixels).
 *  @p lo, @p hi Intensity thresholds for the validity mask.
 *
 *  Returns a newly allocated mask buffer. Caller must free. */
static float *_build_gradient_mask(const float *mag_norm,
                                   const float *raw_norm,
                                   const size_t npix,
                                   const float lo,
                                   const float hi)
{
  float *mask = dt_alloc_align_float(npix);
  if(!mask) return NULL;

  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
  {
    // Gradient magnitude above threshold AND intensity in valid range
    mask[i] = (mag_norm[i] > 0.0f && raw_norm[i] >= lo && raw_norm[i] <= hi)
              ? 1.0f : 0.0f;
  }
  return mask;
}

/** Complete per-level gradient pipeline following the restructured order:
 *
 *    1. Spatial pre-filtering with Gaussian blur (skip at coarse levels)
 *    2. Gradient extraction: compute gx, gy and gradient magnitude
 *    3. Normalize (percentile), threshold and power scaling of magnitude
 *    4. Mask construction (magnitude-based + saturated pixel removal)
 *    5. Prepare ECC input: signed sum gx+gy with mask applied
 *
 *  The caller provides a copy of the pyramid level data (will be modified
 *  in-place by the blur).  Returns the masked gradient image suitable for
 *  ECC.  Also optionally returns the mask via @p out_mask for the caller
 *  to apply to the warped image gradient.
 *
 *  @param level_data  Pyramid level pixel data (modified in-place by blur)
 *  @param w           Width of the level
 *  @param h           Height of the level
 *  @param sigma       Gaussian pre-filter sigma (0 to skip)
 *  @param out_mask    [out, optional] Returns the validity mask if non-NULL
 *  @return            Masked, normalised gradient image (caller frees), or NULL
 */
static float *_compute_level_gradient(float *level_data,
                                      const int w,
                                      const int h,
                                      const float sigma,
                                      float **out_mask)
{
  const size_t npix = (size_t)w * h;

  // Step 1: Spatial pre-filtering with Gaussian blur.
  //         Suppresses sensor noise and Bayer-residual high-frequency
  //         artefacts before gradient extraction.
  _gaussian_prefilter(level_data, w, h, sigma);

  // Step 2: Gradient extraction — compute Sobel gx, gy.
  float *gx = dt_alloc_align_float(npix);
  float *gy = dt_alloc_align_float(npix);
  if(!gx || !gy)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    return NULL;
  }
  _compute_gradients(level_data, w, h, gx, gy);

  // Compute gradient magnitude sqrt(gx² + gy²).
  float *mag = dt_alloc_align_float(npix);
  if(!mag)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    return NULL;
  }
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    mag[i] = sqrtf(gx[i] * gx[i] + gy[i] * gy[i]);

  // Step 3: Normalize (percentile), threshold and power scaling of magnitude.
  _normalize_magnitude_percentile_power(mag, npix,
                                        HDR_ALIGN_GRAD_MAG_POWER,
                                        HDR_ALIGN_GRAD_MAG_THRESHOLD);

  // Step 4: Mask construction.
  //         Normalize the raw intensity to [0,1] for saturated pixel detection.
  //         We need a separate copy because the Gaussian blur modified level_data.
  //         We use the (blurred) level_data for intensity-based mask — the blur
  //         ensures we don't flag isolated hot pixels, only truly saturated regions.
  float *raw_norm = dt_alloc_align_float(npix);
  if(!raw_norm)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    dt_free_align(mag);
    return NULL;
  }
  memcpy(raw_norm, level_data, sizeof(float) * npix);
  _normalize_image_percentile(raw_norm, npix);

  float *mask = _build_gradient_mask(mag, raw_norm, npix,
                                     HDR_ALIGN_GRADIENT_MASK_LO,
                                     HDR_ALIGN_GRADIENT_MASK_HI);
  dt_free_align(raw_norm);
  dt_free_align(mag);

  if(!mask)
  {
    dt_free_align(gx);
    dt_free_align(gy);
    return NULL;
  }

  // Step 5: Prepare ECC input — combine gx+gy (signed sum preserves direction
  //         information for ECC), then normalise and apply the mask.
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    gx[i] = gx[i] + gy[i];
  dt_free_align(gy);

  _normalize_gradient_mad(gx, npix);
  _mask_gradient_by_intensity(gx, mask, npix);

  if(out_mask)
    *out_mask = mask;
  else
    dt_free_align(mask);

  return gx;
}

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
/** Complete per-level gradient pipeline for full-resolution CFA at L0.
 *
 *  Same restructured pipeline as _compute_level_gradient but with
 *  CFA-aware processing:
 *    1. Gaussian pre-filter (if dimensions allow)
 *    2. Per-sublattice percentile normalisation
 *    3. CFA-aware stride-2 Sobel gradient
 *    4. MAD normalisation
 *    5. Mask from gradient magnitude + intensity validity
 *
 *  The CFA-aware Sobel already computes gx+gy internally (stride-2),
 *  so we use gradient magnitude from that output for masking.
 */
static float *_compute_level_gradient_cfa(float *level_data,
                                          const int w,
                                          const int h,
                                          const float sigma,
                                          float **out_mask)
{
  const size_t npix = (size_t)w * h;

  // Step 1: Spatial pre-filtering.
  // For CFA data we apply the Gaussian blur to each sublattice independently
  // to avoid cross-channel mixing.  However, dt_gaussian_mean_blur operates
  // on the full image — at full resolution, the Bayer-frequency content is
  // already suppressed by the stride-2 Sobel.  We still blur the full image
  // here as a noise reduction step; the per-sublattice normalisation that
  // follows ensures channel balance is maintained.
  _gaussian_prefilter(level_data, w, h, sigma);

  // Step 2: Per-sublattice percentile normalisation (CFA-specific).
  _normalize_bayer_per_channel(level_data, w, h);

  // Step 3: CFA-aware stride-2 Sobel gradient (returns gx+gy).
  float *grad = _gradient_bayer_cfa_sobel(level_data, w, h);
  if(!grad) return NULL;

  // For mask construction we need the gradient magnitude.
  // The CFA Sobel returns gx+gy (signed sum), so we compute the absolute
  // value as a proxy for magnitude (adequate for masking purposes).
  float *mag = dt_alloc_align_float(npix);
  if(!mag) { dt_free_align(grad); return NULL; }
  DT_OMP_FOR()
  for(size_t i = 0; i < npix; i++)
    mag[i] = fabsf(grad[i]);

  // Step 4: Normalize, threshold and power-scale the magnitude.
  _normalize_magnitude_percentile_power(mag, npix,
                                        HDR_ALIGN_GRAD_MAG_POWER,
                                        HDR_ALIGN_GRAD_MAG_THRESHOLD);

  // Step 5: Mask from magnitude + intensity validity.
  float *raw_norm = dt_alloc_align_float(npix);
  if(!raw_norm) { dt_free_align(grad); dt_free_align(mag); return NULL; }
  memcpy(raw_norm, level_data, sizeof(float) * npix);
  // For CFA, level_data is already per-sublattice normalised to [0,1].
  // Use it directly for the intensity mask.
  float *mask = _build_gradient_mask(mag, raw_norm, npix,
                                     HDR_ALIGN_GRADIENT_MASK_LO,
                                     HDR_ALIGN_GRADIENT_MASK_HI);
  dt_free_align(raw_norm);
  dt_free_align(mag);

  if(!mask) { dt_free_align(grad); return NULL; }

  // Step 6: MAD normalise and apply mask.
  _normalize_gradient_mad(grad, npix);
  _mask_gradient_by_intensity(grad, mask, npix);

  if(out_mask)
    *out_mask = mask;
  else
    dt_free_align(mask);

  return grad;
}
#endif /* HDR_ALIGN_L0_FULL_CFA */

/** Write a single-channel float image as a grayscale PFM file for debugging.
 *  Only active when the DT_DEBUG_VERBOSE flag is set.
 *  Files are written to /tmp as "hdr_align_grad_<label>.pfm".
 *  Gradient images fed to ECC are already normalised with
 *  _normalize_gradient_mad(), so the values are well-distributed and
 *  no additional contrast enhancement is needed here. */
static void _debug_export_gradient_pfm(const float *grad,
                                       const int w,
                                       const int h,
                                       const char *label)
{
  if(!(darktable.unmuted & DT_DEBUG_VERBOSE)) return;
  if(!grad || w <= 0 || h <= 0) return;

  char fname[64];
  snprintf(fname, sizeof(fname), "hdr_align_grad_%s.pfm", label);
  char *path = g_build_filename("/tmp", fname, NULL);
  dt_write_pfm(path, w, h, grad, sizeof(float));
  dt_print(DT_DEBUG_VERBOSE,
           "[hdr_align] exported gradient image (%dx%d) → %s", w, h, path);
  g_free(path);
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

  // --- Pass 2: norms, mean Jacobian, projection coefficients, and ρ ---
  // 3-DOF Jacobian: J = (J_θ, J_tx, J_ty)
  //   J_θ  = s·(gx·(−sin θ·xn + cos θ·yn) + gy·(−cos θ·xn − sin θ·yn))
  //   J_tx = s·gx
  //   J_ty = s·gy
  //
  // The projection coefficient for parameter k is:
  //   proj_coeff[k] = Σ wgt·tw·(J[k] − mean_J[k]) / norm2_w
  // Since Σ wgt·tw = 0 by definition of mean_w, this simplifies to:
  //   proj_coeff[k] = Σ wgt·tw·J[k] / norm2_w
  // so we can accumulate sJw[k] = Σ wgt·tw·J[k] alongside the existing
  // pass 2 sums and eliminate the former pass 3 sweep entirely.
  double norm2_r = 0.0, norm2_w = 0.0;
  double sum_J0 = 0.0, sum_J1 = 0.0, sum_J2 = 0.0;
  double sJw0 = 0.0, sJw1 = 0.0, sJw2 = 0.0;
  double dot_rw = 0.0;

  DT_OMP_FOR(collapse(2) reduction(+:norm2_r, norm2_w, sum_J0, sum_J1, sum_J2, sJw0, sJw1, sJw2, dot_rw))
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

      sJw0 += wgt * tw * J0;
      sJw1 += wgt * tw * J1;
      sJw2 += wgt * tw * J2;
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

  // Derive projection coefficients from the pass 2 sums (no separate sweep).
  const double mean_J[3] = { sum_J0 / sum_weight,
                              sum_J1 / sum_weight,
                              sum_J2 / sum_weight };

  // proj_coeff[k] = sJw[k] / norm2_w  (the mean_J correction term vanishes
  // because Σ wgt·tw = 0 by definition of mean_w).
  const double proj_coeff[3] = { sJw0 / norm2_w,
                                  sJw1 / norm2_w,
                                  sJw2 / norm2_w };

  // --- Pass 3 (formerly Pass 4): 3×3 Hessian and RHS ---
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
  // Preserve the row norm (= inv_scale) that _homography_from_similarity
  // encodes in the upper-left 2×2 block.  The 3-DOF solver optimises only
  // rotation and translation; any scale component in H must be carried
  // forward unchanged so the 6-DOF escalation at the finest level starts
  // from the correct scale basin.  For scale=1 (normal case) sc=1 and
  // behaviour is identical to before.
  const double sc = sqrt(cos_t * cos_t + sin_t * sin_t);
  const double theta_old = atan2(sin_t, cos_t);
  const double theta_new = theta_old + dtheta;

  Hn[0] =  (float)(sc * cos(theta_new));
  Hn[1] =  (float)(sc * sin(theta_new));
  Hn[2] += (float)dtx;
  Hn[3] = -(float)(sc * sin(theta_new));
  Hn[4] =  (float)(sc * cos(theta_new));
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
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   ECC failed at iteration %d", iter);
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      return -1.0f;
    }

    if(update < HDR_ALIGN_ECC_EPSILON)
    {
      dt_print(DT_DEBUG_HDRMERGE,
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
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   ECC stalled at iteration %d (update=%.6f, best=%.6f)",
               iter, update, best_update);
      return best_update;
    }
  }

  // Max iterations reached: restore best H seen so far.
  memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge]   ECC did not converge in %d iterations",
           HDR_ALIGN_ECC_MAX_ITER);
  return 0.0f;
}

// ---------------------------------------------------------------------------
// Adaptive DOF escalation: 3-DOF → 6-DOF → 8-DOF
// ---------------------------------------------------------------------------

/** Compute the weighted ECC alignment score ρ from *intensity* images.
 *
 *  Both @p ref_intensity and @p img_intensity must be preprocessed intensity
 *  images (e.g. percentile-normalised + log-transformed).  The function warps
 *  img_intensity by H, then computes Sobel gradient *magnitude* images for
 *  both ref and the warped img, and returns their Pearson correlation.
 *
 *  Using gradient magnitude (always ≥ 0, rotation-invariant) instead of the
 *  signed gx+gy sum avoids the sign-cancellation artefact that drives the PCC
 *  to near zero for natural images with diverse edge orientations even when
 *  the images are perfectly aligned.
 *
 *  Returns the correlation coefficient in [-1, 1], or -2 on failure. */
static float _ecc_compute_rho(const float *ref_intensity,
                               const float *img_intensity,
                               const int w,
                               const int h,
                               const float H[HDR_ALIGN_H_NPARAM])
{
  const size_t npix = (size_t)w * h;
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx = ((double)w - 1.0) * 0.5;
  const double cy = ((double)h - 1.0) * 0.5;

  // Warp img_intensity by H and record the valid-pixel mask.
  float *mask = dt_alloc_align_float(npix);
  if(!mask) return -2.0f;

  float *warped = _warp_homography(img_intensity, w, h, H, mask);
  if(!warped)
  {
    dt_free_align(mask);
    return -2.0f;
  }

  // Compute gradient magnitude images: ref_mag from ref_intensity,
  // warp_mag from the already-warped image.  Magnitude is always ≥ 0 and
  // is invariant to the sign convention of the Sobel filter, so the PCC
  // correctly reflects structural similarity regardless of edge orientation.
  float *ref_mag = _gradient_sobel_magnitude(ref_intensity, w, h);
  float *warp_mag = _gradient_sobel_magnitude(warped, w, h);
  dt_free_align(warped);

  if(!ref_mag || !warp_mag)
  {
    dt_free_align(ref_mag);
    dt_free_align(warp_mag);
    dt_free_align(mask);
    return -2.0f;
  }

  // Single-pass weighted covariance formula.
  // Accumulates sum_r, sum_w, sum_rr, sum_ww, sum_rw and sum_weight in one
  // sweep, then derives norm2_r, norm2_w, dot_rw via the identity:
  //   norm2_r = sum_rr - sum_r² / sum_weight
  // Gradient magnitudes are non-negative and well-scaled (mean ≈ 1 after
  // normalization), so catastrophic cancellation is not a concern.
  double sum_r = 0.0, sum_w = 0.0, sum_weight = 0.0;
  double sum_rr = 0.0, sum_ww = 0.0, sum_rw = 0.0;
  long nvalid = 0;

  DT_OMP_FOR(collapse(2) reduction(+:sum_r, sum_w, sum_weight, sum_rr, sum_ww, sum_rw, nvalid))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t i = (size_t)y * w + x;
      // The valid-pixel mask comes from the warp; border pixels where the
      // Sobel kernel reaches outside the image are zeroed by _compute_gradients,
      // but they still have mask=1 if the warp mapped them inside the image.
      // That is fine: zero-gradient boundary pixels contribute correctly.
      if(mask[i] > 0.5f)
      {
        const double xn = ((double)x - cx) / s;
        const double yn = ((double)y - cy) / s;
        const double wgt = _ecc_spatial_weight(xn, yn);
        const double rv = (double)ref_mag[i];
        const double wv = (double)warp_mag[i];
        sum_r      += wgt * rv;
        sum_w      += wgt * wv;
        sum_rr     += wgt * rv * rv;
        sum_ww     += wgt * wv * wv;
        sum_rw     += wgt * rv * wv;
        sum_weight += wgt;
        nvalid++;
      }
    }

  dt_free_align(ref_mag);
  dt_free_align(warp_mag);
  dt_free_align(mask);

  if(nvalid < (long)(npix * HDR_ALIGN_ECC_MIN_VALID_FRAC) || sum_weight < 1e-12)
    return -2.0f;

  // Compute variance / covariance from aggregated sums.
  const double norm2_r = sum_rr - sum_r * sum_r / sum_weight;
  const double norm2_w = sum_ww - sum_w * sum_w / sum_weight;
  const double dot_rw  = sum_rw - sum_r * sum_w / sum_weight;

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

  // --- Pass 2: norms, mean Jacobian, projection coefficients, and ρ ---
  // Use scalar accumulators for OMP reduction (up to 8 DOF).
  //
  // Projection coefficient identity (analogous to the 3-DOF case):
  //   proj_coeff[k] = Σ wgt·tw·(J[k] − mean_J[k]) / norm2_w
  // Since Σ wgt·tw = 0 by definition of mean_w:
  //   proj_coeff[k] = Σ wgt·tw·J[k] / norm2_w
  // We therefore add sJw[k] = Σ wgt·tw·J[k] to this pass and eliminate
  // the former pass 3 sweep entirely.
  double norm2_r = 0.0, norm2_w = 0.0, dot_rw = 0.0;
  double sJ0 = 0.0, sJ1 = 0.0, sJ2 = 0.0, sJ3 = 0.0;
  double sJ4 = 0.0, sJ5 = 0.0, sJ6 = 0.0, sJ7 = 0.0;
  double sJw0 = 0.0, sJw1 = 0.0, sJw2 = 0.0, sJw3 = 0.0;
  double sJw4 = 0.0, sJw5 = 0.0, sJw6 = 0.0, sJw7 = 0.0;

  DT_OMP_FOR(collapse(2) reduction(+:norm2_r, norm2_w, dot_rw, sJ0, sJ1, sJ2, sJ3, sJ4, sJ5, sJ6, sJ7, sJw0, sJw1, sJw2, sJw3, sJw4, sJw5, sJw6, sJw7))
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

      sJw0 += wgt * tw * J[0];  sJw1 += wgt * tw * J[1];
      sJw2 += wgt * tw * J[2];  sJw3 += wgt * tw * J[3];
      sJw4 += wgt * tw * J[4];  sJw5 += wgt * tw * J[5];
      sJw6 += wgt * tw * J[6];  sJw7 += wgt * tw * J[7];
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

  // Derive mean_J and proj_coeff from pass 2 sums (no separate sweep).
  double mean_J[HDR_ALIGN_H_NPARAM];
  for(int k = 0; k < ndof; k++)
    mean_J[k] = sum_J[k] / sum_weight;

  // proj_coeff[k] = sJw[k] / norm2_w  (the mean_J correction term vanishes
  // because Σ wgt·tw = 0 by definition of mean_w).
  double proj_coeff[HDR_ALIGN_H_NPARAM] = { 0 };
  const double sJw[HDR_ALIGN_H_NPARAM] = { sJw0, sJw1, sJw2, sJw3, sJw4, sJw5, sJw6, sJw7 };
  for(int k = 0; k < ndof; k++)
    proj_coeff[k] = sJw[k] / norm2_w;

  // --- Pass 3 (formerly Pass 4): n×n Hessian and RHS ---
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
/** Run iterative higher-DOF ECC at a single pyramid level.
 *  @p ref_grad and @p img_grad are the signed-gradient images used for
 *  the ECC optimisation.  @p ref_intensity and @p img_intensity are the
 *  corresponding pre-Sobel intensity images used by _ecc_compute_rho for
 *  reliable gradient-magnitude quality scoring.
 *
 *  Returns the weighted magnitude-correlation ρ after convergence,
 *  or -2 on failure. */
static float _ecc_refine_level_higher_dof(const float *ref_grad,
                                            const float *img_grad,
                                            const float *ref_intensity,
                                            const float *img_intensity,
                                            const int w,
                                            const int h,
                                            float H[HDR_ALIGN_H_NPARAM],
                                            const int ndof)
{
  float best_update = FLT_MAX;
  int stall_count = 0;

  // Keep a copy of H at the iteration that produced the best update metric
  // so we can restore it on stall or max-iterations exit.  Without this,
  // the optimizer can oscillate for all 50 iterations and leave H in a
  // state that drifts far from the convergence basin (particularly the
  // perspective parameters which accumulate +/-0.001 per iteration and can
  // exceed the +/-0.02 sanity limit after ~20+ oscillating steps).
  float H_best[HDR_ALIGN_H_NPARAM];
  memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);

  for(int iter = 0; iter < HDR_ALIGN_ESCALATION_MAX_ITER; iter++)
  {
    double cond_est = 0.0;
    const float update = _ecc_iteration_higher_dof(ref_grad, img_grad, w, h, H,
                                                    ndof, &cond_est);

    if(update < 0.0f)
    {
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   %d-DOF ECC failed at iteration %d",
               ndof, iter);
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      return -2.0f;
    }

    // Reject if the Hessian is poorly conditioned.
    if(cond_est > HDR_ALIGN_ESCALATION_MAX_COND)
    {
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   %d-DOF ECC rejected: Hessian cond=%.1e > %.1e",
               ndof, cond_est, HDR_ALIGN_ESCALATION_MAX_COND);
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      return -2.0f;
    }

    if(update < HDR_ALIGN_ESCALATION_EPSILON)
    {
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   %d-DOF ECC converged at iteration %d (update=%.6f, cond=%.2f)",
               ndof, iter, update, cond_est);
      break;
    }

    // Plateau / stall detection (mirrors the 3-DOF solver).
    // If the update metric has not improved for HDR_ALIGN_ECC_PATIENCE
    // consecutive iterations, the optimizer is stuck above the convergence
    // threshold.  Continuing will only cause parameters (especially the
    // perspective terms in 8-DOF mode) to oscillate and drift beyond the
    // sanity-check bounds.  Restore H to the best-seen state and stop.
    if(update < best_update)
    {
      best_update = update;
      stall_count = 0;
      memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);
    }
    else if(++stall_count >= HDR_ALIGN_ECC_PATIENCE)
    {
      memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge]   %d-DOF ECC stalled at iteration %d "
               "(update=%.6f, best=%.6f, cond=%.2f)",
               ndof, iter, update, best_update, cond_est);
      break;
    }
  }

  return _ecc_compute_rho(ref_intensity, img_intensity, w, h, H);
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

  // Diagonal elements should be within the coarse scale search range.
  // Lens breathing and aperture changes can introduce up to ±45% scale.
  if(Hn[0] < 0.70f || Hn[0] > 1.45f) return FALSE;
  if(Hn[4] < 0.70f || Hn[4] > 1.45f) return FALSE;

  // Off-diagonal: allow more shear than the rigid check.
  if(fabsf(Hn[1]) > 0.25f) return FALSE;
  if(fabsf(Hn[3]) > 0.25f) return FALSE;

  // Scale consistency: the geometric-mean scale sqrt(det(A)) must stay
  // within HDR_ALIGN_ESCALATION_MAX_SCALE_DEVIATION of 1.0.  Using the
  // geometric mean (rather than raw det) gives a symmetric, per-axis bound.
  const float det_2x2 = Hn[0] * Hn[4] - Hn[1] * Hn[3];
  if(det_2x2 <= 0.0f) return FALSE;
  const float geom_scale = sqrtf(det_2x2);
  if(fabsf(geom_scale - 1.0f) > HDR_ALIGN_ESCALATION_MAX_SCALE_DEVIATION) return FALSE;

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

/** Attempt adaptive DOF escalation at the designated escalation level.
 *  Measures the weighted ECC score of the current 3-DOF result and
 *  selectively tries higher-DOF models when the rigid fit is insufficient.
 *
 *  @p ref_grad / @p img_grad   : signed-gradient images for ECC optimisation.
 *  @p ref_intensity / @p img_intensity : pre-Sobel intensity images for ρ scoring.
 *
 *  Escalation path: 3-DOF → 6-DOF affine → 8-DOF projective.
 *  Each step is only attempted when the previous step improved ρ.
 *  If a step does not improve, escalation stops and falls back to
 *  the last accepted model.  Each accepted step must also pass a
 *  Hessian conditioning check and a geometric sanity check.
 *
 *  When escalation succeeds, the caller sets current_dof to the
 *  accepted DOF level and all subsequent finer pyramid levels use
 *  the higher-DOF solver.
 *
 *  Returns the best ρ achieved (either from the accepted DOF level or the
 *  original 3-DOF result).  Returns -2 on measurement failure. */
static float _try_dof_escalation(const float *ref_grad,
                                 const float *img_grad,
                                 const float *ref_intensity,
                                 const float *img_intensity,
                                 const int w,
                                 const int h,
                                 const float rho_3dof,
                                 float H[HDR_ALIGN_H_NPARAM])
{
  // Save the 3-DOF result so we can fall back.
  float H_3dof[HDR_ALIGN_H_NPARAM];
  memcpy(H_3dof, H, sizeof(H_3dof));

  // --- Stage 1: try 6-DOF affine ---
  float H_6dof[HDR_ALIGN_H_NPARAM];
  memcpy(H_6dof, H_3dof, sizeof(H_6dof));

  const float rho_6dof = _ecc_refine_level_higher_dof(ref_grad, img_grad,
                                                       ref_intensity, img_intensity,
                                                       w, h, H_6dof, 6);

  const float improvement_6 = rho_6dof - rho_3dof;
  const gboolean sane_6 = _homography_is_sane_escalated(H_6dof, w, h, 6);

  // Shear check: the symmetric part of the 2×2 submatrix measures pure
  // shear.  Physical camera movements (rotation, translation, zoom,
  // moderate perspective) produce negligible shear (< 0.004).  Large
  // values indicate the optimizer is fitting non-geometric patterns such
  // as exposure gradients, vignetting, or moving scene content (waves)
  // as spurious affine deformation.
  const float shear_6 = 0.5f * fabsf(H_6dof[1] + H_6dof[3]);
  const gboolean low_shear_6 = shear_6 <= HDR_ALIGN_ESCALATION_MAX_SHEAR;

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] DOF escalation: 6-DOF ρ=%.4f (Δρ=%+.4f vs 3-DOF, sane=%d, shear=%.4f)",
           rho_6dof, improvement_6, sane_6, shear_6);

  const gboolean accept_6 = sane_6
                             && low_shear_6
                             && rho_6dof > rho_3dof
                             && improvement_6 >= HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT;

  // --- Stage 2: try 8-DOF projective only if 6-DOF improved ---
  // We only escalate further when the previous step produced a measurable
  // improvement.  If 6-DOF did not help, the starting point is unlikely
  // to support even more degrees of freedom.
  if(!accept_6)
  {
    // 6-DOF did not improve — keep 3-DOF and stop.
    memcpy(H, H_3dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_HDRMERGE,
             "[hdr_merge] DOF escalation: 6-DOF no improvement, keeping 3-DOF");
    return rho_3dof;
  }

  // 6-DOF improved — try 8-DOF starting from the 6-DOF result.
  float H_8dof[HDR_ALIGN_H_NPARAM];
  memcpy(H_8dof, H_6dof, sizeof(H_8dof));

  const float rho_8dof = _ecc_refine_level_higher_dof(ref_grad, img_grad,
                                                       ref_intensity, img_intensity,
                                                       w, h, H_8dof, 8);

  const float improvement_8 = rho_8dof - rho_6dof;
  const gboolean sane_8 = _homography_is_sane_escalated(H_8dof, w, h, 8);
  const float shear_8 = 0.5f * fabsf(H_8dof[1] + H_8dof[3]);
  const gboolean low_shear_8 = shear_8 <= HDR_ALIGN_ESCALATION_MAX_SHEAR;

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] DOF escalation: 8-DOF ρ=%.4f (Δρ=%+.4f vs 6-DOF, sane=%d, shear=%.4f)",
           rho_8dof, improvement_8, sane_8, shear_8);

  const gboolean accept_8 = sane_8
                             && low_shear_8
                             && rho_8dof > rho_6dof
                             && improvement_8 >= HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT_8DOF;

  // --- Accept best result ---
  if(accept_8)
  {
    memcpy(H, H_8dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_HDRMERGE, "[hdr_merge] DOF escalation: accepted 8-DOF");
    return rho_8dof;
  }
  else
  {
    // 8-DOF did not improve — keep 6-DOF result.
    memcpy(H, H_6dof, sizeof(float) * HDR_ALIGN_H_NPARAM);
    dt_print(DT_DEBUG_HDRMERGE,
             "[hdr_merge] DOF escalation: 8-DOF no improvement over 6-DOF, keeping 6-DOF");
    return rho_6dof;
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

#ifdef HAVE_OPENCL
// Forward declarations: defined further below, after the CFA warp helpers.
static gboolean _ecc_refine_level_cl(const int devid,
                                      const dt_hdr_alignment_cl_global_t *g,
                                      const float *ref_grad,
                                      const float *img_grad,
                                      const int w,
                                      const int h,
                                      float H[HDR_ALIGN_H_NPARAM]);
#endif

gboolean dt_hdr_align_compute(const float *ref_mosaic,
                              const float *img_mosaic,
                              const int wd,
                              const int ht,
                              const float ref_black_level,
                              const float img_black_level,
                              const float relative_exposure,
                              const float relative_iso,
                              dt_hdr_alignment_t *out_align)
{
#ifdef HAVE_OPENCL
  const dt_hdr_alignment_cl_global_t *const g_cl = darktable.opencl->hdr_alignment;
  const gboolean want_cl = (g_cl != NULL) && dt_opencl_running();
  const int devid = want_cl ? dt_opencl_lock_device(0) : DT_DEVICE_CPU;
  const gboolean use_cl = (devid > DT_DEVICE_CPU);
  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] alignment: using %s path", use_cl ? "OpenCL" : "CPU/OpenMP");
#else
  dt_print(DT_DEBUG_HDRMERGE, "[hdr_merge] alignment: using CPU/OpenMP path");
#endif
  out_align->H[0] = 1.0f; out_align->H[1] = 0.0f; out_align->H[2] = 0.0f;
  out_align->H[3] = 0.0f; out_align->H[4] = 1.0f; out_align->H[5] = 0.0f;
  out_align->H[6] = 0.0f; out_align->H[7] = 0.0f;
  _zero_mesh(out_align->mesh_dx, out_align->mesh_dy);

  if(wd < HDR_ALIGN_MIN_DIM || ht < HDR_ALIGN_MIN_DIM)
  {
#ifdef HAVE_OPENCL
    dt_opencl_unlock_device(devid);
#endif
    return FALSE;
  }

  // Step 1: Normalise Bayer mosaics.
  //   Subtract the per-image black level and scale the candidate image by
  //   1 / (relative_exposure × relative_iso) so that both images are on the
  //   same effective photon-count scale.
  const float img_inv_exposure =
    (relative_exposure > 1e-12f && relative_iso > 1e-12f)
      ? 1.0f / (relative_exposure * relative_iso)
      : 1.0f;

  float *bayer_ref = _normalize_bayer(ref_mosaic, wd, ht, ref_black_level, 1.0f);
  float *bayer_img = _normalize_bayer(img_mosaic, wd, ht, img_black_level, img_inv_exposure);
  if(!bayer_ref || !bayer_img)
  {
    dt_free_align(bayer_ref);
    dt_free_align(bayer_img);
#ifdef HAVE_OPENCL
    dt_opencl_unlock_device(devid);
#endif
    return FALSE;
  }

  // Step 2: Build the image pyramid.
  //
  // FULL_CFA:     pyramid built directly from full-resolution Bayer data.
  //               L0 retains CFA; L1+ are grayscale via 2× box downsample.
  // AVG_BAYER:    L0 = half-resolution grayscale from 2×2 block average.
  // GREEN_ONLY:   L0 = half-resolution grayscale from green channel average.
  // In AVG_BAYER and GREEN_ONLY the final H is scaled from half-res to full-res.
  _pyramid_t pyr_ref, pyr_img;
  int pyr_w, pyr_h; // dimensions of pyramid level 0

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
  pyr_w = wd;
  pyr_h = ht;
  if(!_build_pyramid(bayer_ref, pyr_w, pyr_h, &pyr_ref))
  {
    dt_free_align(bayer_ref);
    dt_free_align(bayer_img);
#ifdef HAVE_OPENCL
    dt_opencl_unlock_device(devid);
#endif
    return FALSE;
  }
  if(!_build_pyramid(bayer_img, pyr_w, pyr_h, &pyr_img))
  {
    _free_pyramid(&pyr_ref);
    dt_free_align(bayer_ref);
    dt_free_align(bayer_img);
#ifdef HAVE_OPENCL
    dt_opencl_unlock_device(devid);
#endif
    return FALSE;
  }
  dt_free_align(bayer_ref);
  dt_free_align(bayer_img);
#else /* AVG_BAYER or GREEN_ONLY: convert to half-res grayscale first */
  {
#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_GREEN_ONLY
    float *gray_ref = _mosaic_to_green_only(bayer_ref, wd, ht, &pyr_w, &pyr_h);
    float *gray_img = _mosaic_to_green_only(bayer_img, wd, ht, &pyr_w, &pyr_h);
#else /* HDR_ALIGN_L0_AVG_BAYER */
    float *gray_ref = _mosaic_to_grayscale(bayer_ref, wd, ht, &pyr_w, &pyr_h);
    float *gray_img = _mosaic_to_grayscale(bayer_img, wd, ht, &pyr_w, &pyr_h);
#endif
    dt_free_align(bayer_ref);
    dt_free_align(bayer_img);
    if(!gray_ref || !gray_img)
    {
      dt_free_align(gray_ref);
      dt_free_align(gray_img);
#ifdef HAVE_OPENCL
      dt_opencl_unlock_device(devid);
#endif
      return FALSE;
    }
    if(!_build_pyramid(gray_ref, pyr_w, pyr_h, &pyr_ref))
    {
      dt_free_align(gray_ref);
      dt_free_align(gray_img);
#ifdef HAVE_OPENCL
      dt_opencl_unlock_device(devid);
#endif
      return FALSE;
    }
    if(!_build_pyramid(gray_img, pyr_w, pyr_h, &pyr_img))
    {
      _free_pyramid(&pyr_ref);
      dt_free_align(gray_ref);
      dt_free_align(gray_img);
#ifdef HAVE_OPENCL
      dt_opencl_unlock_device(devid);
#endif
      return FALSE;
    }
    dt_free_align(gray_ref);
    dt_free_align(gray_img);
  }
#endif /* HDR_ALIGN_L0_MODE */

  // Track whether the finest-level alignment concluded that no correction is
  // needed.  When identity is selected we skip the residual mesh computation:
  // if the images are already well-aligned, mesh estimation would only inject
  // noise from the NCC patch search.
  gboolean h_is_identity = FALSE;

  // Step 3: Coarse exhaustive search at the deepest (smallest) pyramid level.
  //         NCC is still appropriate here because the images are tiny
  //         (~64×48) and the ECC algorithm needs a reasonable starting
  //         point to converge.
  const int coarsest = pyr_ref.nlevels - 1;
  const int cw = pyr_ref.width[coarsest];
  const int ch = pyr_ref.height[coarsest];

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] pyramid: %d levels, coarsest %dx%d, finest %dx%d",
           pyr_ref.nlevels, cw, ch, pyr_ref.width[0], pyr_ref.height[0]);

  float best_tx = 0.0f, best_ty = 0.0f, best_angle = 0.0f, best_scale = 1.0f;
  float best_ncc = -2.0f;

  const int search_radius = (int)(MAX(cw, ch) * HDR_ALIGN_COARSE_SEARCH_FRAC);
  const float angle_step_rad = HDR_ALIGN_COARSE_ANGLE_STEP * (float)M_PI / 180.0f;
  const float max_angle_rad = HDR_ALIGN_MAX_ANGLE_DEG * (float)M_PI / 180.0f;

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] coarse search: %dx%d, radius=%d, angle_step=%.1f°",
           cw, ch, search_radius, HDR_ALIGN_COARSE_ANGLE_STEP);

  // Compute signed Sobel gradient images for exposure-invariant coarse search.
  // Raw-pixel NCC can fail badly when the exposure difference between
  // brackets is very large (clipped highlights, noisy shadows).  Signed
  // gradients preserve structure and direction while removing the intensity
  // DC component so that edges drive the match.
  //
  // Pipeline (restructured):
  //   Gaussian blur → Sobel gx,gy → magnitude → percentile+power+threshold
  //   → mask (magnitude + intensity) → gx+gy + MAD norm + apply mask.
  // This is the same preprocessing that the ECC refinement uses.
  const size_t cpix = (size_t)cw * ch;
  float *coarse_ref_grad = NULL;
  float *coarse_img_grad = NULL;
  // Also keep the pre-gradient intensity images alive for gradient-magnitude
  // ρ scoring after the NCC search (see below).
  float *coarse_ref_norm = NULL;
  float *coarse_img_norm = NULL;
  {
    float *tmp_ref = dt_alloc_align_float(cpix);
    float *tmp_img = dt_alloc_align_float(cpix);
    if(tmp_ref && tmp_img)
    {
      memcpy(tmp_ref, pyr_ref.data[coarsest], sizeof(float) * cpix);
      memcpy(tmp_img, pyr_img.data[coarsest], sizeof(float) * cpix);

      // Use the restructured gradient pipeline.
      coarse_ref_grad = _compute_level_gradient(tmp_ref, cw, ch,
                                                HDR_ALIGN_PREFILTER_SIGMA, NULL);
      // tmp_ref was modified in-place by the blur; keep a pre-Sobel copy
      // for ρ scoring.  We re-copy from the pyramid since tmp_ref is now blurred.
      memcpy(tmp_ref, pyr_ref.data[coarsest], sizeof(float) * cpix);
      _normalize_image_percentile(tmp_ref, cpix);

      memcpy(tmp_img, pyr_img.data[coarsest], sizeof(float) * cpix);
      coarse_img_grad = _compute_level_gradient(tmp_img, cw, ch,
                                                HDR_ALIGN_PREFILTER_SIGMA, NULL);
      memcpy(tmp_img, pyr_img.data[coarsest], sizeof(float) * cpix);
      _normalize_image_percentile(tmp_img, cpix);

      // Keep the intensity images for gradient-magnitude ρ scoring below.
      if(coarse_ref_grad && coarse_img_grad)
      {
        coarse_ref_norm = tmp_ref;
        coarse_img_norm = tmp_img;
      }
      else
      {
        dt_free_align(tmp_ref);
        dt_free_align(tmp_img);
      }
    }
    else
    {
      dt_free_align(tmp_ref);
      dt_free_align(tmp_img);
    }
    _debug_export_gradient_pfm(coarse_ref_grad, cw, ch, "coarse_ref");
    _debug_export_gradient_pfm(coarse_img_grad, cw, ch, "coarse_img");
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

  for(float scale = HDR_ALIGN_COARSE_SCALE_MIN; scale <= HDR_ALIGN_COARSE_SCALE_MAX + 1e-4f;
      scale += HDR_ALIGN_COARSE_SCALE_STEP)
  {
  for(float angle = -max_angle_rad; angle <= max_angle_rad;
      angle += angle_step_rad)
  {
    // Warp coarsest-level input by candidate scale×rotation (no translation)
    float *similarity = NULL;
    const float *candidate = coarse_img;

    const gboolean need_warp = (fabsf(angle) > 1e-6f
                                 || fabsf(scale - 1.0f) > 1e-4f);
    if(need_warp)
    {
      similarity = _warp_similarity_nocrop(coarse_img, cw, ch, angle, scale);
      if(!similarity) continue;
      candidate = similarity;
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
          best_scale = scale;
        }
      }

    dt_free_align(similarity);
  }
  } // end scale loop

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] coarse result: tx=%.0f ty=%.0f angle=%.2f° ncc=%.4f"
           " (identity ncc=%.4f)",
           best_tx, best_ty, best_angle * 180.0f / (float)M_PI, best_ncc,
           ncc_identity);

  // Early-out using ECC ρ: compute the weighted gradient-magnitude
  // correlation coefficient at the coarsest level for both the identity
  // transform and the NCC-winning candidate.  If identity ρ >= candidate ρ,
  // the images are already well-aligned and the full ECC pyramid would risk
  // drift for no gain.
  //
  // We use the intensity images (coarse_ref_norm / coarse_img_norm) so that
  // _ecc_compute_rho can form proper gradient *magnitude* PCC, which is
  // robust at all scales unlike the signed-gradient PCC used during
  // optimisation.
  if(coarse_ref_norm && coarse_img_norm)
  {
    const float H_id_coarse[HDR_ALIGN_H_NPARAM]
      = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
    float H_candidate[HDR_ALIGN_H_NPARAM];
    _homography_from_similarity(best_tx, best_ty, best_angle, best_scale,
                                cw, ch, H_candidate);

    const float rho_id_coarse = _ecc_compute_rho(coarse_ref_norm, coarse_img_norm,
                                                   cw, ch, H_id_coarse);
    const float rho_candidate = _ecc_compute_rho(coarse_ref_norm, coarse_img_norm,
                                                   cw, ch, H_candidate);

    dt_print(DT_DEBUG_HDRMERGE,
             "[hdr_merge] coarse ECC ρ: identity=%.5f candidate=%.5f",
             rho_id_coarse, rho_candidate);

    if(rho_id_coarse > -1.0f && rho_id_coarse >= rho_candidate)
    {
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] identity ρ >= candidate ρ at coarsest level"
               " -- using identity as ECC starting point");
      // Don't skip: run the full ECC pyramid and DOF escalation starting from
      // identity.  There may be residual misalignment visible at finer pyramid
      // levels that the coarsest-level comparison cannot detect.  Reset the
      // Euclidean estimate to identity so H_level is initialised correctly.
      best_tx = 0.0f;
      best_ty = 0.0f;
      best_angle = 0.0f;
      best_scale = 1.0f;
    }

    dt_free_align(coarse_ref_norm);
    dt_free_align(coarse_img_norm);
  }

  dt_free_align(coarse_ref_grad);
  dt_free_align(coarse_img_grad);

  // Initialise projective model from coarse similarity estimate.
  float H_level[HDR_ALIGN_H_NPARAM];
  _homography_from_similarity(best_tx, best_ty, best_angle, best_scale, cw, ch, H_level);

  // Note: the 3-DOF rigid ECC at intermediate pyramid levels will overwrite the
  // scale component of H (it only optimises rotation + translation).  Once DOF
  // escalation succeeds at the escalation level, the 6-/8-DOF solver will
  // recover and refine scale for all remaining finer levels.

  // Step 4: Multi-resolution ECC refinement.
  //
  // Starting from the coarsest level, run iterative ECC at each level
  // to refine the (tx, ty, angle) parameters to sub-pixel accuracy.
  // The ECC (Evangelidis & Psarakis 2008) maximises the Enhanced
  // Correlation Coefficient, which is invariant to affine photometric
  // transforms — exactly what we need for HDR brackets with different
  // exposures.  Parameters are scaled by 2× when transitioning to
  // each finer level.

  // Determine the coarsest level that meets the ECC_MIN_DIM threshold.
  // This is used by the adaptive drift guards to scale tolerances: the
  // coarsest ECC level gets the base tolerance, each finer step halves it.
  int first_ecc_level = 0;
  for(int l = coarsest; l >= 0; l--)
  {
    if(MIN(pyr_ref.width[l], pyr_ref.height[l]) >= HDR_ALIGN_ECC_MIN_DIM)
    {
      first_ecc_level = l;
      break;
    }
  }

  // Determine the pyramid level at which DOF escalation will be attempted.
  // This is the coarsest level whose shortest edge is at least
  // HDR_ALIGN_ESCALATION_MIN_DIM.  If no level qualifies (images too small),
  // escalation falls back to level 0 (the old behaviour).
  int escalation_level = 0;
  for(int l = coarsest; l >= 0; l--)
  {
    if(MIN(pyr_ref.width[l], pyr_ref.height[l]) >= HDR_ALIGN_ESCALATION_MIN_DIM)
    {
      escalation_level = l;
      break;
    }
  }
  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] DOF escalation target: level %d (%dx%d)",
           escalation_level,
           pyr_ref.width[escalation_level],
           pyr_ref.height[escalation_level]);

  // Track the currently accepted DOF model.  Starts at 3 (Euclidean);
  // once escalation succeeds this changes to 6 (affine) or 8 (projective)
  // and all subsequent finer levels use the higher-DOF solver.
  int current_dof = 3;

  for(int l = coarsest; l >= 0; l--)
  {
    const int lw = pyr_ref.width[l];
    const int lh = pyr_ref.height[l];

    // Scale homography from parent level to current (finer) level.
    if(l < coarsest)
      _homography_scale_to_finer(H_level);

    // Skip ECC on levels where the shortest edge is below the minimum useful
    // resolution.  At those scales the gradient image carries too little spatial
    // information and the Hessian is ill-conditioned; the coarse NCC estimate
    // already covers the large-displacement component.
    if(MIN(lw, lh) < HDR_ALIGN_ECC_MIN_DIM)
    {
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] ECC level %d (%dx%d): skipped (too small)",
               l, lw, lh);
      continue;
    }

    // Compute gradient images for exposure-invariant ECC.
    //
    // Restructured pipeline (applied identically regardless of L0 mode):
    //   Gaussian blur → Sobel gx,gy → magnitude → percentile+power+threshold
    //   → mask (magnitude + intensity) → gx+gy + MAD norm + apply mask
    //
    // FULL_CFA at L0: CFA-aware variant (per-sublattice norm + stride-2 Sobel).
    // AVG_BAYER / GREEN_ONLY at L0 and all modes at L1+: standard stride-1 Sobel.
    const size_t lpix = (size_t)lw * lh;
    float *ref_norm = dt_alloc_align_float(lpix);
    float *img_norm = dt_alloc_align_float(lpix);
    if(!ref_norm || !img_norm)
    {
      dt_free_align(ref_norm);
      dt_free_align(img_norm);
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] ECC level %d: norm alloc failed", l);
      continue;
    }
    memcpy(ref_norm, pyr_ref.data[l], sizeof(float) * lpix);
    memcpy(img_norm, pyr_img.data[l], sizeof(float) * lpix);

    float *ref_grad = NULL;
    float *img_grad = NULL;

#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
    if(l == 0)
    {
      // L0: full-resolution Bayer — CFA-aware gradient pipeline.
      ref_grad = _compute_level_gradient_cfa(ref_norm, lw, lh,
                                             HDR_ALIGN_PREFILTER_SIGMA, NULL);
      img_grad = _compute_level_gradient_cfa(img_norm, lw, lh,
                                             HDR_ALIGN_PREFILTER_SIGMA, NULL);
      // Restore ref_norm / img_norm for ρ scoring in DOF escalation.
      memcpy(ref_norm, pyr_ref.data[l], sizeof(float) * lpix);
      memcpy(img_norm, pyr_img.data[l], sizeof(float) * lpix);
      _normalize_bayer_per_channel(ref_norm, lw, lh);
      _normalize_bayer_per_channel(img_norm, lw, lh);
    }
    else
#endif /* HDR_ALIGN_L0_FULL_CFA */
    {
      // L1+ (or L0 in AVG_BAYER / GREEN_ONLY): standard gradient pipeline.
      ref_grad = _compute_level_gradient(ref_norm, lw, lh,
                                         HDR_ALIGN_PREFILTER_SIGMA, NULL);
      img_grad = _compute_level_gradient(img_norm, lw, lh,
                                         HDR_ALIGN_PREFILTER_SIGMA, NULL);
      // Keep ref_norm/img_norm alive from escalation_level down to L0 for ρ
      // scoring (DOF escalation at escalation_level, identity check at L0).
      if(l <= escalation_level)
      {
        // Restore for ρ scoring (the blur modified the data).
        memcpy(ref_norm, pyr_ref.data[l], sizeof(float) * lpix);
        memcpy(img_norm, pyr_img.data[l], sizeof(float) * lpix);
        _normalize_image_percentile(ref_norm, lpix);
        _normalize_image_percentile(img_norm, lpix);
      }
      else
      {
        dt_free_align(ref_norm);
        dt_free_align(img_norm);
        ref_norm = img_norm = NULL;
      }
    }

    if(!ref_grad || !img_grad)
    {
      dt_free_align(ref_grad);
      dt_free_align(img_grad);
      dt_free_align(ref_norm);
      dt_free_align(img_norm);
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] ECC level %d: gradient alloc failed", l);
      continue;
    }

    // Export gradient images for debugging when DT_DEBUG_VERBOSE is set.
    {
      char label_ref[32], label_img[32];
      snprintf(label_ref, sizeof(label_ref), "ref_L%d", l);
      snprintf(label_img, sizeof(label_img), "img_L%d", l);
      _debug_export_gradient_pfm(ref_grad, lw, lh, label_ref);
      _debug_export_gradient_pfm(img_grad, lw, lh, label_img);
    }


    dt_print(DT_DEBUG_HDRMERGE,
             "[hdr_merge] ECC level %d (%dx%d, %d-DOF): initial H=[%.4f %.4f %.2f; %.4f %.4f %.2f; %.6f %.6f 1]",
             l, lw, lh, current_dof,
             H_level[0], H_level[1], H_level[2],
             H_level[3], H_level[4], H_level[5],
             H_level[6], H_level[7]);

    // Save pre-ECC homography so we can revert if ECC diverges.
    float H_backup[HDR_ALIGN_H_NPARAM];
    memcpy(H_backup, H_level, sizeof(H_backup));

    if(current_dof > 3)
    {
      // Higher-DOF solver (6 or 8 DOF) — runs on CPU only.
      // After DOF escalation succeeded at an earlier level, all subsequent
      // finer levels use the higher-DOF solver so the extra parameters
      // (scale, shear, perspective) are continuously refined.
      const float rho = _ecc_refine_level_higher_dof(ref_grad, img_grad,
                                                      ref_norm, img_norm,
                                                      lw, lh, H_level,
                                                      current_dof);
      (void)rho; // ρ is logged internally; we rely on sanity checks below.

      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] ECC level %d (%d-DOF): ρ=%.4f", l, current_dof, rho);

      // Sanity check: use the escalated sanity check since we have higher DOF.
      if(!_homography_is_sane_escalated(H_level, lw, lh, current_dof))
      {
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] ECC level %d (%d-DOF): result failed sanity check, reverting",
                 l, current_dof);
        memcpy(H_level, H_backup, sizeof(H_level));
      }
    }
    else
    {
      // 3-DOF Euclidean ECC — use CL path if available, else CPU.
      gboolean ran_cl = FALSE;
#ifdef HAVE_OPENCL
      if(use_cl)
        ran_cl = _ecc_refine_level_cl(devid, g_cl, ref_grad, img_grad, lw, lh, H_level);
#endif
      float ecc_update;
      if(ran_cl)
      {
        // GPU path ran; compute the convergence metric for the guard checks.
        // We don't propagate the per-iteration update from the CL path out of
        // _ecc_refine_level_cl, so treat a successful CL run as converged for
        // the drift guards (they compare H_level vs H_backup regardless).
        ecc_update = 0.0f;
      }
      else
      {
        ecc_update = _ecc_refine_level(ref_grad, img_grad, lw, lh, H_level);
      }
      // ecc_update < HDR_ALIGN_ECC_EPSILON means the optimiser genuinely
      // converged; stall or max-iteration exits return a larger (or zero) value.
      const gboolean ecc_converged = (ecc_update >= 0.0f && ecc_update < HDR_ALIGN_ECC_EPSILON);

      // Sanity check: revert if ECC produced a physically implausible
      // homography (e.g. large scale change or perspective).
      if(!_homography_is_sane(H_level, lw, lh))
      {
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] ECC level %d: result failed sanity check, reverting",
                 l);
        memcpy(H_level, H_backup, sizeof(H_level));
      }

      // Angle drift guard: if ECC changed the rotation by more than the
      // per-level tolerance it wandered to a wrong local maximum.  Revert to
      // the pre-level homography so the error does not cascade to finer levels.
      //
      // The tolerance is adaptive: the base value (HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_BASE)
      // applies at the coarsest ECC level.  For each step toward L0 the
      // tolerance is halved, because the parent estimate is already refined
      // and finer levels should only introduce sub-pixel corrections.
      // The minimum is clamped to HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_MIN.
      //
      // Skip the guard when ECC genuinely converged — a converged rotation is
      // trustworthy even if it moved more than the default tolerance.
      if(!ecc_converged)
      {
        float delta = atan2f(H_level[3], H_level[0])
                    - atan2f(H_backup[3], H_backup[0]);
        if(delta >  (float)M_PI) delta -= 2.0f * (float)M_PI;
        if(delta < -(float)M_PI) delta += 2.0f * (float)M_PI;
        // steps_from_coarsest == 0 at the coarsest ECC level, increases toward L0.
        // Clamped to HDR_ALIGN_MAX_PYRAMID_LEVELS to prevent bit-shift overflow.
        const int steps_from_coarsest = MIN(first_ecc_level - l, HDR_ALIGN_MAX_PYRAMID_LEVELS);
        const float angle_limit_deg = fmaxf(
          HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_BASE / (float)(1 << steps_from_coarsest),
          HDR_ALIGN_ECC_MAX_ANGLE_DELTA_DEG_MIN);
        const float max_delta = angle_limit_deg * (float)M_PI / 180.0f;
        if(fabsf(delta) > max_delta)
        {
          dt_print(DT_DEBUG_HDRMERGE,
                   "[hdr_merge] ECC level %d: angle drift %.2f° > limit %.1f°, reverting",
                   l, delta * 180.0f / (float)M_PI, angle_limit_deg);
          memcpy(H_level, H_backup, sizeof(H_level));
        }
      }

      // Translation drift guard: if ECC shifted the translation by more than
      // the per-level tolerance, the optimiser has wandered.  Each level
      // inherits a 2×-scaled estimate from its parent, so the expected
      // correction is at most a few pixels.  Large shifts at any single level
      // indicate a wrong local minimum, and the error doubles at every finer
      // level, producing catastrophically wrong final alignments.
      //
      // The tolerance is adaptive (same halving schedule as the angle guard).
      // Skip the guard when ECC genuinely converged — the result is trusted.
      if(!ecc_converged)
      {
        const float dtx = H_level[2] - H_backup[2];
        const float dty = H_level[5] - H_backup[5];
        const int steps_from_coarsest = MIN(first_ecc_level - l, HDR_ALIGN_MAX_PYRAMID_LEVELS);
        const float trans_limit = fmaxf(
          HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX_BASE / (float)(1 << steps_from_coarsest),
          HDR_ALIGN_ECC_MAX_TRANS_DELTA_PX_MIN);
        if(fabsf(dtx) > trans_limit
           || fabsf(dty) > trans_limit)
        {
          dt_print(DT_DEBUG_HDRMERGE,
                   "[hdr_merge] ECC level %d: translation drift (%.2f, %.2f) px"
                   " > limit %.1f px, reverting",
                   l, dtx, dty, trans_limit);
          memcpy(H_level, H_backup, sizeof(H_level));
        }
      }
    } // end 3-DOF branch

    // --- Adaptive DOF escalation ---
    // At the designated escalation level, attempt 6-DOF affine and then
    // 8-DOF projective refinement.  If accepted, all subsequent finer
    // levels will use the higher-DOF solver.
    if(l == escalation_level && current_dof == 3 && ref_norm && img_norm)
    {
      // Compute identity ρ first so we can gate DOF escalation correctly.
      // If the 3-DOF result is already worse than identity, the ECC pyramid
      // wandered to a wrong local minimum.  Higher-DOF models starting from
      // that degraded point will only drift further — skip escalation entirely.
      const float H_id[HDR_ALIGN_H_NPARAM]
        = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
      const float rho_id = _ecc_compute_rho(ref_norm, img_norm, lw, lh, H_id);
      const float rho_3dof = _ecc_compute_rho(ref_norm, img_norm, lw, lh, H_level);

      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] DOF escalation at level %d: identity ρ=%.4f",
               l, rho_id);
      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] DOF escalation at level %d: 3-DOF ρ=%.4f (Δρ=%+.4f vs identity)",
               l, rho_3dof, rho_3dof - rho_id);

      if(rho_3dof <= rho_id)
      {
        // The 3-DOF result is worse than identity.  This typically means
        // the coarse NCC search found a false match (wrong scale/rotation
        // basin) and every ECC level since then has been stuck reverting
        // to the same bad H.
        //
        // Recovery: reset H to identity and re-run 3-DOF ECC from scratch
        // at this level.  If ECC from identity finds a useful alignment
        // (ρ > identity ρ), attempt DOF escalation on that result.
        // Either way, finer levels inherit a clean starting point instead
        // of the bad coarse H.
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] DOF escalation: 3-DOF below identity"
                 " -- resetting to identity and re-running ECC at level %d", l);

        memcpy(H_level, H_id, sizeof(float) * HDR_ALIGN_H_NPARAM);

        // Re-run 3-DOF ECC from identity at this level.
        {
          gboolean ran_cl_recovery = FALSE;
#ifdef HAVE_OPENCL
          if(use_cl)
            ran_cl_recovery = _ecc_refine_level_cl(devid, g_cl, ref_grad, img_grad,
                                                    lw, lh, H_level);
#endif
          if(!ran_cl_recovery)
            _ecc_refine_level(ref_grad, img_grad, lw, lh, H_level);

          // Sanity-check the recovery result.
          if(!_homography_is_sane(H_level, lw, lh))
          {
            dt_print(DT_DEBUG_HDRMERGE,
                     "[hdr_merge] DOF recovery: ECC from identity failed sanity"
                     " -- keeping identity");
            memcpy(H_level, H_id, sizeof(float) * HDR_ALIGN_H_NPARAM);
          }
        }

        const float rho_recovery = _ecc_compute_rho(ref_norm, img_norm,
                                                     lw, lh, H_level);
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] DOF recovery at level %d: ρ=%.4f (identity ρ=%.4f)",
                 l, rho_recovery, rho_id);

        if(rho_recovery > rho_id)
        {
          // Recovery succeeded: ECC from identity found a better alignment.
          // Try DOF escalation on this recovered result.
          const float rho_before = rho_recovery;
          const float rho_best = _try_dof_escalation(ref_grad, img_grad,
                                                      ref_norm, img_norm,
                                                      lw, lh, rho_recovery,
                                                      H_level);
          if(rho_best > rho_before)
          {
            if(fabsf(H_level[6]) > 1e-9f || fabsf(H_level[7]) > 1e-9f)
              current_dof = 8;
            else
              current_dof = 6;

            dt_print(DT_DEBUG_HDRMERGE,
                     "[hdr_merge] DOF recovery+escalation: accepted %d-DOF"
                     " (ρ=%.4f→%.4f), using %d-DOF for remaining levels",
                     current_dof, rho_before, rho_best, current_dof);
          }
          else
          {
            dt_print(DT_DEBUG_HDRMERGE,
                     "[hdr_merge] DOF recovery: keeping 3-DOF from identity");
          }
        }
        // else: H_level is identity; finer levels will start from a clean
        // baseline rather than the wrong coarse estimate.
      }
      else
      {
        // 3-DOF improved over identity: attempt higher-DOF refinement.
        const float rho_before = rho_3dof;
        const float rho_best = _try_dof_escalation(ref_grad, img_grad,
                                                    ref_norm, img_norm,
                                                    lw, lh, rho_3dof, H_level);

        // Determine which DOF was accepted by checking H_level parameters:
        // if perspective terms (H[6],H[7]) are nonzero → 8-DOF;
        // if off-diagonal/scale terms differ from rigid → 6-DOF.
        if(rho_best > rho_before)
        {
          // Check if perspective terms are set (8-DOF was accepted).
          // The 6-DOF affine model always keeps H[6]=H[7]=0; only the
          // 8-DOF projective model sets them to nonzero values.
          if(fabsf(H_level[6]) > 1e-9f || fabsf(H_level[7]) > 1e-9f)
            current_dof = 8;
          else
            current_dof = 6;

          dt_print(DT_DEBUG_HDRMERGE,
                   "[hdr_merge] DOF escalation: accepted %d-DOF (ρ=%.4f→%.4f), "
                   "using %d-DOF for remaining levels",
                   current_dof, rho_before, rho_best, current_dof);
        }
        else
        {
          dt_print(DT_DEBUG_HDRMERGE,
                   "[hdr_merge] DOF escalation: no improvement, keeping 3-DOF");
        }
      }
    }

    // --- Final identity check at L0 ---
    // Regardless of which DOF model was used, verify at the finest level
    // that the result is better than identity.  If not, revert.
    if(l == 0 && ref_norm && img_norm)
    {
      const float H_id[HDR_ALIGN_H_NPARAM]
        = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
      const float rho_id = _ecc_compute_rho(ref_norm, img_norm, lw, lh, H_id);
      const float rho_cur = _ecc_compute_rho(ref_norm, img_norm, lw, lh, H_level);

      dt_print(DT_DEBUG_HDRMERGE,
               "[hdr_merge] L0 identity check: current ρ=%.4f, identity ρ=%.4f",
               rho_cur, rho_id);

      if(rho_id >= rho_cur)
      {
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] L0: current ρ=%.5f ≤ identity ρ=%.5f -- reverting to identity",
                 rho_cur, rho_id);
        memcpy(H_level, H_id, sizeof(float) * HDR_ALIGN_H_NPARAM);
        h_is_identity = TRUE;
      }
    }

    // Free intensity images when done with them.
    if(ref_norm)
    {
      dt_free_align(ref_norm);
      ref_norm = NULL;
    }
    if(img_norm)
    {
      dt_free_align(img_norm);
      img_norm = NULL;
    }

    if(l <= 1)
      _corner_refine_level(ref_grad, img_grad, lw, lh, H_level);

    dt_free_align(ref_grad);
    dt_free_align(img_grad);

    dt_print(DT_DEBUG_HDRMERGE,
            "[hdr_merge] ECC level %d result: H=[%.4f %.4f %.2f; %.4f %.4f %.2f; %.6f %.6f 1]",
            l,
            H_level[0], H_level[1], H_level[2],
            H_level[3], H_level[4], H_level[5],
            H_level[6], H_level[7]);
  }

  // Output the homography in full-resolution Bayer pixel coordinates.
#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
  // L0 is already full-resolution — H_level maps directly.
  memcpy(out_align->H, H_level, sizeof(float) * HDR_ALIGN_H_NPARAM);
#else
  // L0 is half-resolution (AVG_BAYER or GREEN_ONLY) — scale H back to full-resolution.
  _homography_local_to_full(H_level, out_align->H);
#endif

  // Only compute the residual mesh when the alignment is not identity.
  // If identity was selected at the finest level the images are already
  // well-aligned and mesh estimation would only add noise.
  if(!h_is_identity)
  {
    float *ref_grad0 = NULL;
    float *img_grad0 = NULL;
    {
      const int l0w = pyr_ref.width[0];
      const int l0h = pyr_ref.height[0];
      const size_t npix0 = (size_t)l0w * l0h;
      float *rn = dt_alloc_align_float(npix0);
      float *in = dt_alloc_align_float(npix0);
      if(rn && in)
      {
        memcpy(rn, pyr_ref.data[0], sizeof(float) * npix0);
        memcpy(in, pyr_img.data[0], sizeof(float) * npix0);
#if HDR_ALIGN_L0_MODE == HDR_ALIGN_L0_FULL_CFA
        // L0 mesh: CFA-aware gradient pipeline (same as ECC level 0).
        ref_grad0 = _compute_level_gradient_cfa(rn, l0w, l0h,
                                                HDR_ALIGN_PREFILTER_SIGMA, NULL);
        img_grad0 = _compute_level_gradient_cfa(in, l0w, l0h,
                                                HDR_ALIGN_PREFILTER_SIGMA, NULL);
#else
        // L0 mesh: standard gradient pipeline (AVG_BAYER or GREEN_ONLY).
        ref_grad0 = _compute_level_gradient(rn, l0w, l0h,
                                            HDR_ALIGN_PREFILTER_SIGMA, NULL);
        img_grad0 = _compute_level_gradient(in, l0w, l0h,
                                            HDR_ALIGN_PREFILTER_SIGMA, NULL);
#endif
      }
      dt_free_align(rn);
      dt_free_align(in);
    }
    if(ref_grad0 && img_grad0)
    {
      // Mesh operates at pyramid L0 resolution.  Compute the mesh H in L0
      // coordinates and convert displacements to full-resolution afterwards.
      float H_mesh[HDR_ALIGN_H_NPARAM];
      memcpy(H_mesh, H_level, sizeof(float) * HDR_ALIGN_H_NPARAM);
      if(_estimate_mesh_residuals(ref_grad0, img_grad0,
                                  pyr_ref.width[0], pyr_ref.height[0],
                                  H_mesh, out_align->mesh_dx, out_align->mesh_dy))
      {
#if HDR_ALIGN_L0_MODE != HDR_ALIGN_L0_FULL_CFA
        // Scale mesh displacements from half-resolution to full-resolution.
        for(int i = 0; i < DT_HDR_ALIGN_MESH_NODES; i++)
        {
          out_align->mesh_dx[i] *= 2.0f;
          out_align->mesh_dy[i] *= 2.0f;
        }
#endif
        // mesh_dx / mesh_dy filled by _estimate_mesh_residuals on success;
        // on failure the arrays remain zeroed from _zero_mesh() earlier.
      }
    }
    dt_free_align(ref_grad0);
    dt_free_align(img_grad0);
  }

  dt_print(DT_DEBUG_HDRMERGE,
           "[hdr_merge] final homography: H=[%.5f %.5f %.2f; %.5f %.5f %.2f; %.7f %.7f 1]",
           out_align->H[0], out_align->H[1], out_align->H[2],
           out_align->H[3], out_align->H[4], out_align->H[5],
           out_align->H[6], out_align->H[7]);

  // Named decomposition of the homography elements.
  // H = [ h11  h12  tx  ]   h11≈cos(θ)·sx  h12≈sin(θ)·sy  tx = translation x
  //     [ h21  h22  ty  ]   h21≈-sin(θ)·sx h22≈cos(θ)·sy  ty = translation y
  //     [ p1   p2    1  ]   p1,p2 = projective (perspective) terms
  // scale_x = column-1 norm = sqrt(h11²+h21²), scale_y = column-2 norm
  // shear   = normalised dot product of the two columns; 0 for pure rotation
  {
    const float sx = sqrtf(out_align->H[0] * out_align->H[0]
                           + out_align->H[3] * out_align->H[3]);
    const float sy = sqrtf(out_align->H[1] * out_align->H[1]
                           + out_align->H[4] * out_align->H[4]);
    const float shear = (sx > 0.0f && sy > 0.0f)
      ? (out_align->H[0] * out_align->H[1] + out_align->H[3] * out_align->H[4])
        / (sx * sy)
      : 0.0f;
    const float approx_angle = atan2f(out_align->H[1], out_align->H[0]);
    const float mesh_max = _mesh_max_abs(out_align->mesh_dx, out_align->mesh_dy);
    dt_print(DT_DEBUG_HDRMERGE,
             "[hdr_merge] final homography decomposed:"
             " tx=%.1f ty=%.1f rotation=%.3f°"
             " scale_x=%.5f scale_y=%.5f shear=%.5f"
             " h11=%.5f h12=%.5f h21=%.5f h22=%.5f"
             " p1=%.7f p2=%.7f, mesh max=%.2f px, mesh center=(%.2f, %.2f)",
             out_align->H[2], out_align->H[5],
             approx_angle * 180.0f / (float)M_PI,
             sx, sy, shear,
             out_align->H[0], out_align->H[1],
             out_align->H[3], out_align->H[4],
             out_align->H[6], out_align->H[7],
             mesh_max,
             out_align->mesh_dx[HDR_ALIGN_MESH_INDEX(1, 1)],
             out_align->mesh_dy[HDR_ALIGN_MESH_INDEX(1, 1)]);
  }

  _free_pyramid(&pyr_ref);
  _free_pyramid(&pyr_img);
#ifdef HAVE_OPENCL
  dt_opencl_unlock_device(devid);
#endif
  return TRUE;
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
// OpenCL ECC iteration and level refinement
// ---------------------------------------------------------------------------

#ifdef HAVE_OPENCL

#define HDR_ALIGN_CL_WG_W 16
#define HDR_ALIGN_CL_WG_H 16

/** One GPU-accelerated ECC iteration.
 *
 *  cl_ref and cl_img are pre-uploaded, static per level (gradient images).
 *  cl_warped, cl_mask, cl_gx, cl_gy are scratch buffers, size npix floats.
 *  cl_sums_p1 is ngroups*4 floats; cl_sums_p23 is ngroups*9 floats (reused).
 *  host_p1 and host_p23 are host-side staging buffers of matching sizes.
 *
 *  Returns the pixel-equivalent parameter update metric, or -1 on failure.
 *  H is updated in-place on success. */
static float _ecc_iteration_cl(const int devid,
                                const dt_hdr_alignment_cl_global_t *g,
                                cl_mem cl_ref,
                                cl_mem cl_img,
                                cl_mem cl_warped,
                                cl_mem cl_mask,
                                cl_mem cl_gx,
                                cl_mem cl_gy,
                                cl_mem cl_sums_p1,
                                cl_mem cl_sums_p23,
                                float *host_p1,
                                float *host_p23,
                                const int w,
                                const int h,
                                const int ngroups,
                                float H[HDR_ALIGN_H_NPARAM])
{
  const size_t npix = (size_t)w * h;
  const double scale = MAX((double)w - 1.0, (double)h - 1.0) * 0.5;
  const double s = scale > 1.0 ? scale : 1.0;
  const double cx_d = ((double)w - 1.0) * 0.5;
  const double cy_d = ((double)h - 1.0) * 0.5;
  const float cx = (float)cx_d;
  const float cy = (float)cy_d;
  const float inv_s = (float)(1.0 / s);
  const float edge_weight = (float)HDR_ALIGN_ECC_EDGE_WEIGHT;

  const int nwgx = (w + HDR_ALIGN_CL_WG_W - 1) / HDR_ALIGN_CL_WG_W;
  const int nwgy = (h + HDR_ALIGN_CL_WG_H - 1) / HDR_ALIGN_CL_WG_H;
  const size_t gsizes[2] = { (size_t)nwgx * HDR_ALIGN_CL_WG_W,
                              (size_t)nwgy * HDR_ALIGN_CL_WG_H };
  const size_t lsizes[2]  = { HDR_ALIGN_CL_WG_W, HDR_ALIGN_CL_WG_H };
  const size_t lmem_p1 = (size_t)HDR_ALIGN_CL_WG_W * HDR_ALIGN_CL_WG_H * 4 * sizeof(float);
  const size_t lmem_p23 = (size_t)HDR_ALIGN_CL_WG_W * HDR_ALIGN_CL_WG_H * 9 * sizeof(float);

  // --- Warp ---
  cl_int err = dt_opencl_enqueue_kernel_2d_args(
      devid, g->kernel_warp_homography, w, h,
      CLARG(cl_img), CLARG(cl_warped), CLARG(cl_mask),
      CLARG(w), CLARG(h),
      CLARGFLOAT(H[0]), CLARGFLOAT(H[1]), CLARGFLOAT(H[2]),
      CLARGFLOAT(H[3]), CLARGFLOAT(H[4]), CLARGFLOAT(H[5]),
      CLARGFLOAT(H[6]), CLARGFLOAT(H[7]));
  if(err != CL_SUCCESS) return -1.0f;

  // --- Gradients of the warped image ---
  err = dt_opencl_enqueue_kernel_2d_args(
      devid, g->kernel_compute_gradients, w, h,
      CLARG(cl_warped), CLARG(cl_gx), CLARG(cl_gy),
      CLARG(w), CLARG(h));
  if(err != CL_SUCCESS) return -1.0f;

  // --- Pass 1: weighted means ---
  err = dt_opencl_enqueue_kernel_2d_local_args(
      devid, g->kernel_ecc_means, gsizes, lsizes,
      CLARG(cl_ref), CLARG(cl_warped), CLARG(cl_mask), CLARG(cl_sums_p1),
      CLARG(w), CLARG(h),
      CLARGFLOAT(cx), CLARGFLOAT(cy), CLARGFLOAT(inv_s), CLARGFLOAT(edge_weight),
      CLLOCAL(lmem_p1));
  if(err != CL_SUCCESS) return -1.0f;

  err = dt_opencl_read_buffer_from_device(devid, (void *)host_p1, cl_sums_p1, 0,
                                          (size_t)ngroups * 4 * sizeof(float), CL_TRUE);
  if(err != CL_SUCCESS) return -1.0f;

  double sum_r = 0.0, sum_w_acc = 0.0, sum_weight = 0.0;
  long nvalid = 0;
  for(int gi = 0; gi < ngroups; gi++)
  {
    sum_r      += (double)host_p1[gi * 4 + 0];
    sum_w_acc  += (double)host_p1[gi * 4 + 1];
    sum_weight += (double)host_p1[gi * 4 + 2];
    nvalid     += (long)host_p1[gi * 4 + 3];
  }

  if(nvalid < (long)(npix * HDR_ALIGN_ECC_MIN_VALID_FRAC) || sum_weight < 1e-12)
    return -1.0f;

  const double mean_r = sum_r / sum_weight;
  const double mean_w_val = sum_w_acc / sum_weight;

  // Extract current angle from the normalised homography.
  float Hn[HDR_ALIGN_H_NPARAM];
  _homography_pixel_to_normalized(H, w, h, Hn);
  const float cos_t = Hn[0];
  const float sin_t = Hn[1];

  // --- Pass 2: norms, Jacobian sums, sJw projection sums ---
  err = dt_opencl_enqueue_kernel_2d_local_args(
      devid, g->kernel_ecc_norms, gsizes, lsizes,
      CLARG(cl_ref), CLARG(cl_warped), CLARG(cl_mask), CLARG(cl_gx), CLARG(cl_gy),
      CLARG(cl_sums_p23),
      CLARG(w), CLARG(h),
      CLARGFLOAT(cx), CLARGFLOAT(cy), CLARGFLOAT(inv_s), CLARGFLOAT((float)s),
      CLARGFLOAT(edge_weight),
      CLARGFLOAT((float)mean_r), CLARGFLOAT((float)mean_w_val),
      CLARGFLOAT(cos_t), CLARGFLOAT(sin_t),
      CLLOCAL(lmem_p23));
  if(err != CL_SUCCESS) return -1.0f;

  err = dt_opencl_read_buffer_from_device(devid, (void *)host_p23, cl_sums_p23, 0,
                                          (size_t)ngroups * 9 * sizeof(float), CL_TRUE);
  if(err != CL_SUCCESS) return -1.0f;

  double norm2_r = 0.0, norm2_w = 0.0, dot_rw = 0.0;
  double sum_J0 = 0.0, sum_J1 = 0.0, sum_J2 = 0.0;
  double sJw0 = 0.0, sJw1 = 0.0, sJw2 = 0.0;
  for(int gi = 0; gi < ngroups; gi++)
  {
    const float *p = host_p23 + gi * 9;
    norm2_r += (double)p[0];
    norm2_w += (double)p[1];
    dot_rw  += (double)p[2];
    sum_J0  += (double)p[3];
    sum_J1  += (double)p[4];
    sum_J2  += (double)p[5];
    sJw0    += (double)p[6];
    sJw1    += (double)p[7];
    sJw2    += (double)p[8];
  }

  const double norm_r = sqrt(norm2_r);
  const double norm_w = sqrt(norm2_w);
  if(norm_r < 1e-12 || norm_w < 1e-12)
    return -1.0f;

  const double rho = dot_rw / (norm_r * norm_w);
  const double mean_J[3] = { sum_J0 / sum_weight,
                              sum_J1 / sum_weight,
                              sum_J2 / sum_weight };
  const double proj_coeff[3] = { sJw0 / norm2_w,
                                  sJw1 / norm2_w,
                                  sJw2 / norm2_w };
  const double scale_rw = norm_w / norm_r;

  // --- Pass 3: Hessian + RHS ---
  err = dt_opencl_enqueue_kernel_2d_local_args(
      devid, g->kernel_ecc_hessian_final, gsizes, lsizes,
      CLARG(cl_ref), CLARG(cl_warped), CLARG(cl_mask), CLARG(cl_gx), CLARG(cl_gy),
      CLARG(cl_sums_p23),
      CLARG(w), CLARG(h),
      CLARGFLOAT(cx), CLARGFLOAT(cy), CLARGFLOAT(inv_s), CLARGFLOAT((float)s),
      CLARGFLOAT(edge_weight),
      CLARGFLOAT((float)mean_r), CLARGFLOAT((float)mean_w_val),
      CLARGFLOAT(cos_t), CLARGFLOAT(sin_t),
      CLARGFLOAT((float)mean_J[0]), CLARGFLOAT((float)mean_J[1]), CLARGFLOAT((float)mean_J[2]),
      CLARGFLOAT((float)proj_coeff[0]), CLARGFLOAT((float)proj_coeff[1]), CLARGFLOAT((float)proj_coeff[2]),
      CLARGFLOAT((float)scale_rw), CLARGFLOAT((float)rho),
      CLLOCAL(lmem_p23));
  if(err != CL_SUCCESS) return -1.0f;

  err = dt_opencl_read_buffer_from_device(devid, (void *)host_p23, cl_sums_p23, 0,
                                          (size_t)ngroups * 9 * sizeof(float), CL_TRUE);
  if(err != CL_SUCCESS) return -1.0f;

  double H00 = 0.0, H01 = 0.0, H02 = 0.0;
  double H11 = 0.0, H12 = 0.0, H22 = 0.0;
  double rhs0 = 0.0, rhs1 = 0.0, rhs2 = 0.0;
  for(int gi = 0; gi < ngroups; gi++)
  {
    const float *p = host_p23 + gi * 9;
    H00  += (double)p[0];
    H01  += (double)p[1];
    H02  += (double)p[2];
    H11  += (double)p[3];
    H12  += (double)p[4];
    H22  += (double)p[5];
    rhs0 += (double)p[6];
    rhs1 += (double)p[7];
    rhs2 += (double)p[8];
  }

  // Assemble symmetric Hessian, apply Tikhonov regularisation, solve inline.
  double Hess[3][3] = { { H00, H01, H02 },
                         { H01, H11, H12 },
                         { H02, H12, H22 } };
  {
    const double trace = Hess[0][0] + Hess[1][1] + Hess[2][2];
    const double lambda = 0.01 * trace / 3.0;
    for(int k = 0; k < 3; k++) Hess[k][k] += lambda;
  }

  double rhs_ecc[3] = { rhs0, rhs1, rhs2 };
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

  const double max_dtheta = 0.01;
  const double max_shift_n = 0.10;
  dtheta = CLAMP(dtheta, -max_dtheta, max_dtheta);
  dtx    = CLAMP(dtx,    -max_shift_n, max_shift_n);
  dty    = CLAMP(dty,    -max_shift_n, max_shift_n);

  // Apply Euclidean update exactly, preserving any scale encoded in Hn.
  const double sc = sqrt((double)cos_t * cos_t + (double)sin_t * sin_t);
  const double theta_old = atan2((double)sin_t, (double)cos_t);
  const double theta_new = theta_old + dtheta;

  Hn[0] =  (float)(sc * cos(theta_new));
  Hn[1] =  (float)(sc * sin(theta_new));
  Hn[2] += (float)dtx;
  Hn[3] = -(float)(sc * sin(theta_new));
  Hn[4] =  (float)(sc * cos(theta_new));
  Hn[5] += (float)dty;
  Hn[6] = 0.0f;
  Hn[7] = 0.0f;

  _homography_normalized_to_pixel(Hn, w, h, H);

  const double trans_pix = s * (fabs(dtx) + fabs(dty));
  const double angle_pix = s * fabs(dtheta);

  return (float)(trans_pix + angle_pix);
}

/** GPU-accelerated ECC level refinement.
 *
 *  Uploads ref_grad and img_grad once, runs the iteration loop on the GPU,
 *  frees all device/host buffers on exit.
 *
 *  Returns TRUE on success.  Returns FALSE on any CL allocation or dispatch
 *  error; the caller should fall back to the CPU path. */
static gboolean _ecc_refine_level_cl(const int devid,
                                      const dt_hdr_alignment_cl_global_t *g,
                                      const float *ref_grad,
                                      const float *img_grad,
                                      const int w,
                                      const int h,
                                      float H[HDR_ALIGN_H_NPARAM])
{
  const size_t npix = (size_t)w * h;
  const int nwgx = (w + HDR_ALIGN_CL_WG_W - 1) / HDR_ALIGN_CL_WG_W;
  const int nwgy = (h + HDR_ALIGN_CL_WG_H - 1) / HDR_ALIGN_CL_WG_H;
  const int ngroups = nwgx * nwgy;

  // --- Allocate device buffers ---
  cl_mem cl_ref    = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_img    = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_warped = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_mask   = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_gx     = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_gy     = dt_opencl_alloc_device_buffer(devid, npix * sizeof(float));
  cl_mem cl_sums_p1  = dt_opencl_alloc_device_buffer(devid, (size_t)ngroups * 4 * sizeof(float));
  cl_mem cl_sums_p23 = dt_opencl_alloc_device_buffer(devid, (size_t)ngroups * 9 * sizeof(float));

  gboolean ok = (cl_ref && cl_img && cl_warped && cl_mask
                 && cl_gx && cl_gy && cl_sums_p1 && cl_sums_p23);

  // --- Allocate host staging buffers ---
  float *host_p1  = ok ? dt_alloc_align_float((size_t)ngroups * 4) : NULL;
  float *host_p23 = ok ? dt_alloc_align_float((size_t)ngroups * 9) : NULL;
  ok = ok && host_p1 && host_p23;

  if(ok)
  {
    // Upload static per-level gradient images once.
    cl_int err;
    err = dt_opencl_write_buffer_to_device(devid, (void *)ref_grad, cl_ref, 0,
                                           npix * sizeof(float), CL_TRUE);
    if(err != CL_SUCCESS) ok = FALSE;
  }
  if(ok)
  {
    cl_int err;
    err = dt_opencl_write_buffer_to_device(devid, (void *)img_grad, cl_img, 0,
                                           npix * sizeof(float), CL_TRUE);
    if(err != CL_SUCCESS) ok = FALSE;
  }

  if(ok)
  {
    float best_update = FLT_MAX;
    int stall_count = 0;
    float H_best[HDR_ALIGN_H_NPARAM];
    memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);

    for(int iter = 0; iter < HDR_ALIGN_ECC_MAX_ITER && ok; iter++)
    {
      const float update = _ecc_iteration_cl(devid, g,
                                              cl_ref, cl_img,
                                              cl_warped, cl_mask, cl_gx, cl_gy,
                                              cl_sums_p1, cl_sums_p23,
                                              host_p1, host_p23,
                                              w, h, ngroups, H);
      if(update < 0.0f)
      {
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] ECC (CL) failed at iteration %d", iter);
        memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
        ok = FALSE;
        break;
      }

      if(update < HDR_ALIGN_ECC_EPSILON)
      {
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] ECC (CL) converged at iteration %d (update=%.6f)",
                 iter, update);
        break;
      }

      if(update < best_update)
      {
        best_update = update;
        stall_count = 0;
        memcpy(H_best, H, sizeof(float) * HDR_ALIGN_H_NPARAM);
      }
      else if(++stall_count >= HDR_ALIGN_ECC_PATIENCE)
      {
        memcpy(H, H_best, sizeof(float) * HDR_ALIGN_H_NPARAM);
        dt_print(DT_DEBUG_HDRMERGE,
                 "[hdr_merge] ECC (CL) stalled at iteration %d"
                 " (update=%.6f, best=%.6f)",
                 iter, update, best_update);
        break;
      }
    }
  }

  dt_opencl_release_mem_object(cl_ref);
  dt_opencl_release_mem_object(cl_img);
  dt_opencl_release_mem_object(cl_warped);
  dt_opencl_release_mem_object(cl_mask);
  dt_opencl_release_mem_object(cl_gx);
  dt_opencl_release_mem_object(cl_gy);
  dt_opencl_release_mem_object(cl_sums_p1);
  dt_opencl_release_mem_object(cl_sums_p23);
  dt_free_align(host_p1);
  dt_free_align(host_p23);

  return ok;
}

// ---------------------------------------------------------------------------
// OpenCL global init / cleanup
// ---------------------------------------------------------------------------

dt_hdr_alignment_cl_global_t *dt_hdr_alignment_init_cl_global(void)
{
  dt_hdr_alignment_cl_global_t *g = malloc(sizeof(*g));
  if(!g) return NULL;

  const int program = 41; // hdr_alignment.cl, from programs.conf

  g->kernel_warp_homography  = dt_opencl_create_kernel(program, "hdr_align_warp_homography");
  g->kernel_compute_gradients = dt_opencl_create_kernel(program, "hdr_align_compute_gradients");
  g->kernel_log1p            = dt_opencl_create_kernel(program, "hdr_align_log1p");
  g->kernel_gradient_sobel_sum = dt_opencl_create_kernel(program, "hdr_align_gradient_sobel_sum");
  g->kernel_normalize_mad    = dt_opencl_create_kernel(program, "hdr_align_normalize_mad");
  g->kernel_gradient_bayer_cfa_sobel = dt_opencl_create_kernel(program, "hdr_align_gradient_bayer_cfa_sobel");
  g->kernel_mosaic_to_green_only = dt_opencl_create_kernel(program, "hdr_align_mosaic_to_green_only");
  g->kernel_downsample_2x    = dt_opencl_create_kernel(program, "hdr_align_downsample_2x");
  g->kernel_ecc_means        = dt_opencl_create_kernel(program, "hdr_align_ecc_means");
  g->kernel_ecc_norms        = dt_opencl_create_kernel(program, "hdr_align_ecc_norms");
  g->kernel_ecc_hessian_final = dt_opencl_create_kernel(program, "hdr_align_ecc_hessian_final");

  return g;
}

void dt_hdr_alignment_free_cl_global(dt_hdr_alignment_cl_global_t *g)
{
  if(!g) return;
  dt_opencl_free_kernel(g->kernel_warp_homography);
  dt_opencl_free_kernel(g->kernel_compute_gradients);
  dt_opencl_free_kernel(g->kernel_log1p);
  dt_opencl_free_kernel(g->kernel_gradient_sobel_sum);
  dt_opencl_free_kernel(g->kernel_normalize_mad);
  dt_opencl_free_kernel(g->kernel_gradient_bayer_cfa_sobel);
  dt_opencl_free_kernel(g->kernel_mosaic_to_green_only);
  dt_opencl_free_kernel(g->kernel_downsample_2x);
  dt_opencl_free_kernel(g->kernel_ecc_means);
  dt_opencl_free_kernel(g->kernel_ecc_norms);
  dt_opencl_free_kernel(g->kernel_ecc_hessian_final);
  free(g);
}

#endif /* HAVE_OPENCL */

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
