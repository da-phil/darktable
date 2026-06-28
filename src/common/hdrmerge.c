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

#include "common/hdrmerge.h"
#include "common/darktable.h"
#include "common/opencl.h"
#include "develop/pixelpipe.h"

#include <float.h>
#include <math.h>

// Weight envelope. weight_type DT_HDRMERGE_WEIGHT_TRIANGULAR uses the vkdt
// triangular hat, otherwise the Jakob/hdrmerge exponential envelope. Keep in
// sync with hdrmerge_weight() in data/kernels/hdrmerge.cl. The argument s is
// the normalized sensor signal (0 = black, 1 = saturation).
static inline float _hdrmerge_weight(const float s, const int weight_type)
{
  if(s <= 0.0f || s >= 1.0f)
    return 0.0f;
  if(weight_type == DT_HDRMERGE_WEIGHT_TRIANGULAR)
    return 1.0f - fabsf(2.0f * s - 1.0f);
  return DT_HDRMERGE_BETA * expf(DT_HDRMERGE_ALPHA * (1.0f / s + 1.0f / (1.0f - s)));
}

// The weight is a 1-D function of the normalized brightness s in [0,1]. The
// exponential envelope costs an expf() that the inner loop would pay 2*N times
// per pixel; precomputing it into a small LUT (built once, read-only) and
// looking it up with linear interpolation makes the merge ~1.5-1.8x faster.
// The interpolation error is <=2.6e-6 on the weight and <=6e-8 on the
// normalized output vs. a live expf() - far below fp16 / sensor noise.
#define DT_HDRMERGE_LUT_SIZE 4096

static void _hdrmerge_build_weight_lut(float *const lut, const int weight_type)
{
  for(int k = 0; k <= DT_HDRMERGE_LUT_SIZE; k++)
    lut[k] = _hdrmerge_weight((float)k / (float)DT_HDRMERGE_LUT_SIZE, weight_type);
}

static inline float _hdrmerge_weight_lut(const float *const lut, const float s)
{
  if(s <= 0.0f || s >= 1.0f)
    return 0.0f;
  const float f = s * (float)DT_HDRMERGE_LUT_SIZE;
  const int k = (int)f;
  return lut[k] + (lut[k + 1] - lut[k]) * (f - (float)k);
}

// De-ghosting fall-off. dev is how far a frame's luminance sits from the
// consensus, span = deghost * L_ref the deviation at which the frame is fully
// rejected. Returns 1 at perfect agreement and ramps smoothly (Hermite) to 0 at
// the threshold. A hard binary veto at the threshold makes the rejected region
// flip in/out frame by frame, which demosaics into blocky (often magenta)
// edges; the feathered ramp keeps the transition smooth. Keep in sync with
// hdrmerge_deghost_falloff() in data/kernels/hdrmerge.cl.
static inline float _hdrmerge_deghost_falloff(const float dev, const float span)
{
  if(span <= 0.0f)
    return 1.0f;
  const float t = dev < span ? dev / span : 1.0f;
  return 1.0f - t * t * (3.0f - 2.0f * t); // 1 - smoothstep(0, 1, t)
}

// Smooth clip exclusion. 1 for a well-exposed sample, ramping continuously to 0
// as the (sharp) luminance approaches the white point, and 0 at/above it. This
// replaces a hard per-pixel clip gate: a binary in/out flips a frame frame-by-
// frame along bright motion borders and demosaics into PIXELATED edges, whereas
// the smooth ramp keeps the transition continuous. It still reaches 0 at the
// white point, so a clipped sample (e.g. a moving specular highlight) cannot leak
// its value into the merge. Evaluated on the SHARP luma so small clipped glints
// are still caught. Keep in sync with hdrmerge_clip_weight() in hdrmerge.cl.
#define DT_HDRMERGE_CLIP_EDGE 0.7f // full weight below this fraction of white
// Upper edge of the "defer a clipping frame to the reference" ramp, as a ratio of
// reference-to-own luminance: defer fully once the reference is this much brighter.
#ifndef DT_HDRMERGE_DEFER_HI
#define DT_HDRMERGE_DEFER_HI 1.10f
#endif
static inline float _hdrmerge_clip_weight(const float L, const float white)
{
  if(L <= 0.0f || L >= white)
    return 0.0f;
  const float s = L / white;
  if(s <= DT_HDRMERGE_CLIP_EDGE)
    return 1.0f;
  const float t = (s - DT_HDRMERGE_CLIP_EDGE) / (1.0f - DT_HDRMERGE_CLIP_EDGE);
  return 1.0f - t * t * (3.0f - 2.0f * t); // smoothstep 1 -> 0 toward white
}

// Robustness cap for the de-ghost median consensus: an on-stack / private array
// holds the per-frame luminance radiances to take their median. Brackets are
// rarely deeper than this; beyond it the de-ghost falls back to the mean.
#define DT_HDRMERGE_MAX_FRAMES 16

// Median of n (<= DT_HDRMERGE_MAX_FRAMES) values; sorts a in place. Used as the
// de-ghost consensus: unlike the weighted mean it is not dragged toward a moving
// object that only shows up in a minority of frames, so the mover stands out
// against it and is faded cleanly. Keep in sync with hdrmerge.cl.
static inline float _hdrmerge_median(float *const a, const int n)
{
  for(int i = 1; i < n; i++) // insertion sort (n is small)
  {
    const float v = a[i];
    int j = i - 1;
    while(j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
    a[j + 1] = v;
  }
  return (n & 1) ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

// Pick the global de-ghost reference frame: the one that resolves the most
// pixels (0 < L < white). Reference-frame de-ghosting anchors every moving
// pixel to this SINGLE frame, so changing content stays spatially coherent (one
// frame's view -> no fragmented ghosts and no per-channel false colour), unlike
// a per-pixel consensus whose frame set varies from pixel to pixel. The count is
// a global statistic, so it is sampled on a stride for speed; ties break toward
// the shorter exposure (larger cal) to stay deterministic across input order.
static int _hdrmerge_pick_reference(const dt_hdrmerge_half_t *const *const luma,
                                    const dt_hdrmerge_half_t *const *const frames,
                                    const float *const cal, const int n,
                                    const size_t npx, const float white)
{
  const size_t cap = (size_t)1 << 20;        // ~1M samples is plenty
  const size_t stride = (npx > cap) ? npx / cap : 1;
  int best = 0;
  long best_cnt = -1;
  for(int i = 0; i < n; i++)
  {
    const dt_hdrmerge_half_t *const pl = luma ? luma[i] : frames[i];
    long cnt = 0;
    for(size_t p = 0; p < npx; p += stride)
    {
      const float L = dt_hdrmerge_half_to_float(pl[p]);
      if(L > 0.0f && L < white) cnt++;
    }
    if(cnt > best_cnt || (cnt == best_cnt && cal[i] > cal[best]))
    {
      best_cnt = cnt;
      best = i;
    }
  }
  return best;
}

// Spatial smoothing of the luma proxy. The per-quad frame SELECTION (the merge
// weights and the de-ghost consensus, both derived from the luma) is what was
// jumping between neighbouring quads on moving content; that injects a quad-grid
// pattern into the merged CFA that the demosaicer turns into magenta/green
// fringing (and pixelated de-ghost borders). Driving the weights from a blurred
// luma makes the selection vary smoothly across space and removes both. The
// merged pixel VALUES stay sharp - only the per-frame weights are smoothed - so
// no real detail is lost. The clip-gate and highlight neutralization keep using
// the SHARP luma, so smoothing never drags clipped data into a valid pixel.
#define DT_HDRMERGE_BLUR_RADIUS 4
#define DT_HDRMERGE_BLUR_SIGMA 2.0f

static void _hdrmerge_blur_plane(const dt_hdrmerge_half_t *const src,
                                 dt_hdrmerge_half_t *const dst,
                                 float *const tmp, const int wd, const int ht)
{
  const int R = DT_HDRMERGE_BLUR_RADIUS;
  float k[DT_HDRMERGE_BLUR_RADIUS + 1];
  float ksum = 0.0f;
  for(int r = 0; r <= R; r++)
  {
    k[r] = expf(-0.5f * (float)(r * r) / (DT_HDRMERGE_BLUR_SIGMA * DT_HDRMERGE_BLUR_SIGMA));
    ksum += (r == 0) ? k[r] : 2.0f * k[r];
  }
  const float inv = 1.0f / ksum;

  // horizontal pass: src (half) -> tmp (float), edge-clamped
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < ht; y++)
    for(int x = 0; x < wd; x++)
    {
      float acc = k[0] * dt_hdrmerge_half_to_float(src[x + (size_t)wd * y]);
      for(int r = 1; r <= R; r++)
      {
        const int xl = (x - r < 0) ? 0 : x - r;
        const int xr = (x + r >= wd) ? wd - 1 : x + r;
        acc += k[r] * (dt_hdrmerge_half_to_float(src[xl + (size_t)wd * y])
                       + dt_hdrmerge_half_to_float(src[xr + (size_t)wd * y]));
      }
      tmp[x + (size_t)wd * y] = acc * inv;
    }

  // vertical pass: tmp (float) -> dst (half), edge-clamped
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < ht; y++)
    for(int x = 0; x < wd; x++)
    {
      float acc = k[0] * tmp[x + (size_t)wd * y];
      for(int r = 1; r <= R; r++)
      {
        const int yl = (y - r < 0) ? 0 : y - r;
        const int yr = (y + r >= ht) ? ht - 1 : y + r;
        acc += k[r] * (tmp[x + (size_t)wd * yl] + tmp[x + (size_t)wd * yr]);
      }
      dst[x + (size_t)wd * y] = dt_hdrmerge_float_to_half(acc * inv);
    }
}

// Blur every luma plane into a freshly allocated packed buffer (n planes of
// npx, plane i at buf + i*npx). Returns NULL on allocation failure or when there
// is no luma plane, in which case the caller falls back to the sharp luma.
static dt_hdrmerge_half_t *_hdrmerge_blur_luma(const dt_hdrmerge_half_t *const *const luma,
                                               const int n, const int wd, const int ht)
{
  if(!luma)
    return NULL;
  const size_t npx = (size_t)wd * ht;
  dt_hdrmerge_half_t *buf = dt_alloc_align_type(dt_hdrmerge_half_t, (size_t)n * npx);
  float *tmp = dt_alloc_align_float(npx);
  if(!buf || !tmp)
  {
    dt_free_align(buf);
    dt_free_align(tmp);
    return NULL;
  }
  for(int i = 0; i < n; i++)
    _hdrmerge_blur_plane(luma[i], buf + (size_t)i * npx, tmp, wd, ht);
  dt_free_align(tmp);
  return buf;
}

// ======================================================================
//  Multi-scale (Laplacian-pyramid) CFA blend
// ----------------------------------------------------------------------
// The single-scale weighted average below has to commit, per pixel, to ONE
// blend of the frames. Where that blend changes quickly across the image (a
// moving edge, a highlight boundary) it either seams - a hard per-quad frame
// selection that the demosaicer turns into a quad-grid of magenta/green
// fringing - or, if the weights are pre-blurred to hide the seam, it halos and
// over-triggers the de-ghost.
//
// A Laplacian-pyramid blend (Burt & Adelson 1983, as used by enfuse's exposure
// fusion, Mertens 2007) removes that dilemma by blending every frequency BAND
// with the weights smoothed to the MATCHING scale:
//     L{R}_l = sum_k  G{W_k}_l * L{E_k}_l
// (G = Gaussian pyramid of a frame's normalized weight, L = Laplacian pyramid
// of its radiance). Coarse scales cross-fade gently (no seam) while fine scales
// keep each frame's own detail (no halo); collapsing L{R} gives the merged
// radiance.
//
// CFA correctness: a Gaussian blur must never mix colour channels, so each
// frame is de-interleaved into its four 2x2 Bayer positions, each a half-
// resolution plane, and the pyramid is built per position. The weight is the
// quad-shared luma proxy - already one value per 2x2 quad - so a SINGLE weight
// pyramid drives all four positions: every channel of a quad stays the same
// frame mixture at every scale, which is exactly what keeps the demosaicer from
// inventing colour. Re-interleaving the four collapsed planes yields the merged
// CFA, still scene-linear. The pyramid's own multi-scale smoothing replaces the
// host-side luma blur the single-scale path needs, so here the weights come
// from the SHARP luma. X-Trans (non-2x2 CFA) and the no-luma legacy case fall
// back to the single-scale blend.

#define DT_HDRMERGE_PYR_MAX 12     // covers > 16 Mpx half-resolution planes
#define DT_HDRMERGE_PYR_MIN_DIM 8  // stop refining below this coarse size

// The pyramid blends in a LOG (perceptual) domain, not linear radiance. A
// multi-scale blend spreads any cross-frame disagreement (a clipped highlight,
// a moving object) into a low-frequency halo; in linear radiance that halo
// scales with the absolute brightness of the disagreement, so a bright mover or
// a clipped highlight casts a large, per-channel-different halo over its
// surroundings that demosaics into a green/yellow tint. In log radiance the
// disagreement is a bounded *contrast* (log of the ratio), so the halo is a
// small, near-achromatic relative ripple instead - the brighter the
// disagreement, the larger the reduction. EPS is the shadow floor: m + EPS is
// linear for m << EPS (no log blow-up / noise gain at black) and logarithmic
// above it. Frames that agree still reconstruct exactly (equal log values).
#define DT_HDRMERGE_PYR_EPS 1e-3f
static inline float _hdrmerge_pyr_fwd(const float m) { return logf(fmaxf(0.0f, m) + DT_HDRMERGE_PYR_EPS); }
static inline float _hdrmerge_pyr_inv(const float g) { return fmaxf(0.0f, expf(g) - DT_HDRMERGE_PYR_EPS); }

typedef struct _hdrmerge_pyr_t
{
  int nlevels;
  int w[DT_HDRMERGE_PYR_MAX];
  int h[DT_HDRMERGE_PYR_MAX];
  float *lev[DT_HDRMERGE_PYR_MAX];
} _hdrmerge_pyr_t;

// number of pyramid levels for a base of w x h: halve until a dimension would
// drop below DT_HDRMERGE_PYR_MIN_DIM, capped at DT_HDRMERGE_PYR_MAX.
static int _hdrmerge_pyr_nlevels(int w, int h)
{
  int n = 1;
  while(n < DT_HDRMERGE_PYR_MAX && w >= DT_HDRMERGE_PYR_MIN_DIM && h >= DT_HDRMERGE_PYR_MIN_DIM)
  {
    w = (w + 1) / 2;
    h = (h + 1) / 2;
    n++;
  }
  return n;
}

static gboolean _hdrmerge_pyr_alloc(_hdrmerge_pyr_t *const p, int w, int h,
                                    const int nlev, const gboolean zero)
{
  p->nlevels = nlev;
  for(int l = 0; l < nlev; l++)
  {
    p->w[l] = w;
    p->h[l] = h;
    p->lev[l] = dt_alloc_align_float((size_t)w * h);
    if(!p->lev[l])
      return FALSE;
    if(zero)
      memset(p->lev[l], 0, sizeof(float) * (size_t)w * h);
    w = (w + 1) / 2;
    h = (h + 1) / 2;
  }
  return TRUE;
}

static void _hdrmerge_pyr_free(_hdrmerge_pyr_t *const p)
{
  for(int l = 0; l < DT_HDRMERGE_PYR_MAX; l++)
  {
    dt_free_align(p->lev[l]);
    p->lev[l] = NULL;
  }
}

// separable [1 4 6 4 1]/16 blur + subsample by 2, edge-clamped. dst is
// ((sw+1)/2) x ((sh+1)/2); sc is scratch of at least dw*sh floats.
static void _hdrmerge_reduce(const float *const src, const int sw, const int sh,
                             float *const dst, const int dw, const int dh, float *const sc)
{
  static const float K[5] = { 1.f, 4.f, 6.f, 4.f, 1.f };
  // horizontal: src(sw x sh) -> sc(dw x sh)
  DT_OMP_FOR()
  for(int y = 0; y < sh; y++)
    for(int xo = 0; xo < dw; xo++)
    {
      const int xc = 2 * xo;
      float acc = 0.f;
      for(int k = -2; k <= 2; k++)
      {
        int xx = xc + k;
        xx = xx < 0 ? 0 : (xx >= sw ? sw - 1 : xx);
        acc += K[k + 2] * src[(size_t)y * sw + xx];
      }
      sc[(size_t)y * dw + xo] = acc * (1.f / 16.f);
    }
  // vertical: sc(dw x sh) -> dst(dw x dh)
  DT_OMP_FOR()
  for(int yo = 0; yo < dh; yo++)
    for(int x = 0; x < dw; x++)
    {
      const int yc = 2 * yo;
      float acc = 0.f;
      for(int k = -2; k <= 2; k++)
      {
        int yy = yc + k;
        yy = yy < 0 ? 0 : (yy >= sh ? sh - 1 : yy);
        acc += K[k + 2] * sc[(size_t)yy * dw + x];
      }
      dst[(size_t)yo * dw + x] = acc * (1.f / 16.f);
    }
}

// separable upsample of src(sw x sh) to dst(dw x dh), dw ~ 2*sw, edge-clamped.
// Burt & Adelson expand with the gain folded into the kernel: an even output
// is (s[-1] + 6 s[0] + s[+1]) / 8, an odd one the linear midpoint (s[lo]+s[hi])/2.
// sc is scratch of at least dw*sh floats.
static void _hdrmerge_expand(const float *const src, const int sw, const int sh,
                             float *const dst, const int dw, const int dh, float *const sc)
{
  // horizontal: src(sw x sh) -> sc(dw x sh)
  DT_OMP_FOR()
  for(int y = 0; y < sh; y++)
  {
    const float *const srow = src + (size_t)y * sw;
    float *const drow = sc + (size_t)y * dw;
    for(int xo = 0; xo < dw; xo++)
    {
      if((xo & 1) == 0)
      {
        const int s = xo / 2;
        const int sm = s - 1 < 0 ? 0 : s - 1;
        const int s0 = s >= sw ? sw - 1 : s;
        const int sp = s + 1 >= sw ? sw - 1 : s + 1;
        drow[xo] = (srow[sm] + 6.f * srow[s0] + srow[sp]) * (1.f / 8.f);
      }
      else
      {
        const int lo = xo / 2;
        const int lo0 = lo >= sw ? sw - 1 : lo;
        const int hi = lo + 1 >= sw ? sw - 1 : lo + 1;
        drow[xo] = 0.5f * (srow[lo0] + srow[hi]);
      }
    }
  }
  // vertical: sc(dw x sh) -> dst(dw x dh)
  DT_OMP_FOR()
  for(int yo = 0; yo < dh; yo++)
  {
    if((yo & 1) == 0)
    {
      const int s = yo / 2;
      const int sm = s - 1 < 0 ? 0 : s - 1;
      const int s0 = s >= sh ? sh - 1 : s;
      const int sp = s + 1 >= sh ? sh - 1 : s + 1;
      const float *const r0 = sc + (size_t)s0 * dw;
      const float *const rm = sc + (size_t)sm * dw;
      const float *const rp = sc + (size_t)sp * dw;
      float *const drow = dst + (size_t)yo * dw;
      for(int x = 0; x < dw; x++)
        drow[x] = (rm[x] + 6.f * r0[x] + rp[x]) * (1.f / 8.f);
    }
    else
    {
      const int lo = yo / 2;
      const int lo0 = lo >= sh ? sh - 1 : lo;
      const int hi = lo + 1 >= sh ? sh - 1 : lo + 1;
      const float *const rl = sc + (size_t)lo0 * dw;
      const float *const rh = sc + (size_t)hi * dw;
      float *const drow = dst + (size_t)yo * dw;
      for(int x = 0; x < dw; x++)
        drow[x] = 0.5f * (rl[x] + rh[x]);
    }
  }
}

// fill lev[1..] from lev[0] (Gaussian pyramid).
static void _hdrmerge_build_gaussian(_hdrmerge_pyr_t *const p, float *const sc)
{
  for(int l = 1; l < p->nlevels; l++)
    _hdrmerge_reduce(p->lev[l - 1], p->w[l - 1], p->h[l - 1],
                     p->lev[l], p->w[l], p->h[l], sc);
}

// convert a Gaussian pyramid in place to a Laplacian one: lev[l] -= expand(lev[l+1]),
// ascending so each lev[l+1] is still Gaussian when read. tmp is scratch the size
// of the base level; sc the reduce/expand scratch.
static void _hdrmerge_to_laplacian(_hdrmerge_pyr_t *const p, float *const tmp, float *const sc)
{
  for(int l = 0; l < p->nlevels - 1; l++)
  {
    _hdrmerge_expand(p->lev[l + 1], p->w[l + 1], p->h[l + 1], tmp, p->w[l], p->h[l], sc);
    const size_t npx = (size_t)p->w[l] * p->h[l];
    DT_OMP_FOR()
    for(size_t i = 0; i < npx; i++)
      p->lev[l][i] -= tmp[i];
  }
}

// collapse a Laplacian pyramid in place (lev[0] becomes the reconstruction),
// descending so each lev[l+1] is already collapsed when read.
static void _hdrmerge_collapse(_hdrmerge_pyr_t *const p, float *const tmp, float *const sc)
{
  for(int l = p->nlevels - 2; l >= 0; l--)
  {
    _hdrmerge_expand(p->lev[l + 1], p->w[l + 1], p->h[l + 1], tmp, p->w[l], p->h[l], sc);
    const size_t npx = (size_t)p->w[l] * p->h[l];
    DT_OMP_FOR()
    for(size_t i = 0; i < npx; i++)
      p->lev[l][i] += tmp[i];
  }
}

// In-place separable Gaussian blur of a float plane (edge-clamped). Used to
// feather the merge: a controllable pre-blur of the per-frame weight plane
// softens exposure/motion transitions, and a blur of the highlight-neutralization
// mask turns its hard per-quad core boundary into a smooth border. sc is scratch
// of at least w*h floats. sigma <= 0 is a no-op.
static void _hdrmerge_blur_float(float *const p, const int w, const int h,
                                 const float sigma, float *const sc)
{
  if(sigma <= 0.0f)
    return;
  int R = (int)ceilf(2.5f * sigma);
  if(R < 1) R = 1;
  if(R > 48) R = 48;
  float k[49];
  float ksum = 0.0f;
  for(int r = 0; r <= R; r++)
  {
    k[r] = expf(-0.5f * (float)(r * r) / (sigma * sigma));
    ksum += (r == 0) ? k[r] : 2.0f * k[r];
  }
  const float inv = 1.0f / ksum;
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      float acc = k[0] * p[(size_t)y * w + x];
      for(int r = 1; r <= R; r++)
      {
        const int xl = (x - r < 0) ? 0 : x - r;
        const int xr = (x + r >= w) ? w - 1 : x + r;
        acc += k[r] * (p[(size_t)y * w + xl] + p[(size_t)y * w + xr]);
      }
      sc[(size_t)y * w + x] = acc * inv;
    }
  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      float acc = k[0] * sc[(size_t)y * w + x];
      for(int r = 1; r <= R; r++)
      {
        const int yl = (y - r < 0) ? 0 : y - r;
        const int yr = (y + r >= h) ? h - 1 : y + r;
        acc += k[r] * (sc[(size_t)yl * w + x] + sc[(size_t)yr * w + x]);
      }
      p[(size_t)y * w + x] = acc * inv;
    }
}

static inline float _hdrmerge_smoothstep(const float a, const float b, const float x)
{
  if(x <= a) return 0.0f;
  if(x >= b) return 1.0f;
  const float t = (x - a) / (b - a);
  return t * t * (3.0f - 2.0f * t);
}

// Multi-scale CFA merge. Writes h->out (scene-linear, normalized to 1.0 = the
// shortest exposure's saturation). Returns FALSE on allocation failure, in which
// case the caller falls back to the single-scale path. Requires a 2x2 Bayer CFA
// with a luma proxy (the caller guarantees this).
static gboolean _hdrmerge_blend_pyramid(dt_hdrmerge_t *const h, const float inv_cal_max,
                                        const float inv_white, const float *const wlut)
{
  const int n = h->num_frames;
  const int W = h->width, H = h->height;
  const int hw = (W + 1) / 2, hh = (H + 1) / 2; // half-resolution per Bayer position
  const size_t hnpx = (size_t)hw * hh;
  const float white = h->white_thresh;
  const float deghost = h->deghost_threshold;
  const float eps = 1e-6f;
  // feather (0..1) -> Gaussian blur sigmas in half-resolution quad units. A
  // pre-blur of the per-frame weight softens the exposure/motion transition; the
  // neutralization mask always gets a light, feather-INDEPENDENT blur (so a
  // clipped-highlight core has a feathered, not pixelated, border - see nsigma).
  const float feather = (h->feather < 0.0f) ? 0.0f : (h->feather > 1.0f ? 1.0f : h->feather);
  const float wsigma = feather * 4.0f;
  // neutralization-border blur is kept small and feather-INDEPENDENT: it only has
  // to de-pixelate the clipped-core edge; widening it bleeds the neutral white core
  // into the coloured corona and desaturates it (-> magenta), so feather must not
  // drive it.
  const float nsigma = 1.5f;
  const dt_hdrmerge_half_t *const *const frames = h->frames;
  const dt_hdrmerge_half_t *const *const luma = h->luma;
  const float *const cal = h->cal;
  float *const out = h->out;
  const int nlev = _hdrmerge_pyr_nlevels(hw, hh);

  // shortest exposure (largest cal) = the highlight reference. Where every frame
  // clips, the merge converges to THIS frame instead of a uniform average of
  // clipped under-estimates (which desaturates highlights to grey -> magenta
  // after white balance), so highlights keep the shortest exposure's colour and
  // rolloff - i.e. they match developing the darkest bracket directly.
  int ishort = 0;
  for(int i = 1; i < n; i++)
    if(cal[i] > cal[ishort]) ishort = i;

  // de-interleave offsets of the four positions of a 2x2 Bayer quad
  static const int dxq[4] = { 0, 1, 0, 1 };
  static const int dyq[4] = { 0, 0, 1, 1 };

  // per-quad pre-pass results (consensus, weight sum, and two clip signals)
  float *const Lcons = dt_alloc_align_float(hnpx);
  float *const sumw = dt_alloc_align_float(hnpx);
  float *const minb = dt_alloc_align_float(hnpx); // min over frames of the max-channel luma
  // shortest-exposure quad MINIMUM channel: a pixel is unrecoverable (-> white)
  // only where even this clips, i.e. ALL channels clip in every frame. Neutralizing
  // on this (not on minb, which trips as soon as the brightest channel clips) keeps
  // the recoverable, still-coloured corona around a blown highlight instead of
  // whitening it into a hard disc.
  float *const mincore = dt_alloc_align_float(hnpx);
  // reduce/expand scratch and the expand-output temp (both base-level sized)
  float *const sc = dt_alloc_align_float(hnpx);
  float *const tmp = dt_alloc_align_float(hnpx);

  // one Laplacian-radiance accumulator per Bayer position, plus the per-frame
  // working pyramids (weight Gaussian, one channel's Laplacian).
  _hdrmerge_pyr_t acc[4] = { 0 };
  _hdrmerge_pyr_t gw = { 0 }, le = { 0 };

  gboolean ok = (Lcons && sumw && minb && mincore && sc && tmp);
  for(int c = 0; c < 4 && ok; c++)
    ok = _hdrmerge_pyr_alloc(&acc[c], hw, hh, nlev, TRUE);
  if(ok) ok = _hdrmerge_pyr_alloc(&gw, hw, hh, nlev, FALSE);
  if(ok) ok = _hdrmerge_pyr_alloc(&le, hw, hh, nlev, FALSE);
  if(!ok)
    goto cleanup;

  // single global de-ghost reference frame (only when de-ghosting is on)
  const int ref = (deghost > 0.0f)
    ? _hdrmerge_pick_reference(luma, frames, cal, n, (size_t)W * H, white) : 0;

  // pre-pass: per quad, the de-ghost consensus, the sum of per-frame weights
  // (for normalization), and whether every frame is clipped here (-> neutralize).
  DT_OMP_FOR()
  for(size_t q = 0; q < hnpx; q++)
  {
    const int qx = (int)(q % (size_t)hw), qy = (int)(q / (size_t)hw);
    const int sx = (2 * qx < W) ? 2 * qx : W - 1;
    const int sy = (2 * qy < H) ? 2 * qy : H - 1;
    const size_t lp = (size_t)sy * W + sx; // quad-shared luma index (top-left)

    float lnum = 0.f, lden = 0.f, lnumf = 0.f, mb = FLT_MAX;
    for(int i = 0; i < n; i++)
    {
      const float Ls = dt_hdrmerge_half_to_float(luma[i][lp]);
      const float w = _hdrmerge_clip_weight(Ls, white) * _hdrmerge_weight_lut(wlut, Ls * inv_white);
      lnum += w * (Ls * cal[i]);
      lden += w;
      lnumf += Ls * cal[i];
      mb = fminf(mb, Ls);
    }
    float Lc = (lden > eps) ? (lnum / lden) : (lnumf / (float)n);
    if(deghost > 0.0f)
    {
      const float Ls_ref = dt_hdrmerge_half_to_float(luma[ref][lp]);
      if(Ls_ref > 0.0f && Ls_ref < white)
        Lc = Ls_ref * cal[ref]; // anchor to the reference frame's view
      else if(n <= DT_HDRMERGE_MAX_FRAMES) // reference unusable here -> robust median
      {
        float lr[DT_HDRMERGE_MAX_FRAMES];
        int nlr = 0;
        for(int i = 0; i < n; i++)
        {
          const float Ls = dt_hdrmerge_half_to_float(luma[i][lp]);
          if(Ls > 0.0f && Ls < white)
            lr[nlr++] = Ls * cal[i];
        }
        if(nlr >= 3)
          Lc = _hdrmerge_median(lr, nlr);
      }
    }
    // sum of the final per-frame weights (clip * exposure * de-ghost falloff),
    // recomputed identically in the blend loop so the normalization is exact.
    float sw = 0.f;
    for(int i = 0; i < n; i++)
    {
      const float Ls = dt_hdrmerge_half_to_float(luma[i][lp]);
      float w = _hdrmerge_clip_weight(Ls, white) * _hdrmerge_weight_lut(wlut, Ls * inv_white);
      if(w > 0.0f && deghost > 0.0f)
        w *= _hdrmerge_deghost_falloff(fabsf(Ls * cal[i] - Lc), deghost * Lc);
      sw += w;
    }
    // shortest-exposure quad minimum channel (min over the 4 Bayer positions):
    // flags a truly unrecoverable (all-channel-clipped) highlight core.
    float mc = FLT_MAX;
    for(int c = 0; c < 4; c++)
    {
      int mcx = 2 * qx + dxq[c]; if(mcx >= W) mcx = W - 1;
      int mcy = 2 * qy + dyq[c]; if(mcy >= H) mcy = H - 1;
      mc = fminf(mc, dt_hdrmerge_half_to_float(frames[ishort][(size_t)mcy * W + mcx]));
    }
    Lcons[q] = Lc;
    sumw[q] = sw;
    minb[q] = mb;
    mincore[q] = mc;
  }

  // accumulate sum_k G{W_k} * L{E_k,c} over the frames, per Bayer position
  for(int i = 0; i < n; i++)
  {
    const float c_i = cal[i];
    // normalized weight plane at base resolution (same weight as the pre-pass)
    DT_OMP_FOR()
    for(size_t q = 0; q < hnpx; q++)
    {
      const int qx = (int)(q % (size_t)hw), qy = (int)(q / (size_t)hw);
      const int sx = (2 * qx < W) ? 2 * qx : W - 1;
      const int sy = (2 * qy < H) ? 2 * qy : H - 1;
      const float Ls = dt_hdrmerge_half_to_float(luma[i][(size_t)sy * W + sx]);
      float w = _hdrmerge_clip_weight(Ls, white) * _hdrmerge_weight_lut(wlut, Ls * inv_white);
      if(w > 0.0f && deghost > 0.0f)
        w *= _hdrmerge_deghost_falloff(fabsf(Ls * c_i - Lcons[q]), deghost * Lcons[q]);
      // fallback when every frame is excluded here: in a clipped highlight
      // (minb >= white) put all weight on the shortest exposure (the highlight
      // reference) so the merge keeps its colour/rolloff rather than averaging
      // clipped under-estimates to grey; elsewhere (e.g. deep shadow) keep uniform.
      gw.lev[0][q] = (sumw[q] > eps) ? (w / sumw[q])
                     : (minb[q] >= white ? ((i == ishort) ? 1.0f : 0.0f)
                                         : (1.0f / (float)n));
    }
    // optional feather: blur the (normalized) weight to soften the exposure/motion
    // transition. But keep the SHARP weight in clipped highlights - there the merge
    // already converges to the single shortest exposure (no exposure seam to
    // feather), and blurring would bleed the neutral white core's selection into
    // the coloured corona, desaturating it into magenta. The handoff is feathered
    // by the shortest-exposure luma (minb). All frames share the kernel and
    // sum_k W_k = 1, which both the blur and the lerp preserve, so the partition of
    // unity (hence no colour bias) holds.
    if(wsigma > 0.0f)
    {
      memcpy(tmp, gw.lev[0], hnpx * sizeof(float)); // sharp copy (tmp is free here)
      _hdrmerge_blur_float(gw.lev[0], hw, hh, wsigma, sc);
      DT_OMP_FOR()
      for(size_t q = 0; q < hnpx; q++)
      {
        const float hl = _hdrmerge_smoothstep(white * 0.6f, white * 0.85f, minb[q]);
        gw.lev[0][q] = (1.0f - hl) * gw.lev[0][q] + hl * tmp[q];
      }
    }
    _hdrmerge_build_gaussian(&gw, sc);

    // a frame whose brightest channel clips here (luma -> white) is unreliable:
    // its clipped channel is capped to `white` and so UNDER-estimated, while its
    // other channels keep their true (lower) values - a colour the demosaicer
    // reads as magenta (green, the first to clip on these sensors, is capped while
    // red/blue are not). Its finest-level weight is already ~0, but the pyramid's
    // COARSE levels still leak this capped colour into the surrounding highlight.
    // So as it approaches clipping, fade its pyramid radiance toward the SHORTEST
    // exposure's (the highlight reference, which still has head-room), splitting
    // COLOUR from BRIGHTNESS: adopt the reference's colour ratios fully (fixes the
    // magenta everywhere the frame clips, including faint speculars and disc
    // coronae), but lift the luminance toward the reference only where it is
    // genuinely brighter (so a real clipped highlight also recovers its brightness,
    // while moved / mis-registered content keeps its own and cannot ring into an
    // edge speckle - see the inner comment). The whole pixel is faded (all four
    // positions) so channels stay coherent - never sourced half from one frame and
    // half from another (that is itself false colour). The fade is a smoothstep on
    // the frame's own luma (not a hard switch at `white`) so it introduces no step
    // in the radiance plane - a step would ring through the pyramid into speckles
    // along high-contrast edges. i == ishort keeps its own value (nothing brighter
    // to defer to; an all-clip core is handled by the neutralization pass below).
    const float cshort = cal[ishort] * inv_cal_max;
    const float defer_lo = white * 0.9f; // start deferring to the reference here
    for(int c = 0; c < 4; c++)
    {
      // this position's radiance plane E = X * cal (edge-clamped on odd dims)
      DT_OMP_FOR()
      for(size_t q = 0; q < hnpx; q++)
      {
        const int qx = (int)(q % (size_t)hw), qy = (int)(q / (size_t)hw);
        int sx = 2 * qx + dxq[c]; if(sx >= W) sx = W - 1;
        int sy = 2 * qy + dyq[c]; if(sy >= H) sy = H - 1;
        const size_t sp = (size_t)sy * W + sx;
        // normalized radiance m = X * cal / cal_max, blended in the log domain
        const float own = dt_hdrmerge_half_to_float(frames[i][sp]) * c_i * inv_cal_max;
        const float Ls = dt_hdrmerge_half_to_float(luma[i][sp]);
        float m = own;
        if(i != ishort && Ls > defer_lo)
        {
          const float t = _hdrmerge_smoothstep(defer_lo, white, Ls);
          const float own_l = Ls * c_i * inv_cal_max;                                 // this frame's luminance-radiance
          const float ref_l = dt_hdrmerge_half_to_float(luma[ishort][sp]) * cshort;   // reference luminance-radiance
          const float ms    = dt_hdrmerge_half_to_float(frames[ishort][sp]) * cshort; // reference channel c radiance
          if(ref_l > eps)
          {
            const float ratio = ref_l / own_l;    // > 1 where the reference is brighter here
            // COLOUR: always adopt the reference's channel ratios. This repairs the
            // capped channel and removes the magenta even on faint speculars and on
            // the coronae of clipped discs - where own and reference luminance are
            // close, so a brightness-gated substitution would leave the colour (and
            // its magenta) untouched. BRIGHTNESS: keep this frame's OWN (capped)
            // luminance, lifted toward the reference's only to the extent the
            // reference is genuinely brighter here (g). Moved / mis-registered
            // content, where THIS frame is the brighter one (g -> 0), keeps its own
            // luminance and so can never punch a dark hole that would ring into a
            // speckle along a high-contrast edge. g=1: target = ms (colour +
            // brightness); g=0: target = ms/ratio (reference colour at own luma).
            const float g = _hdrmerge_smoothstep(1.0f, DT_HDRMERGE_DEFER_HI, ratio);
            const float target = ms * (g + (1.0f - g) / ratio);
            m = (1.0f - t) * own + t * target;
          }
        }
        le.lev[0][q] = _hdrmerge_pyr_fwd(m);
      }
      _hdrmerge_build_gaussian(&le, sc);
      _hdrmerge_to_laplacian(&le, tmp, sc);
      for(int l = 0; l < nlev; l++)
      {
        const size_t npl = (size_t)acc[c].w[l] * acc[c].h[l];
        const float *const gwl = gw.lev[l];
        const float *const lel = le.lev[l];
        float *const al = acc[c].lev[l];
        DT_OMP_FOR()
        for(size_t j = 0; j < npl; j++)
          al[j] += gwl[j] * lel[j];
      }
    }
  }

  // collapse each position and re-interleave into the scene-linear CFA output
  for(int c = 0; c < 4; c++)
    _hdrmerge_collapse(&acc[c], tmp, sc);

  // Highlight neutralization, feathered. neutral = 1 only where a highlight is
  // truly unrecoverable: even the shortest exposure's WEAKEST channel clips
  // (mincore >= white), i.e. all channels clip in every frame. Pinning those to
  // white keeps darktable's highlight handling from demosaicing a channel-clip
  // mismatch into magenta. It is deliberately NOT keyed to minb (the brightest
  // channel clipping), which would whiten the still-recoverable, still-coloured
  // corona into a hard disc; that corona now converges to the shortest exposure
  // instead. A tonal ramp below white plus a spatial blur make the core border
  // diffuse rather than a pixelated ring. sumw is done being read; reuse it.
  float *const nf = sumw;
  DT_OMP_FOR()
  for(size_t q = 0; q < hnpx; q++)
    nf[q] = _hdrmerge_smoothstep(white * 0.90f, white, mincore[q]);
  _hdrmerge_blur_float(nf, hw, hh, nsigma, sc); // feather the core boundary

  DT_OMP_FOR()
  for(size_t q = 0; q < hnpx; q++)
  {
    const int qx = (int)(q % (size_t)hw), qy = (int)(q / (size_t)hw);
    const float f = nf[q];
    for(int c = 0; c < 4; c++)
    {
      const int ox = 2 * qx + dxq[c], oy = 2 * qy + dyq[c];
      if(ox < W && oy < H)
      {
        const float rec = _hdrmerge_pyr_inv(acc[c].lev[0][q]);
        out[(size_t)oy * W + ox] = (1.0f - f) * rec + f; // lerp recovered -> white
      }
    }
  }
  ok = TRUE;

cleanup:
  for(int c = 0; c < 4; c++)
    _hdrmerge_pyr_free(&acc[c]);
  _hdrmerge_pyr_free(&gw);
  _hdrmerge_pyr_free(&le);
  dt_free_align(Lcons);
  dt_free_align(sumw);
  dt_free_align(minb);
  dt_free_align(mincore);
  dt_free_align(sc);
  dt_free_align(tmp);
  return ok;
}

void dt_hdrmerge_process_cpu(dt_hdrmerge_t *const h)
{
  const int n = h->num_frames;
  const size_t npx = (size_t)h->width * h->height;
  const float white_thresh = h->white_thresh;
  const float eps = 1e-6f;
  const int weight_type = h->weight;
  const float deghost = h->deghost_threshold;

  // shortest exposure (largest calibration factor) sets the brightest
  // representable radiance, which we normalize the output to.
  float cal_max = 0.0f;
  for(int i = 0; i < n; i++)
    cal_max = fmaxf(cal_max, h->cal[i]);
  const float inv_cal_max = (cal_max > 0.0f) ? 1.0f / cal_max : 1.0f;
  const float inv_white = 1.0f / white_thresh;

  // build the weight envelope LUT once; the inner loop looks it up instead of
  // evaluating expf() 2*N times per pixel.
  float wlut[DT_HDRMERGE_LUT_SIZE + 1];
  _hdrmerge_build_weight_lut(wlut, weight_type);

  // multi-scale (Laplacian-pyramid) CFA blend: cross-fades exposure / motion
  // transitions across scales so they don't demosaic into quad-grid fringing.
  // Bayer + luma proxy only; on allocation failure fall through to the single-
  // scale blend below (which also handles X-Trans and the no-luma legacy case).
  if(h->pyramid && h->luma && !h->xtrans
     && _hdrmerge_blend_pyramid(h, inv_cal_max, inv_white, wlut))
    return;

  const dt_hdrmerge_half_t *const *const frames = h->frames;
  const dt_hdrmerge_half_t *const *const luma = h->luma;
  const float *const cal = h->cal;
  float *const out = h->out;

  // de-ghost reference frame (only needed when de-ghosting is on): the single
  // well-exposed frame moving content is anchored to.
  const int ref = (deghost > 0.0f) ? _hdrmerge_pick_reference(luma, frames, cal, n, npx, white_thresh) : 0;

  // blurred luma drives the per-frame WEIGHTS and the de-ghost consensus so the
  // frame selection varies smoothly across space; NULL => run unsmoothed.
  dt_hdrmerge_half_t *const bluma_buf = _hdrmerge_blur_luma(luma, n, h->width, h->height);

  DT_OMP_FOR()
  for(size_t p = 0; p < npx; p++)
  {
    // pass 1: weight each frame by the shared luminance proxy, so every CFA
    // colour channel at this position gets the SAME per-frame weight. This is
    // what keeps neighbouring R/G/B from being sourced from different frames
    // (which demosaics into magenta/green/purple where content moves). The
    // value being merged is still the channel's own radiance E = X * cal.
    float num = 0.0f;  // sum w * E      (per-channel radiance)
    float den = 0.0f;  // sum w
    float numf = 0.0f; // sum E          (unweighted fallback)
    float lnum = 0.0f; // sum w * L_rad  (luminance radiance)
    float lnumf = 0.0f; // sum L_rad     (unweighted fallback)
    float min_bright = FLT_MAX; // min over frames of the shared brightness
    for(int i = 0; i < n; i++)
    {
      const float c = cal[i];
      const float X = dt_hdrmerge_half_to_float(frames[i][p]);
      const float Ls = luma ? dt_hdrmerge_half_to_float(luma[i][p]) : X;       // sharp
      const float Lw = bluma_buf ? dt_hdrmerge_half_to_float(bluma_buf[(size_t)i * npx + p])
                                 : Ls;                                          // blurred (weights)
      const float E = X * c;
      // exposure weight from the BLURRED luma (spatial coherence) times a SMOOTH
      // clip weight on the SHARP luma (excludes clipped/black samples without a
      // hard per-pixel step -> no pixelated borders).
      const float w = _hdrmerge_clip_weight(Ls, white_thresh)
                      * _hdrmerge_weight_lut(wlut, Lw * inv_white);
      num += w * E;
      den += w;
      numf += E;
      lnum += w * (Lw * c);
      lnumf += Lw * c;
      min_bright = fminf(min_bright, Ls); // sharp: drives the clip neutralization
    }
    // if all frames were black or clipped here, fall back to the plain mean
    const float E_ref = (den > eps) ? (num / den) : (numf / (float)n);
    const float L_ref = (den > eps) ? (lnum / den) : (lnumf / (float)n);

    // De-ghosting consensus. When de-ghosting is on, anchor it to ONE globally
    // chosen well-exposed reference frame: a pixel whose content moved is
    // resolved to the reference's view, which is spatially coherent (one frame
    // -> no fragmented ghosts) and self-consistent across channels (one frame ->
    // no false colour). Frames that disagree with the reference are faded out in
    // pass 2. Where the reference itself did not resolve this pixel (clipped or
    // black there), fall back to the robust median over the frames that did. With
    // de-ghosting off this stays the plain weighted mean.
    float L_cons = L_ref;
    if(deghost > 0.0f)
    {
      // validity from the SHARP luma (does the reference resolve this pixel?),
      // value from the BLURRED luma (smooth consensus -> smooth de-ghost mask).
      const float Ls_ref = luma ? dt_hdrmerge_half_to_float(luma[ref][p])
                                : dt_hdrmerge_half_to_float(frames[ref][p]);
      const float Lw_ref = bluma_buf ? dt_hdrmerge_half_to_float(bluma_buf[(size_t)ref * npx + p])
                                     : Ls_ref;
      if(Ls_ref > 0.0f && Ls_ref < white_thresh)
        L_cons = Lw_ref * cal[ref]; // anchor to the reference frame
      else if(n <= DT_HDRMERGE_MAX_FRAMES) // reference unusable here -> median
      {
        float lr[DT_HDRMERGE_MAX_FRAMES];
        int nlr = 0;
        for(int i = 0; i < n; i++)
        {
          const float Ls = luma ? dt_hdrmerge_half_to_float(luma[i][p])
                                : dt_hdrmerge_half_to_float(frames[i][p]);
          if(Ls > 0.0f && Ls < white_thresh) // not black, not clipped (sharp)
          {
            const float Lw = bluma_buf ? dt_hdrmerge_half_to_float(bluma_buf[(size_t)i * npx + p])
                                       : Ls;
            lr[nlr++] = Lw * cal[i];
          }
        }
        if(nlr >= 3)
          L_cons = _hdrmerge_median(lr, nlr);
      }
    }

    // pass 2: re-weight by the predicted shared luminance from the consensus,
    // and optionally fade out frames whose luminance disagrees with it
    // (de-ghosting of moving content). Everything here is luminance-driven,
    // hence identical across colour channels.
    num = 0.0f;
    den = 0.0f;
    for(int i = 0; i < n; i++)
    {
      const float c = cal[i];
      const float Ls = luma ? dt_hdrmerge_half_to_float(luma[i][p])
                            : dt_hdrmerge_half_to_float(frames[i][p]); // sharp
      // Smooth clip exclusion (SHARP luma): pass 2 weights by the consensus-
      // predicted value and never re-checks the frame's own clip state, so
      // without this a frame that is actually clipped (a moving specular
      // highlight) but predicted mid-range would be let back in and pull its
      // clipped value into the merge -> magenta. The smooth ramp (vs a hard gate)
      // keeps bright motion borders from demosaicing into pixelated edges.
      const float cg = _hdrmerge_clip_weight(Ls, white_thresh);
      const float predicted = L_cons / c; // expected luminance signal in frame i
      float w = cg * _hdrmerge_weight_lut(wlut, predicted * inv_white);
      if(w > 0.0f && deghost > 0.0f)
      {
        // fade the frame out smoothly as its luminance leaves the consensus,
        // measured on the BLURRED luma so the de-ghost mask has feathered (not
        // pixelated) spatial edges.
        const float Lw = bluma_buf ? dt_hdrmerge_half_to_float(bluma_buf[(size_t)i * npx + p])
                                   : Ls;
        w *= _hdrmerge_deghost_falloff(fabsf(Lw * c - L_cons), deghost * L_cons);
      }
      if(w > 0.0f)
      {
        const float X = dt_hdrmerge_half_to_float(frames[i][p]);
        num += w * (X * c);
        den += w;
      }
    }
    const float E = (den > eps) ? (num / den) : E_ref;

    // Unrecoverable highlight: the brightest channel clips in EVERY frame (even
    // the shortest exposure). Its true value is beyond what the bracket caught,
    // so its merged radiance is a clipped under-estimate while the other,
    // unclipped channels keep their (lower) true values -> after white balance
    // that demosaics into magenta. Pin the whole pixel to the white point so
    // darktable's highlight handling sees a uniformly-clipped, neutral pixel.
    out[p] = (luma && min_bright >= white_thresh) ? 1.0f
                                                  : fmaxf(0.0f, E * inv_cal_max);
  }

  dt_free_align(bluma_buf);
}

#ifdef HAVE_OPENCL
static gpointer _hdrmerge_create_kernel(gpointer data)
{
  // hdrmerge.cl is program 42 in data/kernels/programs.conf. The kernel id is
  // process-global and stable once created, so we only do this once. Offset by
  // +1 so a valid id of 0 is still non-NULL for g_once().
  const int program = 42; // hdrmerge.cl, from programs.conf
  const int kernel = dt_opencl_create_kernel(program, "hdrmerge_merge");
  return GINT_TO_POINTER(kernel + 1);
}

static gboolean _hdrmerge_process_cl(dt_hdrmerge_t *const h)
{
  if(!dt_opencl_is_enabled())
    return FALSE;

  // the kernel's highlight neutralization needs the shared brightness signal;
  // without it, fall back to the CPU (which handles the no-luma legacy case).
  if(!h->luma)
    return FALSE;

  static GOnce kernel_once = G_ONCE_INIT;
  const int kernel = GPOINTER_TO_INT(g_once(&kernel_once, _hdrmerge_create_kernel, NULL)) - 1;
  if(kernel < 0)
    return FALSE;

  const int devid = dt_opencl_lock_device(DT_DEV_PIXELPIPE_EXPORT);
  if(devid < 0) // DT_DEVICE_CPU: no OpenCL device available
    return FALSE;

  // luma is guaranteed non-NULL here (the early return above bails to the CPU
  // when it is absent), so the kernel always gets a real shared-brightness plane.
  const int w = h->width;
  const int ht = h->height;
  const int n = h->num_frames;
  const size_t plane = (size_t)w * ht;
  const size_t half_sz = sizeof(dt_hdrmerge_half_t);
  const size_t frames_bytes = (size_t)n * plane * half_sz;
  const size_t out_bytes = plane * sizeof(float);
  const size_t cal_bytes = (size_t)n * sizeof(float);
  const size_t luma_bytes = frames_bytes;
  const float eps = 1e-6f;
  const int weight_type = h->weight;
  const float deghost = h->deghost_threshold;
  // de-ghost reference frame (picked on the host, same as the CPU path).
  const int ref_frame = (deghost > 0.0f)
    ? _hdrmerge_pick_reference(h->luma, h->frames, h->cal, n, plane, h->white_thresh) : 0;

  cl_mem dev_frames = NULL;
  cl_mem dev_luma = NULL;
  cl_mem dev_bluma = NULL;
  cl_mem dev_cal = NULL;
  cl_mem dev_out = NULL;
  gboolean ok = FALSE;
  cl_int err = CL_SUCCESS;

  // blurred luma (computed host-side): drives the weights/consensus for spatial
  // coherence while the sharp luma drives the clip-gate / neutralization.
  // Uploaded as a companion buffer; NULL => fall back to the CPU (also smooths).
  dt_hdrmerge_half_t *const hbluma = _hdrmerge_blur_luma(h->luma, n, w, ht);
  const size_t bluma_bytes = frames_bytes;

  float cal_max = 0.0f;
  for(int i = 0; i < n; i++)
    cal_max = fmaxf(cal_max, h->cal[i]);
  const float inv_cal_max = (cal_max > 0.0f) ? 1.0f / cal_max : 1.0f;

  // memory guard: the packed frame buffer must fit a single allocation and
  // everything together must fit the device's currently available memory.
  // Fall back to the CPU otherwise.
  if(!hbluma)
    goto cleanup;
  if(frames_bytes > darktable.opencl->dev[devid].max_mem_alloc)
    goto cleanup;
  if(frames_bytes + luma_bytes + bluma_bytes + cal_bytes + out_bytes
       > (size_t)dt_opencl_get_device_available(devid))
    goto cleanup;

  dev_frames = dt_opencl_alloc_device_buffer(devid, frames_bytes);
  dev_cal = dt_opencl_alloc_device_buffer(devid, cal_bytes);
  dev_out = dt_opencl_alloc_device_buffer(devid, out_bytes);
  dev_bluma = dt_opencl_alloc_device_buffer(devid, bluma_bytes);
  dev_luma = dt_opencl_alloc_device_buffer(devid, frames_bytes);
  if(!dev_frames || !dev_cal || !dev_out || !dev_bluma || !dev_luma)
    goto cleanup;

  // upload the frames (one plane at a time), the luminance proxy, and calibration.
  for(int i = 0; i < n; i++)
  {
    err = dt_opencl_write_buffer_to_device(devid, (void *)h->frames[i], dev_frames,
                                           (size_t)i * plane * half_sz,
                                           plane * half_sz, CL_TRUE);
    if(err != CL_SUCCESS)
      goto cleanup;
    err = dt_opencl_write_buffer_to_device(devid, (void *)h->luma[i], dev_luma,
                                           (size_t)i * plane * half_sz,
                                           plane * half_sz, CL_TRUE);
    if(err != CL_SUCCESS)
      goto cleanup;
  }
  err = dt_opencl_write_buffer_to_device(devid, (void *)h->cal, dev_cal, 0, cal_bytes, CL_TRUE);
  if(err != CL_SUCCESS)
    goto cleanup;
  // hbluma is one contiguous n-plane buffer, upload in a single transfer.
  err = dt_opencl_write_buffer_to_device(devid, (void *)hbluma, dev_bluma, 0, bluma_bytes, CL_TRUE);
  if(err != CL_SUCCESS)
    goto cleanup;

  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, kernel, w, ht,
                                           CLARG(dev_frames), CLARG(dev_luma), CLARG(dev_bluma),
                                           CLARG(dev_cal), CLARG(dev_out),
                                           CLARG(w), CLARG(ht), CLARG(n),
                                           CLARG(h->white_thresh), CLARG(inv_cal_max), CLARG(eps),
                                           CLARG(weight_type), CLARG(deghost), CLARG(ref_frame));
    if(err != CL_SUCCESS)
      goto cleanup;
  }

  err = dt_opencl_read_buffer_from_device(devid, h->out, dev_out, 0, out_bytes, CL_TRUE);
  if(err != CL_SUCCESS)
    goto cleanup;

  ok = TRUE;

cleanup:
  dt_opencl_release_mem_object(dev_frames);
  dt_opencl_release_mem_object(dev_luma);
  dt_opencl_release_mem_object(dev_bluma);
  dt_opencl_release_mem_object(dev_cal);
  dt_opencl_release_mem_object(dev_out);
  dt_opencl_unlock_device(devid);
  dt_free_align(hbluma);

  if(ok)
    dt_print(DT_DEBUG_OPENCL, "[hdrmerge] merged %d frames of %dx%d on device %d",
             n, w, ht, devid);
  return ok;
}
#endif // HAVE_OPENCL

gboolean dt_hdrmerge_process(dt_hdrmerge_t *const h)
{
  if(h->num_frames < 1 || h->width < 1 || h->height < 1 || !h->out)
    return FALSE;
  if(!(h->white_thresh > 0.0f))
    h->white_thresh = DT_HDRMERGE_DEFAULT_WHITE_THRESH;

#ifdef HAVE_OPENCL
  // the multi-scale blend is a CPU-only path for now (the OpenCL kernel
  // implements the single-scale merge); only try the GPU when it is not used.
  const gboolean use_pyramid = h->pyramid && h->luma && !h->xtrans;
  if(!use_pyramid && _hdrmerge_process_cl(h))
    return TRUE;
#endif

  dt_hdrmerge_process_cpu(h);
  return FALSE;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; tab-width 2; remove-trailing-spaces modified;
// clang-format on
