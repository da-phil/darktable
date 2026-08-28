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

/* Unit tests for the exposure-bracket HDR merge (common/hdrmerge).
 *
 * These synthesise a small Bayer bracket (3 exposures, cal 1/4/16) and drive the
 * public CPU entry point dt_hdrmerge_process_cpu() - which is what the OpenCL
 * dispatcher also falls back to, and the only path for the default multi-scale
 * (pyramid) blend. They lock in the properties that were hard-won bug fixes:
 *
 *   1. the multi-scale blend is energy-preserving  - agreeing frames reconstruct
 *      out = E / cal_max per CFA channel (partition of unity + a lossless pyramid
 *      round-trip + the de-interleave/re-interleave), and so does the linear path;
 *   2. an unrecoverable highlight (every channel clips in every frame) neutralizes
 *      to the white point (1.0), not a channel-clip mismatch (magenta);
 *   3. a bright coloured patch that clips only in the longer exposures keeps its
 *      colour and brightness - a near-clipped frame is faded toward the shortest
 *      exposure's radiance (scaled by how much brighter that reference is) so it
 *      does not leak a capped under-estimate that desaturates the highlight into
 *      magenta and halves its value (regression guard for the deghost=0 magenta fix).
 *
 * The merge is deterministic (the per-pixel loops carry no cross-pixel reduction),
 * so the numeric thresholds hold regardless of the OpenMP thread count.
 */

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cmocka.h>

#include "common/hdrmerge.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

#define TW 128
#define TH 128
#define NF 3

static float _h2f(const dt_hdrmerge_half_t h) { return dt_hdrmerge_half_to_float(h); }
static dt_hdrmerge_half_t _f2h(const float f) { return dt_hdrmerge_float_to_half(f); }

// RGGB phase: (row%2,col%2) -> (0,0)=R, (1,1)=B, else G. The scene is coloured so
// the de-interleave (one pyramid per Bayer position) is genuinely exercised.
static int _site_ch(const int x, const int y)
{
  const int r = y & 1, c = x & 1;
  return (r == 0 && c == 0) ? 0 : ((r == 1 && c == 1) ? 2 : 1);
}

// Shared brightness proxy = 2x2 mosaic-block maximum (matches _hdr_compute_luma),
// so all four CFA channels of a quad get the same per-frame weight.
static void _make_luma(const dt_hdrmerge_half_t *const fr, dt_hdrmerge_half_t *const lu)
{
  for(int y = 0; y < TH; y++)
    for(int x = 0; x < TW; x++)
    {
      const int xx = x & ~1, yy = y & ~1;
      const int x1 = (xx + 1 < TW) ? xx + 1 : TW - 1;
      const int y1 = (yy + 1 < TH) ? yy + 1 : TH - 1;
      float m = _h2f(fr[xx + (size_t)TW * yy]);
      m = fmaxf(m, _h2f(fr[x1 + (size_t)TW * yy]));
      m = fmaxf(m, _h2f(fr[xx + (size_t)TW * y1]));
      m = fmaxf(m, _h2f(fr[x1 + (size_t)TW * y1]));
      lu[x + (size_t)TW * y] = _f2h(m);
    }
}

// Fill a bracket from a per-pixel scene radiance E(ch,x,y): each frame stores the
// clipped normalized sample X = min(E / cal, 1). cal = {1, 4, 16} (longest first).
typedef float (*scene_fn)(int ch, int x, int y);
static const float CAL[NF] = { 1.0f, 4.0f, 16.0f };
static const float CAL_MAX = 16.0f;

static void _build_bracket(scene_fn scene, dt_hdrmerge_half_t *f[NF], dt_hdrmerge_half_t *l[NF])
{
  for(int i = 0; i < NF; i++)
  {
    for(int y = 0; y < TH; y++)
      for(int x = 0; x < TW; x++)
      {
        const float E = scene(_site_ch(x, y), x, y);
        float X = E / CAL[i];
        if(X > 1.0f) X = 1.0f;
        f[i][x + (size_t)TW * y] = _f2h(X);
      }
    _make_luma(f[i], l[i]);
  }
}

static dt_hdrmerge_t _make_job(const dt_hdrmerge_half_t *const frames[NF],
                               const dt_hdrmerge_half_t *const luma[NF],
                               float *out, gboolean pyramid, float feather)
{
  dt_hdrmerge_t h = { .width = TW, .height = TH, .num_frames = NF,
                      .frames = frames, .luma = luma, .cal = CAL,
                      .white_thresh = DT_HDRMERGE_DEFAULT_WHITE_THRESH,
                      .weight = DT_HDRMERGE_WEIGHT_EXPONENTIAL,
                      .deghost_threshold = 0.0f,
                      .pyramid = pyramid, .xtrans = FALSE, .feather = feather,
                      .out = out };
  return h;
}

// ---- scenes -----------------------------------------------------------------

// A smooth, coloured, everywhere-unclipped field (longest exposure X <= 0.42).
static const float STATIC_CF[3] = { 0.30f, 0.42f, 0.16f };
static float _scene_static(int ch, int x, int y)
{
  const float g = 0.6f + 0.4f * sinf(x * 0.03f) * cosf(y * 0.021f); // ~[0.2, 1.0]
  return STATIC_CF[ch] * g;
}

// A large, hugely over-exposed core that clips every channel in every frame.
static float _scene_blown(int ch, int x, int y)
{
  const int core = (x >= 32 && x < 96 && y >= 32 && y < 96);
  return core ? 200.0f : 0.3f * STATIC_CF[ch];
}

// A bright green-dominant horizontal bar: clips R/G in the long & mid exposures
// but is fully captured by the short one; the surround is a dim version of it.
static const float BAR_CH[3] = { 0.85f, 1.0f, 0.35f };
static float _scene_clipbar(int ch, int x, int y)
{
  (void)x;                                   // horizontal bar: depends on y only
  const int bar = (y >= 48 && y < 80);
  return (bar ? 10.0f : 0.4f) * BAR_CH[ch];
}

// ---- helpers ----------------------------------------------------------------

static void _alloc_all(dt_hdrmerge_half_t *f[NF], dt_hdrmerge_half_t *l[NF], float **out)
{
  for(int i = 0; i < NF; i++)
  {
    f[i] = malloc((size_t)TW * TH * sizeof(dt_hdrmerge_half_t));
    l[i] = malloc((size_t)TW * TH * sizeof(dt_hdrmerge_half_t));
    assert_non_null(f[i]);
    assert_non_null(l[i]);
  }
  *out = malloc((size_t)TW * TH * sizeof(float));
  assert_non_null(*out);
}

static void _free_all(dt_hdrmerge_half_t *f[NF], dt_hdrmerge_half_t *l[NF], float *out)
{
  for(int i = 0; i < NF; i++) { free(f[i]); free(l[i]); }
  free(out);
}

// ---- tests ------------------------------------------------------------------

// A frame-identical scene must reconstruct out = E / cal_max on every CFA channel:
// energy preservation of the multi-scale (pyramid) blend.
static void test_pyramid_reconstructs_static(void **state)
{
  (void)state;
  dt_hdrmerge_half_t *f[NF], *l[NF]; float *out;
  _alloc_all(f, l, &out);
  _build_bracket(_scene_static, f, l);
  const dt_hdrmerge_half_t *frames[NF] = { f[0], f[1], f[2] };
  const dt_hdrmerge_half_t *luma[NF] = { l[0], l[1], l[2] };

  dt_hdrmerge_t h = _make_job(frames, luma, out, TRUE, 0.5f);
  dt_hdrmerge_process_cpu(&h);

  double maxrel = 0.0;
  for(int y = 16; y < TH - 16; y++)          // interior: avoid the clamp border
    for(int x = 16; x < TW - 16; x++)
    {
      const float ref = _scene_static(_site_ch(x, y), x, y) / CAL_MAX;
      const double rel = fabs(out[x + (size_t)TW * y] - ref) / ref;
      if(rel > maxrel) maxrel = rel;
    }
  assert_true(maxrel < 0.01);                // measured ~0.001
  _free_all(f, l, out);
}

// The single-scale (linear) path must recover the same radiance.
static void test_linear_reconstructs_static(void **state)
{
  (void)state;
  dt_hdrmerge_half_t *f[NF], *l[NF]; float *out;
  _alloc_all(f, l, &out);
  _build_bracket(_scene_static, f, l);
  const dt_hdrmerge_half_t *frames[NF] = { f[0], f[1], f[2] };
  const dt_hdrmerge_half_t *luma[NF] = { l[0], l[1], l[2] };

  dt_hdrmerge_t h = _make_job(frames, luma, out, FALSE, 0.0f);
  dt_hdrmerge_process_cpu(&h);

  double maxrel = 0.0;
  for(int y = 16; y < TH - 16; y++)
    for(int x = 16; x < TW - 16; x++)
    {
      const float ref = _scene_static(_site_ch(x, y), x, y) / CAL_MAX;
      const double rel = fabs(out[x + (size_t)TW * y] - ref) / ref;
      if(rel > maxrel) maxrel = rel;
    }
  assert_true(maxrel < 0.01);
  _free_all(f, l, out);
}

// An unrecoverable highlight (every channel clips in every frame) must neutralize
// to the white point, not a magenta channel-clip mismatch.
static void test_pyramid_neutralizes_blown_core(void **state)
{
  (void)state;
  dt_hdrmerge_half_t *f[NF], *l[NF]; float *out;
  _alloc_all(f, l, &out);
  _build_bracket(_scene_blown, f, l);
  const dt_hdrmerge_half_t *frames[NF] = { f[0], f[1], f[2] };
  const dt_hdrmerge_half_t *luma[NF] = { l[0], l[1], l[2] };

  dt_hdrmerge_t h = _make_job(frames, luma, out, TRUE, 0.5f);
  dt_hdrmerge_process_cpu(&h);

  for(int y = 48; y < 80; y++)               // well inside the blown core
    for(int x = 48; x < 80; x++)
    {
      const float o = out[x + (size_t)TW * y];
      assert_true(o > 0.999f && o < 1.001f);
    }
  _free_all(f, l, out);
}

// A bright coloured patch that clips only in the longer exposures must keep its
// colour AND its brightness: a near-clipped frame is faded toward the shortest
// exposure's radiance (scaled by how much brighter that reference is), so it does
// not leak a capped under-estimate that halves the value and desaturates it toward
// magenta. Regression guard for the deghost=0 magenta fix.
static void test_pyramid_clipped_channel_keeps_colour(void **state)
{
  (void)state;
  dt_hdrmerge_half_t *f[NF], *l[NF]; float *out;
  _alloc_all(f, l, &out);
  _build_bracket(_scene_clipbar, f, l);
  const dt_hdrmerge_half_t *frames[NF] = { f[0], f[1], f[2] };
  const dt_hdrmerge_half_t *luma[NF] = { l[0], l[1], l[2] };

  dt_hdrmerge_t h = _make_job(frames, luma, out, TRUE, 1.0f);
  dt_hdrmerge_process_cpu(&h);

  // average each CFA channel across the bar interior, and the short-exposure
  // reference it should track (out = E / cal_max, recovered from the short frame).
  double s[3] = { 0, 0, 0 }, sr[3] = { 0, 0, 0 };
  int cnt[3] = { 0, 0, 0 };
  for(int y = 60; y < 68; y++)
    for(int x = 24; x < 104; x++)
    {
      const int ch = _site_ch(x, y);
      s[ch] += out[x + (size_t)TW * y];
      sr[ch] += 10.0f * BAR_CH[ch] / CAL_MAX;
      cnt[ch]++;
    }
  const float R = s[0] / cnt[0], G = s[1] / cnt[1], B = s[2] / cnt[2];
  const float rR = sr[0] / cnt[0], rG = sr[1] / cnt[1], rB = sr[2] / cnt[2];

  // chroma (channel ratios) must track the short-exposure reference (measured
  // ~0.06; the pre-pin magenta bug was ~0.18).
  const double cdev = fmax(fabs(R / G - rR / rG), fabs(B / G - rB / rG));
  assert_true(cdev < 0.13);
  // green stays dominant (a magenta result pushes R and B up past G).
  assert_true(G > R && R > B);
  // brightness restored to ~the reference: the raw coarse-leak halved it (G ~ 0.33
  // against a reference rG of 0.625; the fade-to-reference recovers G ~ 0.60).
  assert_true(G > 0.9f * rG);
  _free_all(f, l, out);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_pyramid_reconstructs_static),
    cmocka_unit_test(test_linear_reconstructs_static),
    cmocka_unit_test(test_pyramid_neutralizes_blown_core),
    cmocka_unit_test(test_pyramid_clipped_channel_keeps_colour),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
