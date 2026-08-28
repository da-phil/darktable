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

#pragma once

#include <glib.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// hdrmerge-style high dynamic range merge of an exposure bracket.
//
// This implements the merge algorithm of Wenzel Jakob's hdrmerge
// (https://github.com/wjakob/hdrmerge), structured as the efficient
// two-pass per-pixel scheme used by the vkdt port
// (https://github.com/hanatos/vkdt/pull/269).
//
// Each input is a single channel CFA frame as produced by the rawprepare
// module: black point subtracted and rescaled so that 1.0 is the sensor
// saturation level. The frames must be perfectly aligned and of identical
// geometry. The result is written normalized so that 1.0 corresponds to the
// brightest radiance that any frame of the bracket can represent (the
// saturation level of the shortest exposure), matching the convention of
// darktable's previous HDR merge so the downstream raw pipeline is unchanged.
//
// Frames are stored as IEEE half floats (fp16) to halve the memory footprint
// of holding the whole bracket at once; the ~0.1% relative quantization is far
// below sensor noise for post-rawprepare values in [0, ~1].

// IEEE 754 binary16 bit pattern. Convert with the helpers below.
typedef uint16_t dt_hdrmerge_half_t;

// Weight envelope selection. Both reject under- and over-exposed samples; the
// exponential (Jakob/hdrmerge) is far more peaked at the mid value and so
// selects the best-exposed frame more decisively (fewer blend artefacts),
// while the triangular (vkdt) is cheaper and softer.
typedef enum dt_hdrmerge_weight_t
{
  DT_HDRMERGE_WEIGHT_EXPONENTIAL = 0, // beta * exp(alpha*(1/s + 1/(1-s)))
  DT_HDRMERGE_WEIGHT_TRIANGULAR = 1,  // 1 - |2s - 1|
} dt_hdrmerge_weight_t;

// Exponential envelope (Jakob/hdrmerge). Keep in sync with the copy in
// data/kernels/hdrmerge.cl. The weight peaks at the mid value (s = 0.5) and
// falls off rapidly towards black (s -> 0) and saturation (s -> 1).
#define DT_HDRMERGE_ALPHA (-0.1f)
// beta = 1 / exp(4 * alpha), normalizes the envelope to 1.0 at s = 0.5
#define DT_HDRMERGE_BETA (1.4918246976412703f)
// default saturation safety margin, mirrors the previous algorithm's
// 3000/UINT16_MAX head room for upsampling and 16-bit quantization/dithering
#define DT_HDRMERGE_DEFAULT_WHITE_THRESH (1.0f - 3000.0f / 65535.0f)

typedef struct dt_hdrmerge_t
{
  int width;
  int height;
  int num_frames;
  // num_frames buffers, each width*height fp16 samples, single channel CFA,
  // post-rawprepare (0 = black, 1 = saturation)
  const dt_hdrmerge_half_t *const *frames;
  // per-frame brightness proxy (fp16): a signal shared by every CFA colour
  // channel at a position - the 2x2 mosaic-block MAXIMUM (the brightest,
  // first-to-clip channel). The per-frame weights are derived from this, NOT
  // from each channel's own value, so neighbouring R/G/B are merged with
  // identical weights and cannot break up into false colour where content moves
  // (waves, foliage). Being the max, it also flags saturation: a pixel whose
  // brightest channel clips in every frame is an unrecoverable highlight and is
  // neutralized to the white point (avoids magenta around the sun / on specular
  // crests). If NULL, weights fall back to each channel's own value (legacy,
  // false-colours on motion; no highlight neutralization).
  const dt_hdrmerge_half_t *const *luma;
  // num_frames calibration factors: scene radiance = pixel * cal.
  // cal = 100 / (aperture * exposure_time * iso), as in the previous merge.
  const float *cal;
  // normalized saturation threshold in [0,1]; samples at or above this are
  // treated as clipped. Use DT_HDRMERGE_DEFAULT_WHITE_THRESH if unsure.
  float white_thresh;
  // weight envelope to use.
  dt_hdrmerge_weight_t weight;
  // de-ghosting: 0 disables it; > 0 rejects, in pass 2, any frame whose
  // luminance deviates from the consensus by more than this fraction
  // (e.g. 0.5 = 50%), suppressing content that moved between frames.
  float deghost_threshold;
  // multi-scale (Laplacian-pyramid) CFA blend. When TRUE - and a luma proxy is
  // present and the CFA is a 2x2 Bayer mosaic (xtrans == FALSE) - the bracket is
  // merged band by band with a Laplacian pyramid: every frequency scale is
  // cross-faded with the per-frame weights smoothed to the MATCHING scale, so
  // exposure / motion transitions blend gently at coarse scales (no quad-grid
  // demosaic fringing -> no magenta/green) while fine detail is preserved (no
  // halos). Falls back to the single per-pixel weighted average when FALSE, for
  // X-Trans, or when no luma proxy is supplied. CPU only for now.
  gboolean pyramid;
  // TRUE if the CFA is X-Trans (filters == 9u). Its 6x6 pattern is not the 2x2
  // quad the pyramid de-interleave assumes, so an X-Trans bracket always uses
  // the single-scale path regardless of 'pyramid'. Ignored when pyramid == FALSE.
  gboolean xtrans;
  // feathering of the multi-scale blend, clamped to [0,1] (0 = crispest). Blurs
  // the per-frame weight (in the mid-tones only) to soften exposure/motion
  // transitions. It does NOT widen the highlight-neutralization border - that is
  // feathered independently and kept narrow, because widening it bleeds the
  // neutral white core into the coloured corona (magenta). Only the pyramid path.
  float feather;
  // width*height result, allocated by the caller. Normalized so that 1.0 is
  // the brightest radiance representable by the bracket.
  float *out;
} dt_hdrmerge_t;

// Merge h->frames into h->out. Tries OpenCL first and transparently falls
// back to a multi-threaded CPU implementation, so this always produces a
// result. Returns TRUE if the OpenCL path was used, FALSE if the CPU path ran.
gboolean dt_hdrmerge_process(dt_hdrmerge_t *const h);

// Reference multi-threaded CPU implementation, exposed for testing and used
// as the fallback. Always succeeds.
void dt_hdrmerge_process_cpu(dt_hdrmerge_t *const h);

// --- fp16 <-> float conversion (portable, branch-correct) -----------------

static inline dt_hdrmerge_half_t dt_hdrmerge_float_to_half(const float f)
{
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 31) & 1u;
  const uint32_t exp = (x >> 23) & 0xFFu;
  const uint32_t mant = x & 0x7FFFFFu;

  if(exp == 0) // zero or float32 denormal (underflows to half zero)
    return (dt_hdrmerge_half_t)(sign << 15);
  if(exp == 255) // inf / nan
    return (dt_hdrmerge_half_t)((sign << 15) | 0x7C00u | (mant ? 1u : 0u));

  const int new_exp = (int)exp - 127 + 15;
  if(new_exp >= 31) // overflow to inf
    return (dt_hdrmerge_half_t)((sign << 15) | 0x7C00u);
  if(new_exp <= 0) // half denormal or underflow
  {
    const int shift = 1 - new_exp;
    if(shift > 24)
      return (dt_hdrmerge_half_t)(sign << 15);
    const uint32_t full_mant = (1u << 23) | mant;
    const uint16_t half_mant = (uint16_t)(full_mant >> (13 + shift));
    return (dt_hdrmerge_half_t)((sign << 15) | half_mant);
  }
  return (dt_hdrmerge_half_t)((sign << 15) | ((uint32_t)new_exp << 10) | (mant >> 13));
}

static inline float dt_hdrmerge_half_to_float(const dt_hdrmerge_half_t h)
{
  // bit trick from Fabian Giesen (https://gist.github.com/rygorous/2156668),
  // as also used in src/imageio/imageio_tiff.c; handles denormals correctly.
  uint32_t o = (uint32_t)(h & 0x7fffu) << 13; // exponent/mantissa bits
  const uint32_t exp = 0x0f800000u & o;       // (0x7c00 << 13): just the exponent
  o += (uint32_t)(127 - 15) << 23;            // exponent adjust
  if(exp == 0x0f800000u) // inf / nan
  {
    o += (uint32_t)(128 - 16) << 23; // extra exp adjust
  }
  else if(exp == 0) // zero / denormal
  {
    o += 1u << 23; // extra exp adjust
    float of;
    memcpy(&of, &o, sizeof(of));
    uint32_t magic_u = 113u << 23;
    float magic_f;
    memcpy(&magic_f, &magic_u, sizeof(magic_f));
    of -= magic_f; // renormalize
    memcpy(&o, &of, sizeof(o));
  }
  o |= (uint32_t)(h & 0x8000u) << 16; // sign bit

  float f;
  memcpy(&f, &o, sizeof(f));
  return f;
}

#ifdef __cplusplus
}
#endif
