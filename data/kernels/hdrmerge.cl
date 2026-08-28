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

#include "common.h"

// hdrmerge high dynamic range merge, two-pass per-pixel scheme.
// Mirrors the CPU reference in src/common/hdrmerge.c and the vkdt port
// (https://github.com/hanatos/vkdt/pull/269) of Wenzel Jakob's hdrmerge.
//
// Frames are stored as fp16 and read with vload_half (core OpenCL, no fp16
// extension required) to halve the bracket's memory footprint.

// Weight envelope. weight_type 1 = triangular (vkdt), else exponential
// (Jakob/hdrmerge). Keep in sync with _hdrmerge_weight() in src/common/hdrmerge.c.
static inline float
hdrmerge_weight(const float s, const int weight_type)
{
  if(s <= 0.0f || s >= 1.0f)
    return 0.0f;
  if(weight_type == 1) // triangular
    return 1.0f - fabs(2.0f * s - 1.0f);
  const float alpha = -0.1f;
  const float beta = 1.4918246976412703f; // 1 / exp(4 * alpha)
  // native_exp: a weight only needs a few digits, so the fast hardware
  // approximation is plenty and noticeably cheaper than full-precision exp.
  return beta * native_exp(alpha * (1.0f / s + 1.0f / (1.0f - s)));
}

// De-ghosting fall-off. dev = how far a frame's luminance sits from the
// consensus, span = deghost * L_ref the deviation at which it is fully rejected.
// 1 at perfect agreement, smooth (Hermite) ramp to 0 at the threshold, so the
// rejected region is feathered instead of a hard binary cut (which demosaics
// into blocky, often magenta, edges). Keep in sync with _hdrmerge_deghost_falloff()
// in src/common/hdrmerge.c.
static inline float
hdrmerge_deghost_falloff(const float dev, const float span)
{
  if(span <= 0.0f)
    return 1.0f;
  const float t = dev < span ? dev / span : 1.0f;
  return 1.0f - t * t * (3.0f - 2.0f * t); // 1 - smoothstep(0, 1, t)
}

// Smooth clip exclusion: 1 for a well-exposed sample, ramping continuously to 0
// toward the white point (0 at/above it). Replaces a hard per-pixel clip gate
// whose binary in/out pixelated bright motion borders; still reaches 0 at white
// so a clipped sample cannot leak its value in. Keep in sync with
// _hdrmerge_clip_weight() in src/common/hdrmerge.c.
#define HDRMERGE_CLIP_EDGE 0.7f
static inline float
hdrmerge_clip_weight(const float L, const float white)
{
  if(L <= 0.0f || L >= white)
    return 0.0f;
  const float s = L / white;
  if(s <= HDRMERGE_CLIP_EDGE)
    return 1.0f;
  const float t = (s - HDRMERGE_CLIP_EDGE) / (1.0f - HDRMERGE_CLIP_EDGE);
  return 1.0f - t * t * (3.0f - 2.0f * t); // smoothstep 1 -> 0 toward white
}

// Robustness cap for the de-ghost median consensus (private array of per-frame
// luminance radiances). Beyond it the de-ghost falls back to the mean.
#define HDRMERGE_MAX_FRAMES 16

// Median of n (<= HDRMERGE_MAX_FRAMES) values; sorts a in place. The de-ghost
// consensus: unlike the weighted mean it is not dragged toward a moving object
// seen in a minority of frames, so the mover stands out and is faded cleanly.
// Keep in sync with _hdrmerge_median() in src/common/hdrmerge.c.
static inline float
hdrmerge_median(float *a, const int n)
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

// frames:  num_frames planes of width*height fp16 samples, CFA post-rawprepare.
// luma:    companion planes (fp16) of a per-position brightness proxy shared by
//          all CFA channels; the weights are derived from this so neighbouring
//          R/G/B stay consistent and do not false-colour on motion. The host
//          always supplies it (it falls back to the CPU when no proxy exists).
// cal:     num_frames calibration factors, scene radiance = pixel * cal.
// out:     width*height, normalized so 1.0 = brightest representable radiance.
kernel void
hdrmerge_merge(global const half *const frames,
               global const half *const luma,
               global const half *const bluma,
               global const float *const cal,
               global float *const out,
               const int width,
               const int height,
               const int num_frames,
               const float white_thresh,
               const float inv_cal_max,
               const float eps,
               const int weight_type,
               const float deghost,
               const int ref_frame)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height)
    return;

  // size_t offsets: a packed multi-frame buffer easily exceeds the 24-bit
  // range of mad24/mul24 at high resolution.
  const size_t idx = (size_t)y * (size_t)width + (size_t)x;
  const size_t plane = (size_t)width * (size_t)height;

  // pass 1: weight each frame by the shared luminance proxy, so every CFA
  // channel at this position gets the same per-frame weight (no chroma break-up
  // on motion). The merged value is still the channel's own radiance X * cal.
  float num = 0.0f;  // sum w * E
  float den = 0.0f;  // sum w
  float numf = 0.0f; // sum E         (unweighted fallback)
  float lnum = 0.0f; // sum w * L_rad
  float lnumf = 0.0f; // sum L_rad    (unweighted fallback)
  float min_bright = MAXFLOAT; // min over frames of the shared brightness
  for(int i = 0; i < num_frames; i++)
  {
    const size_t o = (size_t)i * plane + idx;
    const float c = cal[i];
    const float X = vload_half(o, frames);
    const float Ls = vload_half(o, luma);   // sharp
    const float Lw = vload_half(o, bluma);  // blurred (drives the weights)
    const float E = X * c;
    // exposure weight from the BLURRED luma (spatial coherence) times a SMOOTH
    // clip weight on the SHARP luma (excludes clipped/black without a hard
    // per-pixel step -> no pixelated borders).
    const float w = hdrmerge_clip_weight(Ls, white_thresh)
                    * hdrmerge_weight(Lw / white_thresh, weight_type);
    num += w * E;
    den += w;
    numf += E;
    lnum += w * (Lw * c);
    lnumf += Lw * c;
    min_bright = fmin(min_bright, Ls); // sharp: drives the clip neutralization
  }
  const float E_ref = (den > eps) ? (num / den) : (numf / (float)num_frames);
  const float L_ref = (den > eps) ? (lnum / den) : (lnumf / (float)num_frames);

  // De-ghosting consensus: when on, anchor it to ONE globally chosen well-exposed
  // reference frame (ref_frame, picked on the host). A pixel whose content moved
  // is resolved to the reference's view, which is spatially coherent (one frame
  // -> no fragmented ghosts) and self-consistent across channels (one frame -> no
  // false colour); frames disagreeing with it are faded out below. Where the
  // reference did not resolve this pixel (clipped/black), fall back to the median
  // over the frames that did. With de-ghosting off this stays the weighted mean.
  float L_cons = L_ref;
  if(deghost > 0.0f)
  {
    // validity from the SHARP luma, value from the BLURRED luma (smooth consensus)
    const float Ls_ref = vload_half((size_t)ref_frame * plane + idx, luma);
    const float Lw_ref = vload_half((size_t)ref_frame * plane + idx, bluma);
    if(Ls_ref > 0.0f && Ls_ref < white_thresh)
      L_cons = Lw_ref * cal[ref_frame]; // anchor to the reference frame
    else if(num_frames <= HDRMERGE_MAX_FRAMES) // reference unusable here -> median
    {
      float lr[HDRMERGE_MAX_FRAMES];
      int nlr = 0;
      for(int i = 0; i < num_frames; i++)
      {
        const float Ls = vload_half((size_t)i * plane + idx, luma);
        if(Ls > 0.0f && Ls < white_thresh) // not black, not clipped (sharp)
          lr[nlr++] = vload_half((size_t)i * plane + idx, bluma) * cal[i];
      }
      if(nlr >= 3)
        L_cons = hdrmerge_median(lr, nlr);
    }
  }

  // pass 2: re-weight by the predicted shared luminance, and optionally fade out
  // frames whose luminance disagrees with the consensus (de-ghosting). All
  // luminance-driven, hence identical across colour channels.
  num = 0.0f;
  den = 0.0f;
  for(int i = 0; i < num_frames; i++)
  {
    const size_t o = (size_t)i * plane + idx;
    const float c = cal[i];
    const float Ls = vload_half(o, luma); // sharp
    // Smooth clip exclusion (SHARP luma): pass 2 weights by the consensus-
    // predicted value and never re-checks the frame's own clip state, so without
    // this a clipped moving highlight predicted mid-range would leak its clipped
    // value in -> magenta. The smooth ramp (vs a hard gate) keeps bright motion
    // borders from demosaicing into pixelated edges.
    const float cg = hdrmerge_clip_weight(Ls, white_thresh);
    const float predicted = L_cons / c; // expected luminance signal in frame i
    float w = cg * hdrmerge_weight(predicted / white_thresh, weight_type);
    if(w > 0.0f && deghost > 0.0f)
    {
      // fade the frame out smoothly as its luminance leaves the consensus,
      // measured on the BLURRED luma so the de-ghost mask has feathered edges.
      const float Lw = vload_half(o, bluma);
      w *= hdrmerge_deghost_falloff(fabs(Lw * c - L_cons), deghost * L_cons);
    }
    if(w > 0.0f)
    {
      const float X = vload_half(o, frames);
      num += w * (X * c);
      den += w;
    }
  }
  const float E = (den > eps) ? (num / den) : E_ref;

  // Unrecoverable highlight: the brightest channel clips in every frame, so its
  // clipped under-estimate next to the unclipped (lower) channels demosaics into
  // magenta after white balance. Pin the pixel to the white point so darktable's
  // highlight handling treats it as uniformly clipped (neutral). Otherwise
  // normalize to the convention 1.0 = saturation of the shortest exposure.
  out[idx] = (min_bright >= white_thresh) ? 1.0f : fmax(0.0f, E * inv_cal_max);
}
