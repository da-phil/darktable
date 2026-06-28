# HDR Merge — hdrmerge algorithm (CPU + OpenCL)

Design notes and progress for replacing darktable's exposure-bracket HDR merge
with the *hdrmerge* algorithm, implemented as efficient CPU and OpenCL paths.

- **Feature branch:** `claude/darktable-hdr-merge-algorithm-mn8uow`
- **Status:** implementation complete, validated numerically and by isolated
  compilation (full darktable build not available in the dev container — see
  [Validation](#validation)).
- **Entry point:** *lighttable → selected bracket → merge HDR*, which runs
  `dt_control_merge_hdr()` → `_control_merge_hdr_job_run()` in
  `src/control/jobs/control_jobs.c`.

---

## 1. Motivation

darktable's previous HDR merge is a **single weighted average** of all
exposures. For every output pixel it accumulates

```
pixels[p] += w · value · cal      weight[p] += w
```

over all frames and finally divides. The weight `w` is a *photon-count* term
times a smooth *envelope* of the **3×3 CFA block maximum** around the pixel,
with special cases for saturated and previously-blown pixels.

Two properties of that scheme cause the artefacts reported on detailed,
slightly-non-static scenes (foliage, ripples on water) **even when the frames
are perfectly aligned**:

1. **Every non-clipped exposure contributes to every pixel.** Where content
   differs slightly between frames (a leaf in one frame, a gap in the next),
   the average blends both → smeared / ghost-like texture.
2. **The saturation envelope is computed from a 3×3 spatial block**, so a
   pixel's weight depends on its neighbours. In high-frequency texture the
   block maximum is noisy and inconsistent, modulating the blend and adding
   structured artefacts.

Other tools (`enfuse`, Wenzel Jakob's
[`hdrmerge`](https://github.com/wjakob/hdrmerge)) use a cleaner per-pixel
radiometric estimator. The Vulkan port of darktable (vkdt) recently adopted the
hdrmerge approach as a compute shader
([hanatos/vkdt#269](https://github.com/hanatos/vkdt/pull/269)). This change
brings the same algorithm to darktable's CPU and OpenCL code.

> **Scope / honest caveat.** This algorithm removes the previous *blend/envelope*
> artefacts, the **CFA false colour** on changing content (waves, foliage —
> [§3.2](#32-luminance-guided-weighting-avoids-cfa-false-colour)) and the
> **magenta blown highlights** around the sun / on specular crests
> ([§3.4](#34-blown-highlight-neutralization-magenta-sun--specular-crests)), and
> the **harsh per-pixel transitions / quad-grid fringing** on moving content via
> the multi-scale Laplacian-pyramid blend
> ([§3.8](#38-multi-scale-laplacian-pyramid-blend), Bayer). It
> still assumes well-aligned frames. The optional **reference-frame de-ghost**
> ([§3.6](#36-runtime-options)) resolves moving content coherently to one
> best-exposed frame (no fragmented ghosts, no motion false colour), at the cost
> of that frame's exposure in the moving region; a *user-selectable* reference
> and optical-flow registration for whole-bracket motion are future work
> ([§9](#9-limitations--future-work)).

---

## 2. The previous algorithm (recap)

`_control_merge_hdr_process()` is invoked once per frame by the export
pipeline (stopping after `rawprepare`, so the buffer is single-channel CFA with
the black point subtracted and the white point rescaled to `1.0`). Per pixel:

- `cal = 100 / (aperture · exposure · iso)` — scales the raw value to a common
  radiance unit.
- `photoncnt = 100 · aperture · exposure / iso` — base weight.
- `w = photoncnt · (epsw + envelope(blockMax + offset))`, where `envelope()`
  is a smooth hat over the **3×3 block** maximum and `offset = 3000/65535` is a
  saturation safety margin.
- Saturated pixels (`blockMax + offset ≥ 1`) are only used if nothing better
  has been written; everything else is accumulated into the running weighted
  sum.

A final pass normalizes `pixels[p] / (whitelevel · weight[p])` with
`whitelevel = max_i cal_i`, then the result is written as a floating-point DNG
and re-imported.

---

## 3. The hdrmerge algorithm

The merge is a **per-pixel, two-pass radiometric estimate**. Notation, per
frame *i*:

- `Xᵢ` — measured CFA sample, normalized so `0 = black`, `1 = saturation`
  (exactly what `rawprepare` outputs).
- `calᵢ = 100 / (aperture · exposure · iso)` — same calibration as before;
  scene radiance `≈ Xᵢ · calᵢ`.
- `Eᵢ = Xᵢ · calᵢ` — this frame's estimate of the scene radiance.

### 3.1 Weight envelope

The weight is Jakob's smooth envelope of a normalized brightness `s`:

```
w(s) = β · exp( α · (1/s + 1/(1−s)) ),   α = −0.1,  β = 1/exp(4α) ≈ 1.49182
w(s) = 0  for s ≤ 0 or s ≥ 1
```

`β` normalizes the peak to `w(0.5) = 1`. The envelope is sharply peaked at the
mid-tone and decays very fast towards both ends:

| s    | 0.02   | 0.1   | 0.3   | 0.5  | 0.7   | 0.9   | 0.98   |
|------|--------|-------|-------|------|-------|-------|--------|
| w(s) | 0.0091 | 0.491 | 0.928 | 1.00 | 0.928 | 0.491 | 0.0091 |

So noisy shadow samples and near-clipped highlight samples contribute almost
nothing; the best-exposed frame dominates. A saturation safety margin is applied
by scaling the argument by `white_thresh = 1 − 3000/65535 ≈ 0.9542` (same head
room as the old `offset`), so brightness at/above it counts as clipped.

### 3.2 Luminance-guided weighting (avoids CFA false colour)

**`s` is a shared per-position luminance `L`, not each channel's own value.**
This is essential and was the subject of a real bug fix. The merge runs on the
**mosaiced CFA**: each pixel is a single colour (R, G *or* B). If the weight
came from a pixel's own value, then where content moves between frames (waves,
foliage) neighbouring R/G/B could be sourced from *different* frames — and,
because white balance drives green towards saturation first, green routinely
drops the very frame red/blue keep. Demosaicing mismatched channels produces
vivid **magenta / green / purple** false colour.

The fix: derive every frame's weight from a brightness proxy `Lᵢ` that is
**shared by all CFA channels at a position** — the 2×2 mosaic-block **maximum**
(all four pixels of a Bayer quad get the same `L`). Neighbouring R/G/B are then
merged with *identical* per-frame weights, so they always come from the same
frame mixture and cannot break into false colour. The value being merged is
still each channel's own radiance `Eᵢ = Xᵢ·calᵢ`.

The block **maximum** (not the average) is deliberate: it tracks the brightest,
**first-to-clip** channel — green, on most sensors — which makes the very same
signal also the saturation detector used in [§3.4](#34-blown-highlight-neutralization).

A worked moving-water example (neutral grey, 3-stop bracket, a wave crest
clipping green in the long exposure): per-channel weighting yields `R/G = 0.51`
(a cyan cast); luminance-guided yields `R/G = 0.60` — neutral, as it should be.

### 3.3 Two passes

`Lᵢ` is the shared luminance of frame *i*; `Eᵢ = Xᵢ·calᵢ` the channel radiance.

```
# pass 1 — weight by the shared measured luminance
num  = Σᵢ w(Lᵢ/wt)·Eᵢ ;        den = Σᵢ w(Lᵢ/wt) ;   numf = Σᵢ Eᵢ
lnum = Σᵢ w(Lᵢ/wt)·(Lᵢ·calᵢ) ;                       lnumf = Σᵢ Lᵢ·calᵢ
E_ref = den>ε ? num/den : numf/N      # per-channel radiance consensus
L_ref = den>ε ? lnum/den : lnumf/N    # shared luminance-radiance consensus

# pass 2 — re-weight by the predicted shared luminance from the consensus
num = Σᵢ w(predᵢ/wt)·Eᵢ ;  den = Σᵢ w(predᵢ/wt) ,   predᵢ = L_ref / calᵢ
E   = den>ε ? num/den : E_ref
```

Pass 2 is the robustness step from `hdrmerge`'s `hdr.cpp` and the vkdt shader,
but driven by the shared luminance so it stays consistent across channels:
*"given the consensus luminance, where should this frame sit in its exposure?"*.
Because `Lᵢ` and `L_ref` are quad-constant, the pass-2 weights are too — the
whole Bayer quad is merged from one frame mixture.

### 3.4 Blown-highlight neutralization (magenta sun / specular crests)

§3.2 keeps neighbouring channels *consistent*, but it does not by itself fix the
**magenta highlight** — and that took a second fix. Around the sun, and on
specular wave crests, the scene is brighter than green's clip point but red and
blue are still captured. Green is then pinned at the white point while red/blue
hold their true (lower) values; after white balance multiplies red and blue up,
they overshoot green → **magenta**. Per-channel consistency cannot help: the
channels genuinely clip at different scene levels.

The signal is already in hand. Because `Lᵢ` is the block **maximum**, the merge
can see, per pixel, the smallest brightness any frame reached:

```
min_bright = minᵢ Lᵢ                 # the least-exposed frame's brightest channel
out = (min_bright ≥ white_thresh) ? 1.0 : max(0, E · 1/max_i calᵢ)
```

If even the **shortest exposure** clips the brightest channel
(`min_bright ≥ white_thresh`) the pixel is *unrecoverable* — the bracket never
caught it unclipped. Pinning the whole pixel to the white point (`1.0`) makes it
a uniformly-clipped, neutral pixel, which is what darktable's highlight handling
expects (this is exactly what the previous algorithm did for fully-blown
pixels). A pixel that the shortest exposure *did* catch is **not** flagged, so
recoverable highlights keep their detail and colour. Worked example (3-stop
bracket): the sun's corona goes from `out = 0.66, 1.00, 0.84` (magenta) to
`1.00, 1.00, 1.00` (neutral); a recoverable highlight one stop down is untouched.

### 3.5 Output normalization

`E` is in radiance units. darktable's DNG pipeline expects the merged raw
normalized so that `1.0` is the brightest representable value. The shortest
exposure (largest `cal`) saturates at the brightest radiance, so:

```
out[p] = max(0, E · t_ref),   t_ref = 1 / max_i calᵢ
```

This is **algebraically identical** to the previous code's
`pixels / (whitelevel · weight)` normalization (`whitelevel = max_i calᵢ`), so
the downstream DNG writer, white balance, demosaic and highlight handling are
unchanged. A fully-clipped pixel (every frame saturated) resolves to `1.0`,
matching the previous behaviour.

> **vkdt difference:** the vkdt module scales to the *mean* exposure time with a
> user EV-push because it feeds a display pipeline. darktable instead targets
> the shortest exposure so the DNG keeps the full captured range with `1.0 =
> brightest`, which is what its raw pipeline expects.

**Deterministic, order-independent result.** Because the output is normalized to
the shortest exposure, that frame is also the *metadata reference*: the merged
DNG inherits **its** exif (exposure time / ISO / aperture and, crucially,
exposure-bias) and its file name. The merged *pixels* were always
order-independent (the normalization is a `max` over frames and the two passes
sum over all frames), but the exif used to be copied from whichever frame the
user selected *first*. An auto-exposure bracket records a different
exposure-bias per frame (e.g. `−2 / 0 / +2 EV`), and darktable's scene-referred
default exposure is `0.7 EV − exif_exposure_bias`, so inheriting an arbitrary
frame's bias made the **re-developed brightness swing by the whole bracket range
depending on selection order**. Keying the metadata to the shortest exposure
(`max_i calᵢ`, the same frame the radiance is normalized to) removes that: the
same bracket now yields the same DNG, developed at the same brightness, no
matter what order the frames were picked in. (White-balance coefficients and the
colour matrix still come from the first collected frame; for a real bracket
these are identical across frames.)

### 3.6 Runtime options

The merge has no dialog (it is a one-click lighttable action), so these knobs
are exposed as **config keys** — settable in *preferences → processing → HDR
merge* (next to the auto-align section) or directly in `darktablerc`. They are
read once per merge in
`_control_merge_hdr_job_run()`. The code's hard fallbacks are benign if an entry
is missing entirely: the string keys and `deghost_threshold` fall back to their
documented defaults, and `feather` falls back to `0` (crispest) rather than its
registered `0.5` — so the merge still runs even without the registered entries.

| Config key (`plugins/lighttable/hdrmerge/…`) | Type | Default | Effect |
|----------------------------------------------|------|---------|--------|
| `weight` | enum | `exponential` | Weight envelope: `exponential` (Jakob, decisive) or `triangular` (vkdt, softer): `w(s)=1−|2s−1|`. `legacy` instead runs the **original pre-branch merge** (photon-count-weighted average, order-dependent) for A/B comparison — it ignores `blend`/`feather`/`deghost_threshold`. |
| `deghost_threshold` | float | `0.0` | `0` = off. `>0` *feathers out*, per pixel in pass 2, any frame whose **luminance** deviates from the **reference frame**, reaching full rejection at this fraction (e.g. `0.4`), so moving content resolves to the reference's view (coherent, no fragments/false colour). **Off by default — set a positive value to enable de-ghosting at all.** |
| `blend` | enum | `pyramid` | `pyramid` = multi-scale Laplacian blend ([§3.8](#38-multi-scale-laplacian-pyramid-blend), Bayer, CPU); `linear` = single per-pixel weighted average (faster, GPU path, X-Trans fallback). |
| `feather` | float | `0.5` | Softness of the `pyramid` blend, `0..1`. Blurs the per-frame weight to feather exposure/motion transitions (mid-tones only; clipped highlights stay sharp). `0` = crispest; no effect on `linear`. |

Luminance-guided weighting ([§3.2](#32-luminance-guided-weighting-avoids-cfa-false-colour))
is **always on** — it is correctness, not a tunable, so there is no longer a
`channel_coupling` option (it was removed; the old key is ignored if present).

**fp16 frame storage.** Frames are always stored as IEEE half floats. Holding a
whole bracket at once is the cost of the two-pass estimator; fp16 halves it. The
quantization is ≤ 0.1 % relative over the full normal-half range (within ~14
stops of saturation) and exact at 0 / 0.5 / 1.0 — far below sensor noise for
post-rawprepare values in `[0, ~1]`. The OpenCL kernel reads samples with
`vload_half` (core OpenCL, no `cl_khr_fp16` extension needed); the CPU path uses
the branch-correct converters in `hdrmerge.h`. One extra fp16 plane per frame
holds the luminance proxy (the 2×2 mosaic-block maximum, see
[§3.2](#32-luminance-guided-weighting-avoids-cfa-false-colour)).

**De-ghosting** is a pass-2 *soft* fall-off against a **reference frame** (it is
**off by default**; `deghost_threshold = 0`).

*Why a reference frame.* The earlier attempts used a *per-pixel* consensus (the
weighted mean, then the median over the valid frames). Both share a fatal flaw
for visible motion: the consensus is computed independently at every pixel, so
the set of frames that "wins" varies from pixel to pixel. In a moving region
that makes the result **spatially incoherent** — neighbouring pixels resolve to
different frames and the moving content breaks into *fragmented, visible ghost
pieces* the moment de-ghosting is turned up. (The mean had the extra problem of
being dragged toward the mover; the median fixed that for a *minority* mover but
still tracks a *majority* mover, and neither is spatially coherent.)

*The fix.* Pick **one** global reference frame — the frame that resolves the most
pixels (`0 < L < white`), i.e. the best-exposed one (`_hdrmerge_pick_reference`,
sampled on a stride; ties break toward the shorter exposure for determinism).
Every pixel is then de-ghosted against **that same frame's** luminance radiance
`L_cons = L_ref_frame · cal_ref_frame`. Moving content therefore resolves to the
reference's view *everywhere* — one frame, so it is both **spatially coherent**
(no fragments) and **self-consistent across CFA channels** (no false colour).
Static regions still merge every frame normally (full dynamic range and noise
averaging), because there the reference and all other frames agree. Where the
reference itself did not resolve a pixel (clipped or black there) the consensus
falls back to the **median** over the frames that did (`DT_HDRMERGE_MAX_FRAMES =
16` cap, else the mean). The reference is picked once on the host and passed to
the kernel as `ref_frame`; the whole block is skipped when de-ghosting is off, so
the default path is unchanged.

*Soft fall-off.* As a frame's `Lᵢ·calᵢ` departs from `L_cons`, its weight is
multiplied by a smooth Hermite ramp `1 − smoothstep(0,1, dev/span)` with
`dev = |Lᵢ·calᵢ − L_cons|` and `span = threshold · L_cons` — `1` at agreement,
smoothly to `0` at the threshold. This replaced an earlier **hard binary veto**
(`w = 0` past the threshold), whose flip-in/flip-out made the rejected region
demosaic into **blocky, often magenta, edges**.

The trade-off is the usual one for reference-based de-ghosting: in a moving
region you get the reference frame's exposure and noise (no multi-frame DR or
averaging *there*). The reference is auto-picked (best-exposed); letting the user
*choose* it (which instance of the motion to keep) is the natural next step, as
is optical-flow registration for content that moves across the *whole* bracket.

### 3.7 Spatial coherence (clip-gate + blurred-luma weights)

The merge decides *which frames to source* per pixel (really per 2×2 quad,
since the luma is quad-shared). With quad-shared weights the per-quad result is a
convex blend of the frames, so it cannot manufacture false colour on its own —
but the *selection* is decided independently at each quad. Where it jumps between
neighbouring quads (moving content, noise near a threshold) it stamps a quad-grid
pattern into the merged CFA mosaic, and the **demosaicer turns that step into
magenta/green fringing**. The same incoherence, made coarser by a strong de-ghost
threshold, is what shows up as **pixelated borders**. Two mechanisms keep the
selection coherent:

**Clip weight (smooth, sharp luma).** Pass 1 drops a frame whose own luminance is
clipped or black (its envelope is `0` there). Pass 2 weighted purely by the
consensus-*predicted* value, so it never re-checked the frame's own clip state —
a frame that is actually clipped (a specular highlight that *moved* into this
pixel) but predicted mid-range was let back in and pulled its clipped value into
the merge. Both passes now multiply the weight by a **clip weight**
`_hdrmerge_clip_weight(Lᵢ)`: `1` for a well-exposed sample, a smooth Hermite ramp
to `0` between `0.7·white` and `white`, `0` at/above it. It is evaluated on the
**sharp** luma so small clipped glints are still caught, but because it is a
*continuous ramp* rather than a binary gate it does **not** flip a frame in/out
pixel-by-pixel along a bright motion border — an earlier hard gate (`skip if
Lᵢ ≥ white`) did exactly that and the border demosaiced into **pixelated edges**.
The ramp still reaches `0` at the white point, so clipped values can't leak in
(the magenta stays fixed).

**Blurred-luma weights (spatial).** The exposure weight and the de-ghost
consensus are derived from a **spatially blurred** copy of the luma (a small
separable Gaussian, σ≈2, computed once on the host). Because the weights now vary
smoothly across space, the frame selection no longer flips quad-to-quad — the
quad-grid pattern (hence the fringing) and the pixelated de-ghost borders
disappear. Crucially, **only the weights are smoothed; the merged pixel values
use the sharp frame data**, so no real detail is lost, and the **clip weight and
highlight neutralization keep using the sharp luma**, so smoothing never drags a
clipped frame's value into a valid pixel. It is the raw-domain counterpart of
`enfuse`'s pyramid weight-blending — superseded, for Bayer, by the full
multi-scale blend in [§3.8](#38-multi-scale-laplacian-pyramid-blend). A plain
Gaussian can still bleed slightly across a clip edge; an *edge-aware* (guided)
blur is the listed refinement for the single-scale path.

### 3.8 Multi-scale (Laplacian-pyramid) blend

The single-scale weighted average has to commit, *per pixel*, to **one** blend
of the frames. Where that blend changes quickly across the image (a moving edge,
a highlight boundary) it faces a dilemma: a **sharp** weight field seams (the
quad-grid → magenta/green fringing of [§3.7](#37-spatial-coherence-clip-gate--blurred-luma-weights)),
while pre-**blurring** the weights to hide the seam introduces halos and
over-triggers the de-ghost. There is no single blur radius that is right at every
scale at once.

A **Laplacian-pyramid blend** (Burt & Adelson 1983; the core of `enfuse`'s
exposure fusion, Mertens 2007) removes the dilemma by blending each frequency
**band** with the weights smoothed to the *matching* scale:

```
L{R}_l  =  Σ_k  G{W_k}_l · L{E_k}_l        (per pyramid level l)
```

where `G{W_k}` is the Gaussian pyramid of frame *k*'s normalized weight and
`L{E_k}` the Laplacian pyramid of its radiance `E = X·cal`. Coarse scales
cross-fade gently (no seam, no fringing) while fine scales keep each frame's own
detail (no halo); collapsing `L{R}` gives the merged radiance.

**Staying in the CFA/RAW domain.** A Gaussian blur must never mix colour
channels, so each frame is **de-interleaved into its four 2×2 Bayer positions**
(R, G, G, B), each a half-resolution plane, and a pyramid is built per position.
The weight is the quad-shared luma proxy — already one value per 2×2 quad — so a
**single** weight pyramid drives all four positions: every channel of a quad
stays the *same* frame mixture at every scale, which is exactly what stops the
demosaicer inventing colour. Re-interleaving the four collapsed planes yields the
merged CFA, **still scene-linear** (output `= E/cal_max`, the unchanged
convention). The pyramid's own multi-scale smoothing *is* the spatial coherence,
so here the weights come from the **sharp** luma (the host-side luma blur of
[§3.7](#37-spatial-coherence-clip-gate--blurred-luma-weights) is dropped on this
path — which also removes the de-ghost over-trigger it caused at higher
thresholds). The clip weight carries over unchanged; the "all frames clipped ⇒
white" neutralization also carries over as a per-quad mask applied after collapse,
but **re-keyed to the shortest exposure's *weakest* channel** rather than the
brightest — see *Highlight neutralization on the weakest channel* below.

**Weights.** Per quad, `w_k = clip(L_k) · envelope(L_k/white) · deghost_falloff(…)`
(the de-ghost term only when enabled, anchored to the same reference-frame /
median consensus as [§3.6](#36-runtime-options)). Weights are normalized across
frames so `Σ_k Ŵ_k = 1`; the Gaussian pyramid preserves that partition of unity
at every level, so the band blend is a true convex combination — energy is
preserved and a flat scene reconstructs exactly.

**Log-domain blend (no colour casts).** A multi-scale blend spreads any
cross-frame *disagreement* — a clipped highlight, a moving object, a specular
glint — into a low-frequency halo (Laplacian ringing). In **linear radiance**
that halo scales with the *absolute* brightness of the disagreement, and because
each channel rings differently it demosaics into a **green/yellow tint** that
bleeds far from the feature (foliage/water, full of small bright movers, cast it
everywhere). So the pyramid blends in a **log (perceptual) domain**: `g(m) =
log(m + ε)` on the normalized radiance `m = X·cal/cal_max`, collapsed and
inverted `m = eᵍ − ε`. In log, a disagreement is a bounded *contrast* (the log
of the ratio), so its halo is a small, near-achromatic *relative* ripple — and
the brighter the disagreement, the larger the reduction (a 5× mover's colour
halo drops ~3×; a clipped highlight, tens to hundreds ×, far more). `ε`
(`DT_HDRMERGE_PYR_EPS`) is the shadow floor: `m + ε` is linear for `m ≪ ε` (no
log blow-up or noise gain at black) and logarithmic above. Frames that *agree*
have equal log values, so static content still reconstructs exactly — this is
also why `enfuse` fuses in a gamma/perceptual domain, not linear.

**Feathering (`feather` config key, 0..1).** The pyramid already blends across
scales, but the *finest*-scale weight edge can still leave a faint visible
exposure transition. `feather` blurs the per-frame *normalized* weight before
building its Gaussian pyramid, softening that transition; because every frame
shares the kernel and `Σ_k Ŵ_k = 1` (a linear blur of a constant is constant),
the partition of unity — hence the no-colour-cast guarantee — is preserved. The
blur is kept SHARP in clipped highlights (handed off by the shortest-exposure
luma): there the merge already resolves to a single exposure, so there is no seam
to feather, and blurring would only bleed the neutral core into the corona. `0`
is crispest; the default `0.5` is a gentle feather.

**Highlights: converge to the shortest exposure.** Where every frame clips the
brightest channel, the exposure weights all vanish. Averaging the frames there
(the old `1/N` fallback) mixes each exposure's *clipped under-estimate* of the
radiance, which desaturates the highlight to grey — and grey, after white
balance, is **magenta**. Instead the fallback puts all the weight on the
**shortest exposure** (largest `cal`), the frame with the most highlight head
room. The whole highlight then resolves to that one frame, so it keeps its colour
and its smooth rolloff — i.e. it matches developing the darkest bracket directly,
rather than inventing a grey/magenta blend.

**A clipping frame defers its colour to the reference, and its brightness where
the reference is brighter.** The `1/N` fallback above only fires where *every*
frame clips. The subtler case is a bright, still-recoverable highlight — a specular
wave-crest, the sun's corona — that clips in the **longer** exposures but not in
the shortest. On these warm highlights **green is the brightest raw channel and
clips first**: the longer frame's green is capped to `white` while its red/blue
stay valid, so that frame is green-deficient — magenta. Its finest-level weight is
already ~0 (the clip weight keys on the max channel = green), but the pyramid's
**coarse** levels still leak that capped colour into the neighbourhood (the
exposure-fusion halo), dragging the merged highlight toward magenta at
`deghost = 0` where nothing rejects it. So as a frame approaches clipping (a
smoothstep `t` on its own luma from `0.9·white` to `white`) its pyramid radiance is
faded toward the shortest exposure's — the highlight reference that still has
head-room — **splitting colour from brightness**:

- **Colour** — always adopt the reference's channel ratios. This repairs the capped
  channel *everywhere the frame clips*, including faint far speculars and the
  **coronae** of clipped discs, where own and reference luminance are close (so a
  purely brightness-gated substitution would leave the colour, and its magenta,
  untouched — the residual seen in an earlier iteration, worst in the dim
  foreground waves).
- **Brightness** — keep the frame's own (capped) luminance, lifted toward the
  reference's only to the extent the reference is genuinely brighter here,
  `g = smoothstep(1, 1.1, L_ref/L_own)`. A real clipped under-estimate
  (`L_ref ≫ L_own`) recovers the reference's brightness; content this frame captured
  that the reference did not — or that moved / mis-registered between them, where
  this frame is the *brighter* one (`g → 0`) — keeps its own luminance, so the
  substitution can never punch a dark hole and ring into a speckle along an edge.

Per channel `target = X_ref·cal_ref/cal_max · (g + (1−g)·L_own/L_ref)`, faded by
`t`; the whole pixel is faded (all four Bayer positions) so channels stay coherent.
On the real two-frame sunset bracket this pulls the specular green-deficiency
(`√(R·B)/G`) from ~0.96–0.98 (visible magenta, worst in the foreground) down to
~0.92 — the shortest exposure's own value — in both the mid and foreground water,
`deghost`-independent, while the clipped patch fully recovers its brightness (the
regression-guard test reconstructs `G` to its reference), with the sun core,
shadows and static edges unchanged and no edge speckles.

**Highlight neutralization on the *weakest* channel (the "blown sun").** A pixel
is only truly unrecoverable — and pinned to the white point so darktable renders
it as a neutral clipped highlight, not a channel-clip mismatch (magenta) — when
**all** channels clip in every frame. So the mask is keyed to the shortest
exposure's quad **minimum** channel (`mincore`), not to `minb` (the brightest
channel). Keying it to the brightest channel whitened the whole zone where only
green clips, discarding the recoverable red/blue and demosaicing into a hard,
defined **white disc** with a magenta rim; keying it to the weakest channel keeps
that still-coloured corona (it converges to the shortest exposure, above), so the
blown core stays small and its border stays diffuse. A narrow tonal ramp below
white, then a small **feather-independent** spatial blur (widening it would bleed
white into the corona → magenta), lerp the output recovered→white for a smooth,
not pixelated, edge.

**Scope & cost.** Implemented on the **CPU** only for now (the merge is a
one-shot export step, so the GPU's single-scale kernel is kept for the `linear`
mode and a multi-kernel pyramid port is left as future work). **Bayer only**:
X-Trans (6×6, not a 2×2 quad) and the no-luma legacy case fall back to the
single-scale path. Memory is bounded in the frame count by accumulating the
blended Laplacian incrementally (four position accumulators + one frame's working
pyramids at a time). Selected by `plugins/lighttable/hdrmerge/blend` (`pyramid`,
the default, or `linear`).

---

## 4. Differences vs. the previous algorithm

| Aspect | Previous (weighted average) | New (hdrmerge two-pass) |
|--------|------------------------------|--------------------------|
| Combination | One weighted average of **all** frames | Per-pixel estimate that strongly prefers the **single best-exposed** frame |
| Spatial blending | single weighted average per pixel | **multi-scale Laplacian-pyramid** band blend (Bayer, `pyramid` default) — transitions cross-fade across scales, no quad-grid demosaic fringing |
| Weight domain | Photon-count × envelope of the **3×3 block max** | Envelope of a **shared 2×2 block max** (consistent across CFA channels) |
| CFA false colour on motion | avoided by averaging everything (blurry) | avoided by luminance-guided weights (sharper, colour-correct) |
| Magenta blown highlights | fully-clipped pixels forced to white | unrecoverable pixels forced to white — `linear`/GPU path: brightest channel clips every frame; `pyramid` default: the shortest exposure's *weakest* channel, keeping the coloured corona ([§3.8](#38-multi-scale-laplacian-pyramid-blend)) |
| Weight shape | Broad hat (`envelope`, mild falloff) | Sharply peaked `exp(α(1/s+1/(1−s)))` (strong extreme rejection) |
| Robustness pass | none (single accumulation) | **second pass** re-weighting by the consensus-predicted value |
| Saturation handling | special-cased "use only if nothing better" branch | falls out of `w→0` near clipping + plain-mean fallback when all clipped |
| Weight shape choice | fixed | `exponential` **or** `triangular` (config) |
| De-ghosting | none | **optional** *reference-frame* soft fall-off — moving content resolves coherently to one frame (config threshold) |
| Output determinism | exif/name from first **selected** frame → exposure varied with order | exif/name from the shortest-exposure frame → identical result regardless of order |
| Cross-frame texture (foliage/waves) | blends differing content → smear/ghosts | near-single-frame selection, plus optional de-ghost fall-off → far less blending |
| Calibration | `cal = 100/(aperture·exp·iso)` | identical |
| Output scale | `÷ (max cal)` → `1.0 = brightest` | identical (`× 1/max cal`) |
| Memory model | streaming: 2 accumulators (`pixels`,`weight`) | collect all `N` frames (fp16), then one merge pass |
| Acceleration | OpenMP only | OpenMP **and** OpenCL (with CPU fallback) |

What is intentionally **kept**: the radiometric calibration, the saturation
safety margin (`3000/65535`), the `1.0 = brightest` output convention, and the
"fully clipped ⇒ white" result — so the merged DNG drops into the existing
pipeline with no downstream changes. The previous algorithm's cross-channel
consistency (its 3×3-block coupling) is kept in spirit by the always-on
luminance-guided weighting, which achieves it per Bayer quad rather than by
spatial blur.

---

## 5. Architecture & data flow

The previous code accumulated incrementally and therefore needed only two
buffers. The two-pass estimator needs every frame available together (pass 2
revisits all frames with the pass-1 result), so the collection step changed:

```
_control_merge_hdr_job_run()                     (control_jobs.c)
  └─ for each bracketed image:
       dt_imageio_export_with_flags(... "pre:rawprepare" ...)
          └─ _control_merge_hdr_process()        # optional auto-align, then
                                                 #   *stores* the (aligned)
                                                 #   frame + its cal factor
  └─ dt_hdrmerge_process(&hm)                     (common/hdrmerge.c)
       ├─ _hdrmerge_process_cl()                  # OpenCL, if available & fits
       └─ dt_hdrmerge_process_cpu()               # OpenMP fallback
  └─ dt_imageio_dng_write_float(...) → import     # unchanged
```

`_control_merge_hdr_process()` allocates `frames[total]` on the first call and
stores each `rawprepare` output as an fp16 plane, a companion fp16 `luma` plane
(the 2×2 mosaic-block maximum), and its `cal`. All geometry/metadata capture
(filters, xtrans, white-balance, color matrix, orientation) is unchanged.

**Auto-alignment integration.** This work sits on the `add_hdr_merge_auto_align`
branch, which registers handheld/tripod brackets onto a reference frame
(OpenCV, `common/hdr_alignment.*`) before merging. The two features compose
cleanly: each frame is first warped onto the reference (`in_buf`), and it is the
**aligned** buffer that is converted to fp16 and collected for the two-pass
merge. Alignment removes geometric mis-registration (camera motion); the merge's
optional de-ghosting then handles residual *content* changes (foliage, ripples).
Without OpenCV, alignment is a no-op and the merge is unaffected.

### Files

| File | Change |
|------|--------|
| `src/common/hdrmerge.h` | **new** — `dt_hdrmerge_t` struct, weight enum, fp16 converters, constants, API |
| `src/common/hdrmerge.c` | **new** — `dt_hdrmerge_process_cpu()` (OpenMP) + `_hdrmerge_process_cl()` (OpenCL) + dispatcher |
| `data/kernels/hdrmerge.cl` | **new** — `hdrmerge_merge` compute kernel |
| `data/kernels/programs.conf` | registered `hdrmerge.cl` as program `42` |
| `data/darktableconfig.xml.in` | registered the three `plugins/lighttable/hdrmerge/*` config keys |
| `src/CMakeLists.txt` | added `common/hdrmerge.c` |
| `src/control/jobs/control_jobs.c` | collect fp16 frames + luma planes, read config, call `dt_hdrmerge_process()`; updated cleanup |

---

## 6. CPU implementation

`dt_hdrmerge_process_cpu()` first tries the multi-scale (`pyramid`) blend of
[§3.8](#38-multi-scale-laplacian-pyramid-blend) — the default for Bayer — and
only runs the single-scale estimate below when the pyramid is disabled, the CFA
is X-Trans, there is no luma proxy, or a pyramid allocation fails. That
single-scale path is a single `DT_OMP_FOR()` loop over pixels; each
iteration is independent (no reduction). Per pixel it runs the two-pass
estimate of [§3.3](#33-two-passes), decoding each fp16 sample and its shared
`luma` with `dt_hdrmerge_half_to_float()`, deriving the weight from the luma and
applying the optional (luminance) de-ghost fall-off. Frames and luma are separate
aligned fp16 buffers (`frames[i][p]`, `luma[i][p]`), so the inner loop is a
small gather across frames with sequential access within each frame. The weight
envelope (`_hdrmerge_weight`, kept in sync with the kernel copy) is evaluated
through a precomputed LUT in the hot loop
([§8.1](#81-weight-envelope-lut-cpu-hot-path)).

---

## 7. OpenCL implementation

`data/kernels/hdrmerge.cl : hdrmerge_merge` mirrors the CPU path. The `N`
fp16 frames are uploaded into **one packed device buffer** (`frame i` at offset
`i·width·height`), read with `vload_half`, plus the matching packed `luma`
buffer and a tiny `cal[]` buffer; one work-item per pixel runs the two passes
and writes the normalized result. Offsets use `size_t` (a packed multi-frame
buffer easily exceeds the 24-bit range of `mad24`/`mul24` at high resolution).

Host side (`_hdrmerge_process_cl`):

1. Gated on `dt_opencl_is_enabled()`; kernel id is registered lazily once via
   `g_once` (`dt_opencl_create_kernel` just appends to the global registry, so
   no change to `opencl.c` startup is required).
2. `dt_opencl_lock_device(DT_DEV_PIXELPIPE_EXPORT)`; `< 0` ⇒ CPU fallback.
3. **Memory guard** — the packed buffer must fit a single allocation
   (`dev[devid].max_mem_alloc`) and the total must fit
   `dt_opencl_get_device_available(devid)`; otherwise CPU fallback.
4. Allocate, upload (one plane at a time), enqueue 2D over `(width,height)`
   (the kernel bounds-checks the rounded-up global size), read back.
5. Any failure `goto cleanup` releases buffers, unlocks the device and returns
   `FALSE` so the dispatcher runs the CPU path. **The merge therefore always
   succeeds.**

---

## 8. Performance & memory

- **Memory:** the new scheme holds all `N` frames at once, stored as fp16, each
  with a companion fp16 luma plane: `N · width · height · 4 bytes` (host) and the
  same in device allocations. For a 5-frame 24 MP bracket that is ≈ 0.48 GB. On
  top of that both single-scale paths transiently allocate a **blurred-luma** copy
  (`N · w · h · 2 bytes`, ≈ 0.24 GB for that bracket) — the CPU `linear` path and
  the GPU upload use it; the `pyramid` path does **not** (it returns before the
  blur). The `pyramid` path instead allocates its own float working set — four
  half-resolution Bayer-position accumulators, one frame's working pyramids and a
  few half-resolution scratch planes, ≈ `14 · w · h bytes` (independent of `N`,
  ≈ 0.34 GB at 24 MP). The GPU path falls back to CPU when the packed frame buffer
  would not fit a single allocation, or when any of these allocations fails (the
  `pyramid` path then drops to the single-scale blend). The previous code used two
  buffers regardless of `N`; computing the luma on the fly to drop its plane is a
  possible future memory optimization.
- **Compute:** O(N) per pixel per pass, fully parallel. The weight envelope is
  evaluated `2N` times per pixel (once per frame in each pass); the CPU path
  serves it from a precomputed **LUT** ([§8.1](#81-weight-envelope-lut-cpu-hot-path))
  instead of a live `exp()`, and the GPU kernel uses `native_exp`. The CPU also
  pays one fp16→float decode per sample per pass. The `pyramid` path spreads the
  same `DT_OMP_FOR` row/pixel parallelism through its per-scale primitives — the
  `reduce`, `expand`, Laplacian-difference and `collapse` passes are all
  row-independent and threaded — so it too scales across cores (on a 4-core box a
  24 MP × 3 bracket merges in ≈ 0.9 s, ≈ 1.9× the earlier serial-primitive
  version; the win grows with the core count).
- **GPU↔host transfers:** frames are uploaded once; only `width·height` floats
  are read back. On the GPU path this transfer, not the arithmetic, dominates —
  which is exactly what fp16 storage halves.

### 8.1 Weight-envelope LUT (CPU hot path)

Profiling the CPU merge (single thread, 12 MP × 5 frames) decomposes the per-
pixel cost into two roughly comparable parts — the fp16→float **decode**
(≈ 30 %) and the **weight evaluation** — with the rest (calibration, the two
weighted sums, the highlight test) a small remainder. The weight was the larger,
and the more reducible:

- A live `expf()` per call (the literal envelope) is the baseline.
- The envelope is a fixed 1-D function of the normalized brightness `s`, so it is
  precomputed once into a **4096-entry LUT** (`_hdrmerge_build_weight_lut`) and
  read with linear interpolation (`_hdrmerge_weight_lut`). This is **≈ 1.5–1.8×
  faster** end-to-end than the `expf` path while staying numerically exact for
  this use: ≤ 2.6 × 10⁻⁶ on the weight and ≤ 6 × 10⁻⁸ on the normalized output
  vs. `expf` — far below fp16 quantization and sensor noise. The table is 16 KB,
  built once outside the parallel loop, and read-only across threads.

The remaining LUT cost is its two data-dependent L1 loads per call; that is the
floor for any table, and the alternatives traded accuracy or correctness for
little or nothing, so they were **measured and rejected**:

| Alternative | Result | Verdict |
|-------------|--------|---------|
| Bit-hack `exp` (Schraudolph) instead of the LUT | only ≈ 9 % faster than the LUT, **overflows to garbage** for the very-negative arguments at `s→0/1` (needs a clamp), and ≈ 4 % weight error where it is valid | rejected — accuracy/robustness loss for a one-time export |
| Cache each frame's decoded value in pass 1 for reuse in pass 2 | the extra store/load ≥ the re-decode it saves (`N` is small) | rejected — no gain (slightly slower) |
| Branchless LUT (clamp `s`, no edge branch) | slower — the clamp costs more than the well-predicted edge branch | rejected |
| Precompute `1/cal` to turn pass 2's `L_ref/cal` into a multiply | within run-to-run noise — the division was never the bottleneck | not worth the code |

The conclusion: the LUT is the sweet spot — it removes the only expensive,
*safely* removable term while keeping the result numerically identical. Beyond
it, the decode and the table's dependent loads are near-irreducible without an
accuracy sacrifice or a SIMD-across-pixels rewrite (which the strided per-frame
fp16 layout works against); both are deliberately left alone.

---

### Implemented

- **Luminance-guided weighting** — fixes CFA false colour on moving content
  ([§3.2](#32-luminance-guided-weighting-avoids-cfa-false-colour)); always on.
- **Blown-highlight neutralization** — fixes magenta sun / specular crests
  ([§3.4](#34-blown-highlight-neutralization-magenta-sun--specular-crests)); always on.
- **fp16 frame storage** — halves the per-plane memory ([§3.6](#36-runtime-options)).
- **Weight choice** — `exponential` (default) or `triangular`, config key.
- **De-ghosting** — opt-in *reference-frame* soft fall-off: moving content
  resolves coherently to one auto-picked best-exposed frame (no fragmented ghosts
  / false colour), median fallback where the reference is clipped/black, config
  threshold (replaced the hard veto, then the mean, then the median consensus).
- **Order-independent output** — metadata (exif, file name) keyed to the
  shortest-exposure frame, so the merge is deterministic regardless of selection
  order ([§3.5](#35-output-normalization)).
- **Multi-scale (Laplacian-pyramid) blend** — the raw-domain analogue of
  `enfuse`'s exposure fusion: per-Bayer-position pyramids, band-by-band weight
  blending, scene-linear output. Removes the seam-vs-halo dilemma of the
  single-scale average (no quad-grid fringing, no halos) while staying in the CFA
  domain. Bayer + CPU; default (`blend = pyramid`)
  ([§3.8](#38-multi-scale-laplacian-pyramid-blend)).

### Remaining future work

- **User-selectable de-ghost reference + whole-bracket motion.** The de-ghost
  reference is auto-picked (best-exposed); letting the user choose *which* frame
  (hence which instance of the motion to keep) is the natural GUI step. And
  because moving content is resolved to the reference's *exposure*, a region that
  moves *and* is out of the reference's range loses dynamic range there;
  optical-flow registration (warp each frame onto the reference, then merge with
  full DR) would lift that limit. An IRLS-refined consensus is a cheaper partial
  step for the latter.
- **Spatial / multi-scale blending (the `enfuse` model).** This whole algorithm
  is a *per-pixel, raw-domain* radiometric estimator: each pixel is decided
  independently, on the CFA mosaic, with no spatial coupling. `enfuse` solves
  ghosting and "harsh transition" complaints differently — it is **exposure
  fusion** (Mertens–Kautz–Van Reeth 2007), not an HDR merge:
  1. it works on the **already-demosaiced, colour-managed, gamma-encoded LDR**
     images (the exported JPEGs), so per-channel CFA mismatch — the source of
     our magenta — *cannot occur*; it never touches a Bayer site;
  2. it picks pixels by perceptual **quality weights** (local contrast,
     saturation, well-exposedness) rather than reconstructing scene radiance;
  3. it blends through a **multi-resolution Laplacian pyramid**: the weight maps
     are smoothed per frequency band and recombined, so the seam between
     "which frame won" is feathered across *every spatial scale* — there is no
     hard per-pixel boundary to demosaic into a blocky/magenta edge.
  The trade-off: `enfuse` outputs a finished, tone-mapped LDR (you lose the
  scene-linear DNG and the raw pipeline), whereas this module outputs a linear
  raw you can still develop with filmic etc. The key insight — `enfuse`'s
  multi-resolution pyramid blend — is now brought **into the raw domain**: the
  **multi-scale Laplacian-pyramid blend** ([§3.8](#38-multi-scale-laplacian-pyramid-blend))
  feathers the "which frame won" selection across every spatial scale while
  keeping the scene-linear CFA output. What remains deferred: (a) a **GPU
  (multi-kernel) port** of the pyramid — it is CPU-only today; (b) **X-Trans**
  support (it falls back to the single-scale path); (c) an **edge-aware (guided)**
  blur for the single-scale path; (d) an optional **tone-mapped fusion** output
  mode for users who want the finished-LDR look directly.
- **Saturated single-channel highlights (linear / GPU path).** On the
  single-scale path the blown-highlight neutralization keys off the *brightest*
  channel, so a *saturated colour* whose dominant channel clips in every frame
  (e.g. an intense monochromatic light) is pushed toward white rather than kept
  saturated. The default `pyramid` path already avoids this: it keys the mask on
  the shortest exposure's *weakest* channel — computed on the fly from the four
  Bayer positions, **no extra plane** — and lets the recoverable corona converge
  to the shortest exposure ([§3.8](#38-multi-scale-laplacian-pyramid-blend)).
  Bringing the same weakest-channel rule to the linear/GPU path is the remaining
  step; a highlight that is genuinely all-channel-blown stays unrecoverable
  either way.
- **Luma plane cost.** The luminance proxy is a second fp16 plane per frame.
  Computing it on the fly (2×2 reads in the merge loop) would drop the plane at
  some compute cost.
- **GUI.** The knobs are config keys (preferences → processing → HDR merge); a
  dialog on the merge action (also exposing `white_thresh` / an EV-push, already
  `dt_hdrmerge_t` parameters) would be friendlier.
- **On-hardware GPU run** and CPU-vs-OpenCL output comparison (blocked here by
  missing build deps).

---

## 10. Validation

A full darktable build is not available in this container (no GTK3 / exiv2 /
lensfun / OpenEXR), so verification was done in layers:

1. **Algorithm correctness** — a standalone program reproducing
   `dt_hdrmerge_process_cpu()` exactly merges a synthetic 3-exposure bracket
   (cal `4 / 1 / 0.25`). The merged output recovers the ground-truth normalized
   radiance to display precision across > 4 stops, a fully-clipped pixel
   resolves to `1.0`, and the weight envelope peaks at `w(0.5)=1`.
2. **False-colour & magenta fixes** — using the real `hdrmerge.h`: (a) a moving
   neutral-grey bracket — per-channel weighting gives `R/G = 0.51` (cyan cast),
   luminance-guided gives `R/G = 0.60` (neutral); (b) a blown highlight whose
   green clips in every frame resolves to uniform white `1,1,1` (neutral) instead
   of magenta; (c) a recoverable highlight one stop down keeps its detail and
   stays neutral; (d) radiance recovery exact across stops.
3. **fp16 converters** — round-trip error ≤ 0.1 % over the full normal-half
   range (within ~14 stops), exact at `0 / 0.5 / 1.0`; denormals (≳16 stops
   down) correct via the rygorous bit-trick.
4. **OpenCL kernel syntax** — `hdrmerge.cl` (incl. `half` / `vload_half`, the
   shared `native_exp` weight and the de-ghost fall-off) passes clang's OpenCL
   frontend (`-x cl -cl-std=CL1.2 -finclude-default-header`).
4b. **De-ghost fall-off** — the Hermite ramp is `1` at consensus, monotonically
   non-increasing, exactly `0` at the threshold (matching the old veto's
   rejection point) with zero slope there (C¹, no seam), and `0` beyond — checked
   numerically against the old hard veto.
4c. **De-ghost consensus** — a synthetic Bayer quad over 5 frames with a green
   streak. With a *minority* mover (1 of 5) the mean barely helps (`G/R = 1.025`
   at threshold `0.5`) while the median fully rejects it (`1.000`). With a
   *majority* mover (3 of 5) the **median tracks the mover** (`G/R = 1.487` —
   reproduces the "ghosting gets worse when I turn it up" report), whereas the
   **reference frame** resolves the pixel to one frame coherently: anchored to a
   static frame the merged G equals that frame's value *exactly* (`G/R = 1.000`,
   mover removed); anchored to an object frame it keeps the real object — no
   false colour either way. Static R/B sites stay exactly neutral throughout.
4d. **Clip weight** — a static water background with a *clipped* specular glint
   crossing one of five frames: the merged radiance is contaminated `0.30 →
   0.485` without the pass-2 clip weight and stays exactly `0.30` (no leak, since
   the ramp is `0` at the white point) with it. And across a bright border that
   sweeps the clip point under sensor noise, the old **hard gate** flips a frame
   in/out 5× (total per-pixel variation `5.0` → pixelation) while the **smooth
   ramp** has variation `0.43` — 11.6× less, no binary flips.
4e. **Spatial coherence** — near a clip threshold with sensor noise, the sharp
   luma toggles a frame in/out 28× across a 64-px row (the checkerboard that
   demosaics into fringing); the blurred luma toggles 6× — selection coherence
   restored. The full `dt_hdrmerge_process_cpu` (blur + clip-gate + both passes)
   was also run on a 32×32 synthetic bracket with a mover and a clipped glint:
   output is finite and correctly normalized (`[0.05, 0.20]` for a 0.2–0.8
   radiance gradient ÷ cal_max), zero bad pixels, at both `deghost = 0` and `0.4`.
4f. **Multi-scale pyramid blend** — the *real* `src/common/hdrmerge.c` (stub
   headers, no OpenCL) compiles clean under `-Wall -Wextra` and passes a numeric
   suite: a frame-identical scene radiance (smooth gradient, 3 exposures) is
   reconstructed to `out = E/cal_max` within **0.1 %** — confirming partition of
   unity, a lossless pyramid round-trip (build → Laplacian → collapse) and the
   de-interleave/re-interleave — including with de-ghosting on, **odd dimensions**
   (edge clamping) and a degenerate 2×2. A 4-decade exposure ramp blends finite,
   in range, with no quad-grid curvature; a fully-clipped quad neutralizes to
   exactly `1.0` while normal pixels do not.
4g. **Colour-cast (halo) diagnosis + log-domain fix** — a *coloured* static scene
   (R≠G≠B, agreeing frames) reconstructs each channel to 0.07 %: no per-channel
   bug. The tint was localized to *disagreement*: a green-only mover present in
   one frame casts, in **linear** radiance, a **G-only halo** on the static
   background (`G = 0.94` at 8–16 px, recovering by ~30 px) while the single-scale
   blend stays flat — Laplacian ringing spreading per channel. Blending in the
   **log domain** shrinks it to `G = 0.98` (5× mover) and `G = 1.01` for a 78×
   mover (catastrophic in linear), with the exact reconstruction of agreeing
   frames unchanged — confirms the green/yellow "tints all over" and the fix.
4h. **Feather + neutralization border** — the `feather` weight-blur keeps the
   coloured-static reconstruction exact (0.07 %) and per-channel bias `≈1.0` at
   `feather = 0.5/1.0` (partition of unity survives the blur), and does not
   worsen the green-mover halo (`G = 0.99`). A large clipped "sun" core stays
   white (`1.000`) while its border is a smooth, monotone ramp (12–24 intermediate
   samples vs a hard step's 0) at `feather = 0` and `1`, with the far background
   unchanged (`0.100`) — no hollowing and no white bleed into the corona.
4i. **Blown-sun highlights (magenta + hard disc)** — a synthetic colored radial
   sun (green clips first) merged, then compared per channel against the shortest
   exposure (the "darkest bracket" the user compares to). The corona is now
   green-dominant and tracks the shortest exposure's chroma to within **0.095**
   (R/G, B/G ratios) at the default `feather = 0.5` — versus a grey, ~2× larger
   deviation before, i.e. no magenta — while the core→corona transition stays
   smooth (no hard disc). Keying neutralization to the shortest exposure's
   *weakest* channel and making the neutralization blur feather-independent were
   both necessary: keyed to the brightest channel, or with a feather-widened blur,
   the deviation regressed to ~0.16–0.18 (visible magenta).
4j. **Clipped-highlight leak** — a bright coloured patch clipped in the long/mid
   exposures but recoverable in the short one merges to only ~½ its reference value
   (chroma error ~0.18) because the clipped frames' capped `X·cal` under-estimate
   leaks through the pyramid's coarse levels; the result is `deghost`-independent
   in linear synthetic data (confirming it is the leak, not motion). Fading a
   near-clipped frame toward the shortest exposure's radiance — scaled by how much
   brighter the reference is ([§3.4](#34-blown-highlight-neutralization-magenta-sun--specular-crests))
   — restores the patch's colour and brightness, with the sun corona and all other
   tests unchanged.
4k. **In-tree unit tests** — `src/tests/unittests/test_hdrmerge.c` (cmocka) drives
   the public `dt_hdrmerge_process_cpu()` on synthetic Bayer brackets and asserts:
   the `pyramid` and `linear` paths reconstruct a frame-identical coloured scene to
   `out = E/cal_max` within **1 %** (energy preservation); an all-channel-blown core
   neutralizes to `1.0`; and a bright coloured patch clipped in the long/mid
   exposures keeps its colour (chroma deviation from the shortest-exposure reference
   well under the 0.13 bound, vs ≈ 0.18 for the raw coarse-leak) and its brightness
   (green recovers to ≈ its reference, not the ≈ ½ of the leak bug). Results are
   thread-count-independent (the per-pixel loops carry no cross-pixel reduction),
   verified identical at 1 and 4 OpenMP threads.
5. **CPU + OpenCL host paths** — the CPU functions compile clean under
   `gcc -Wall -Wextra` with stubbed darktable signatures; the *real*
   `_hdrmerge_process_cl` (compiled with `-DHAVE_OPENCL` against OpenCL stubs)
   checks the 14-arg kernel call (incl. `bluma` and `ref_frame`), the
   `frames`/`luma`/`bluma` buffers and the `goto cleanup` flow.
6. **API/signature review** — every darktable call checked against the headers
   (`opencl.h`, `darktable.h`, `pixelpipe.h`, `conf.h`).

> Remaining verification once a full toolchain is available: compile the tree,
> run an actual GPU merge, and compare CPU vs OpenCL outputs (they share the
> same math) on a real bracket, including the de-ghost path.

---

## 11. Task progress

- [x] Study the previous algorithm (`_control_merge_hdr_job_run` / `_process`)
- [x] Extract the hdrmerge weight + two-pass scheme from `wjakob/hdrmerge` and
      the vkdt port (`hanatos/vkdt#269`)
- [x] OpenCL kernel `data/kernels/hdrmerge.cl` (+ `programs.conf`)
- [x] CPU + OpenCL driver `src/common/hdrmerge.{c,h}` (+ `CMakeLists.txt`)
- [x] Refactor `control_jobs.c` to collect frames and call `dt_hdrmerge_process`
- [x] Numerical + isolated-compilation validation
- [x] This design document
- [x] fp16 frame storage (CPU + OpenCL via `vload_half`)
- [x] Selectable weight (`exponential` / `triangular`) — config key
- [x] Luminance-guided weighting (always on) — fixes CFA false colour on motion
- [x] Optional de-ghosting — config key (soft Hermite fall-off against a
      **reference frame**; moving content resolves coherently to one frame —
      replaced the hard veto, then the mean, then the per-pixel median consensus)
- [x] Register config keys in `darktableconfig.xml.in`
- [x] Weight-envelope LUT (CPU hot path, ≈ 1.5–1.8×) + `native_exp` (GPU) —
      profiled, numerically exact ([§8.1](#81-weight-envelope-lut-cpu-hot-path))
- [x] Deterministic, order-independent output — metadata keyed to the
      shortest-exposure frame ([§3.5](#35-output-normalization))
- [x] Reference-frame de-ghosting — moving content resolves coherently to one
      auto-picked best-exposed frame (CPU + OpenCL `ref_frame`)
- [x] Smooth clip weight — exclude a frame clipped/black by its own (sharp)
      luminance via a continuous ramp, not a binary gate (fixes magenta on moving
      specular highlights without pixelating bright borders)
      ([§3.7](#37-spatial-coherence-clip-gate--blurred-luma-weights))
- [x] Spatial coherence — weights/consensus from a host-blurred luma so the frame
      selection varies smoothly (no demosaic fringing / pixelated borders); values
      stay sharp (CPU + OpenCL `bluma`) ([§3.7](#37-spatial-coherence-clip-gate--blurred-luma-weights))
- [x] Multi-scale (Laplacian-pyramid) CFA blend — per-Bayer-position pyramids,
      band-by-band weight blending, scene-linear output; the raw-domain analogue
      of `enfuse` exposure fusion (Bayer, CPU, `blend = pyramid` default)
      ([§3.8](#38-multi-scale-laplacian-pyramid-blend))
- [x] Log-domain pyramid blend — bounds the per-channel disagreement halo so
      clipped highlights / movers no longer cast green/yellow tints
      ([§3.8](#38-multi-scale-laplacian-pyramid-blend))
- [x] `feather` control + feathered highlight neutralization — soft exposure
      transitions and a smooth (not pixelated) border around a blown-highlight
      core ([§3.6](#36-runtime-options), [§3.8](#38-multi-scale-laplacian-pyramid-blend))
- [x] `legacy` weight option — the original pre-branch weighted-average merge,
      reproduced over the collected frames for direct A/B comparison
      ([§3.6](#36-runtime-options))
- [x] Blown-highlight fix — highlights converge to the shortest exposure (no
      grey/magenta), neutralization keyed to the weakest channel with a
      feather-independent border (diffuse blown sun, not a hard white disc)
      ([§3.8](#38-multi-scale-laplacian-pyramid-blend))
- [x] Clipped highlights fade to the shortest exposure where it is brighter —
      recovers the correct colour *and* brightness and stops the coarse-level leak
      that desaturated highlights to magenta at `deghost = 0`, without punching
      speckles along mis-registered edges ([§3.8](#38-multi-scale-laplacian-pyramid-blend))
- [x] Parallelized the pyramid primitives (`reduce` / `expand` / Laplacian /
      `collapse`) — ≈ 1.9× faster on 4 cores, output bit-identical ([§8](#8-performance--memory))
- [x] In-tree cmocka tests `src/tests/unittests/test_hdrmerge.c` — energy
      preservation, highlight neutralization, clipped-channel colour/brightness
      recovery ([§10](#10-validation) item 4k)
- [x] Code-review hardening — `mincore` added to the pyramid alloc guard (a
      failed allocation dropped to the single-scale path but could deref NULL
      first); the `legacy` merge now aborts with a message instead of writing an
      all-black DNG on a scratch-allocation failure; dead no-luma branch removed
      from the OpenCL host path
- [ ] Full-tree build & on-hardware GPU run (blocked: build deps absent here)
- [ ] Optional: GPU (multi-kernel) pyramid port, X-Trans pyramid support,
      edge-aware (guided) weight blur, user-selectable de-ghost reference,
      optical-flow registration for whole-bracket motion, tone-mapped fusion
      output mode, merge dialog

---

## 12. References

- Wenzel Jakob, **hdrmerge** — <https://github.com/wjakob/hdrmerge>
  (core weight + two-pass logic in `hdr.cpp`).
- **vkdt** HDR merge module — <https://github.com/hanatos/vkdt/pull/269>
  (`src/pipe/modules/hdrmerge/linear.comp`).
- Previous darktable merge: `_control_merge_hdr_job_run()` /
  `_control_merge_hdr_process()` in `src/control/jobs/control_jobs.c`.
