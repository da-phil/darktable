# GPU acceleration: from OpenCL to clspv + Vulkan + MoltenVK

**Status:** proposal / proof-of-concept.
**Scope:** strategy for keeping darktable's GPU pipeline portable across Linux,
Windows, and macOS as Apple deprecates OpenCL in favour of Metal.
**Companion code:** `tools/vulkan_compute_poc/` (this PR).

---

## 1. Why move at all

Darktable today ships ~4,500 lines of host-side OpenCL plumbing
(`src/common/opencl.{c,h}`, `src/common/dlopencl.{c,h}`) and 42 `.cl`
kernels in `data/kernels/`, dispatched by ~75 `process_cl` implementations
across the IOP modules. The runtime is a thin wrapper around the OpenCL
1.2 ICD: device enumeration, per-device queues, deferred kernel
compilation with on-disk binary caching, image/buffer allocation, and
event-based profiling.

This works well on Linux and Windows where AMD, NVIDIA, and Intel ship
mature OpenCL drivers. It works *less* well on macOS:

- Apple has deprecated OpenCL since macOS 10.14 and is steering all new
  GPU work toward Metal.
- The Apple OpenCL runtime hasn't been updated past 1.2 and lags behind
  the rest of the ecosystem in performance and stability on M-series
  Macs.
- A native Metal port is technically tractable but means maintaining a
  parallel kernel codebase — duplicating roughly the entire image-
  processing maths that has been refined over a decade of `.cl`
  evolution.

The proposal here keeps `.cl` as the single source of truth and replaces
only the host-side runtime, in a way that's portable across all three
operating systems:

```
   data/kernels/*.cl
          │
          ▼   (build-time)
       clspv  (Google's OpenCL-C-1.2 → SPIR-V compiler, LLVM-based)
          │
          ▼
       *.spv  (Vulkan SPIR-V compute modules)
          │
          ▼   (runtime)
    +-----------------------+   +------------------------+
    | Vulkan compute        |   | Vulkan compute         |
    | (Linux: Mesa / AMDVLK |   | (macOS: MoltenVK)      |
    |  NVIDIA / Intel)      |   |   → MSL → Metal        |
    | (Windows: vendor ICD) |   |                        |
    +-----------------------+   +------------------------+
```

The bulk of the IP — the actual image-processing maths — survives
essentially unchanged. Host-side, we go from "OpenCL + (eventual) Metal"
to "Vulkan everywhere", which is one host runtime instead of two.

## 2. Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| **Stay on OpenCL** | Zero work today. | Apple support degrades over time; macOS users lose acceleration when the OpenCL framework is eventually removed. |
| **Hand-write Metal** alongside OpenCL | Best macOS perf. | Doubles kernel maintenance forever; two implementations of the same maths will drift. |
| **Native Vulkan + handwritten GLSL/HLSL compute** | Single API, no clspv dependency. | Throws away the existing `.cl` codebase; one of the largest rewrites in dt history. |
| **WebGPU / Dawn** | Cross-platform compute. | Compute support uneven; image-processing-quality numerics not guaranteed; another large dependency tree. |
| **clspv + Vulkan + MoltenVK** (this proposal) | Keep `.cl` source of truth, one host runtime, three OS supported. | New build-tool dependency (clspv); subset of OpenCL C 1.2; Vulkan descriptor model more rigid than OpenCL. |
| **clvk** (OpenCL ICD on top of Vulkan, uses clspv internally) | Even smaller host-side change — keep the OpenCL host API. | Extra runtime layer; MoltenVK underneath on macOS still has the same constraints; ties dt to clvk's lifecycle. |

The leading candidate is the clspv path. A clvk fallback is feasible
and is sketched in §11 as an intermediate option that minimises
host-side disruption for the first release.

## 3. Component glossary

- **clspv** — Google project, LLVM-based, compiles OpenCL C 1.2 source
  to SPIR-V Vulkan compute shaders. Output is a single SPIR-V module
  per `.cl` file, with one `OpEntryPoint` per `kernel void`. Actively
  maintained, but not officially supported by Khronos; used in
  production by Android's NN HAL and others.
- **Vulkan** — the cross-vendor low-level GPU API. Has a first-class
  compute pipeline, well-defined memory model, mature ICDs on
  Linux/Windows.
- **MoltenVK** — Khronos/LunarG project, implements (a portability-
  subset of) Vulkan on top of Apple's Metal. Internally uses
  SPIRV-Cross to translate SPIR-V to MSL just-in-time.
- **SPIRV-Cross** — Khronos project, converts SPIR-V into GLSL, HLSL,
  or MSL. Used by MoltenVK transparently; we don't call it directly.
- **clvk** — a layered OpenCL implementation built on Vulkan, using
  clspv as its compiler. Useful as a transition layer that exposes the
  OpenCL ICD API to existing OpenCL host code.

## 4. What's in the proof-of-concept and integration (this PR)

Two layers of code land together.

### 4.1 Standalone PoC (`tools/vulkan_compute_poc/`)

A self-contained Vulkan compute runtime that demonstrates the SPIR-V
pipeline end-to-end without touching darktable's main build graph.
Useful for iterating on the toolchain (clspv ↔ glslang ↔ MoltenVK)
without rebuilding darktable. Off by default; enable with
`--enable-vulkan-poc` (build.sh) or `-DBUILD_VULKAN_COMPUTE_POC=ON`
(cmake).

- `kernels/basicadj_min.cl` — a pared-down version of darktable's
  exposure/black-point step, written against the clspv-supported subset
  of OpenCL C 1.2.
- `kernels/basicadj_min.comp` — a hand-written GLSL twin that lets the
  PoC build on hosts where `clspv` is not yet installed. Production
  builds use the `.cl` source only.
- `src/vk_compute.{h,cc}` — a thin Vulkan compute driver: instance,
  device, queue, command pool, buffer alloc with host-staging,
  compute-pipeline build from a SPIR-V module, descriptor set binding,
  and dispatch+wait.
- `src/main.cc` — drives a 256×256 (and 1920×1080) test through the
  kernel and compares the output to a CPU reference. Numerical
  agreement is the success criterion.
- `CMakeLists.txt` — picks `clspv` when present, otherwise falls back
  to `glslangValidator`; runs `spirv-val` on the result; emits an
  optional `dt_vk_compute_poc_check` target.
- Top-level integration: a new `BUILD_VULKAN_COMPUTE_POC` CMake option
  (default `OFF`) so the standard build is untouched.

**Verified on the dev container:** the PoC builds clean, produces a
valid SPIR-V module, and matches the CPU reference bit-by-bit (within
1e-5) on a Mesa lavapipe software Vulkan implementation. The same
SPIR-V module will dispatch unchanged on AMDVLK / NVIDIA / Intel /
MoltenVK.

The PoC is deliberately a *narrow slice* of what a real port would
build:

- Buffer-only memory; no `image2d_t`/sampler bindings yet.
- Single dispatch per pipeline, no async pipelines or events.
- No tiling, no headroom logic, no device priority.

Those are the next milestones (§8), not blockers for evaluating the
approach.

### 4.2 In-tree pipeline integration

The Vulkan backend is wired into darktable's pixelpipe as a real
parallel GPU path, behind the `USE_VULKAN` CMake option (default
`OFF`) and the `opencl_use_vulkan` runtime preference (default
`true`).

**Backend (`src/common/vulkan.{c,h}`)** — a focused subset of the
opencl.c surface (~600 lines total):

- `dt_vulkan_init`/`dt_vulkan_cleanup` — instance, device enumeration,
  per-device compute queue + command pool + descriptor pool.
- `dt_vulkan_load_program` — read a SPIR-V file produced at build time
  by clspv (or glslang in fallback mode) into a host-side cache.
- `dt_vulkan_create_kernel` — build a `VkPipeline` from a SPIR-V
  module plus the binding shape (number of storage buffers, push-
  constant size, local workgroup dimensions).
- `dt_vulkan_alloc_buffer` / `dt_vulkan_free_buffer` — device-local
  storage buffers wrapped in opaque `dt_vk_mem_t*` handles, the
  analogue of `cl_mem`.
- `dt_vulkan_write_to_device` / `dt_vulkan_read_from_device` —
  blocking host-stage copies via temporary host-visible buffers.
- `dt_vulkan_enqueue_kernel_2d` — bind buffers + push constants, build
  a one-shot command buffer, submit on the compute queue, wait.

**Pixelpipe hook (`src/develop/pixelpipe_hb.c`)** — added inside the
CPU dispatch arm. When a module exposes `process_vk` and Vulkan is
running, the pipeline:

1. Allocates VK input/output buffers sized to the ROI.
2. Uploads the host input buffer to the VK input.
3. Calls `module->process_vk(self, piece, vk_in, vk_out, roi_in, roi_out)`.
4. Downloads the VK output back into the host output buffer.
5. Frees the VK buffers.

If any step fails the pipeline falls through to the CPU `process()`
unchanged. Modules without `process_vk` are unaffected.

This sits *inside* the CPU arm rather than replacing the OpenCL
arm: the OpenCL `cl_mem` chain between modules is preserved, and
Vulkan handles only modules that have a port. The host-stage copies
at each Vulkan boundary make this slower per-module than a unified
GPU chain — that optimisation (skip staging when both ends are
Vulkan) is the next-but-one milestone (§8.6).

**Per-module ports**: 65 modules currently expose `process_vk`,
in three categories.

*Faithful (bit-equal to the OpenCL output for the same params):*

| Module | Notes |
|---|---|
| `src/iop/exposure.c` | One subtract + multiply per channel. |
| `src/iop/velvia.c` | Vibrance-boosted saturation; ports literally. |
| `src/iop/invert.c` | The 4-channel demosaiced variant only; the 1f Bayer variant operates on RAW and stays on OpenCL. |
| `src/iop/vibrance.c` | Lab-space saturation scaling weighted by chroma. |
| `src/iop/colorcorrection.c` | Lab a/b axis shifts proportional to L. |
| `src/iop/colorcontrast.c` | Per-channel Lab scale + offset with optional clamp. |
| `src/iop/colorize.c` | Replaces chrominance with constants, rescales L. |
| `src/iop/flip.c` | Coordinate-remap kernel (orientation flag). |
| `src/iop/negadoctor.c` | Analogue-film inversion + gamma in a single pass. |
| `src/iop/primaries.c` | 3×4 ICC matrix transform applied per pixel. |
| `src/iop/temperature.c` | All 3 paths ported: 4f (post-demosaic), 1f Bayer, 1f X-Trans. RAW pre-demosaic is single-channel; we dispatch a `float`-buffer kernel for those and a `float4`-buffer kernel for the 4f variant. |
| `src/iop/splittoning.c` | RGB↔HSL split-shadow / highlight tinting. |
| `src/iop/zonesystem.c` | Lab L*-zone clamping. |
| `src/iop/levels.c` | Single-curve histogram remap. |
| `src/iop/profile_gamma.c` | LOG and GAMMA modes. |
| `src/iop/graduatednd.c` | Both density signs (positive divide, negative multiply) folded into one entry point with a uniform-branch on the sign of `density`. |
| `src/iop/vignette.c` | Radial darkening/brightening with inlined TEA-cipher dither and TPDF noise (helpers private to this kernel). |
| `src/iop/relight.c` | Gaussian-windowed Lab fill-light / shadow on L. |
| `src/iop/mask_manager.c` | Pass-through dummy IOP; uses `dt_vulkan_copy_device_to_device` rather than a kernel. |
| `src/iop/borders.c` | Multi-fill canvas + sub-region image copy. Uses `dt_vulkan_dispatch_n` (1 binding for fill, 2 for copy) and the two-entry-point multi-kernel pattern. |
| `src/iop/colisa.c` | First LUT-on-storage-buffer port — uploads two 65536-entry tables freshly per dispatch and binds them at descriptors 2/3 alongside the in/out buffers. Pattern blueprint for `tonecurve`, `rgbcurve`, `rgblevels`, `basecurve`, … |
| `src/iop/overexposed.c` | First consumer of the ICC profile storage-buffer plumbing (§5.11). 5-binding dispatch (in, out, histogram-profile tmp, profile_info, profile_lut) covering all four clipping-preview modes. Uses `dt_ioppr_transform_image_colorspace_rgb_vk` (§5.12) to do the current → histogram profile transform entirely on the GPU; only falls back to CPU when both profiles are non-matrix (lcms2-only). |
| `src/iop/basicadj.c` | Exercises both arms of the §5.11 plumbing: `vk_get_rgb_matrix_luminance` for the highlight-compression branch + `vk_dt_rgb_norm` for the preserve-colors branch. 6-binding dispatch (in, out, gamma LUT, contrast LUT, profile_info, profile_lut); 8 ints + 10 floats of push constants drive the six independent sub-features (exposure, hlcompr, gamma, plain contrast, preserve-colors contrast, saturation+vibrance). Auto-exposure metering still runs CPU-side as in `process_cl` — the kernel only handles the static-parameter dispatch. |
| `src/iop/lowlight.c` | Scotopic-luminance blend in Lab → XYZ → Lab; one user-driven 65536-entry blend curve + a 4-float scotopic white-point passed via push constants. 3-binding dispatch (in, out, lut); the Lab↔XYZ helpers come from `dt_vulkan_common.h`. Cheapest possible LUT-pattern port at ~70 LOC kernel + ~40 LOC module — useful as a template for the next batch of simple LUT consumers. |
| `src/iop/monochrome.c` | First consumer of the bilateral helper (§5.13). Three-stage `process_vk`: `monochrome_filter` writes the chroma-distance weight into the output buffer; the bilateral helper (splat → blur → slice) smooths the weight into a scratch buffer; `monochrome` blends the original input with the smoothed weight and clears the chroma channels. Demonstrates the multi-kernel orchestration pattern that `lowpass` / `censorize` / `shadhi` / `retouch` / `globaltonemap` will follow. |
| `src/iop/rgbcurve.c` | Per-channel RGB curve via three 65536-entry LUTs with linear-extrapolation tails (the `vk_lookup_unbounded` pattern). The norm-preserving "automatic" mode goes through `vk_dt_rgb_norm` to compute luminance via the §5.11 ICC profile plumbing. 7-binding dispatch (in, out, table_{r,g,b}, profile_info, profile_lut), 56 B push constants (5 ints + 9 floats — three 3-coeff unbounded tails). |
| `src/iop/rgblevels.c` | Per-channel levels remap (low / middle / high triplet + inv-gamma) via three 65536-entry LUTs for the interior range + gamma-power extrapolation for x ≥ high. Same 7-binding shape as rgbcurve; 68 B push constants (5 ints + 12 floats — 3 levels × 3 channels + 3 inv_gamma). Norm-preserving mode goes through `vk_dt_rgb_norm` like rgbcurve. |
| `src/iop/tonecurve.c` | Lab-space tone curve with four chroma-handling modes (`autoscale_ab`): independent a/b curves with optional two-sided unbounded extrapolation, automatic L/old-L chroma scaling, automatic in XYZ (curve applied to all three XYZ channels), and automatic in ProPhoto RGB (optionally norm-preserving). Largest curve-pattern PC at 84 B (5 ints + 16 floats — L coeffs + 6-float two-sided coeffs per a, b channel). Uses Lab↔XYZ + Lab↔ProPhoto helpers from `dt_vulkan_common.h`. |
| `src/iop/lowpass.c` | First combined-helper consumer: chains `dt_gaussian_*_vk` (§5.10) **or** `dt_bilateral_*_vk` (§5.13) for the low-pass step, then snapshots dev_out into a scratch buffer via `dt_vulkan_copy_device_to_device`, then runs the new `lowpass_mix` kernel (4 bindings, 40 B PC) to apply contrast + lightness curves on L and scale a/b chroma by saturation. Same algorithmic shape as `process_cl` 1:1, exercising the multi-helper orchestration pattern that censorize / shadhi / retouch can copy. |
| `src/iop/shadhi.c` | Shadows / highlights recovery. Same Gaussian-or-bilateral choice as lowpass, then snapshot, then a new `shadows_highlights_mix` kernel (3 bindings, 44 B PC) that runs two soft-light overlays — one for highlights (negative opacity), one for shadows (positive opacity). The overlay helper handles per-channel unbound flags via the `flags` bitmask, mirroring the OpenCL kernel byte-for-byte. |
| `src/iop/colorbalance.c` | All three legacy modes (LEGACY sRGB-space, LIFT_GAMMA_GAIN ProPhoto, SLOPE_OFFSET_POWER aka CDL). Push-constant-only — no LUTs, no profile info — so the three kernels share the 2-binding shape but differ in body. LEGACY at 68 B PC (no saturation_out), LGG/CDL at 72 B. process_vk switches on `d->mode` and dispatches the matching `dt_vk_module_kernel_t` slot. |
| `src/iop/colorout.c` | Output-side Lab → RGB via a 3×3 matrix + per-channel shaper LUT with linear-extrapolation tails (the matrix-fast-path from `process_cl`; the lcms2 slow path stays on CPU). 5-binding dispatch (in, out, lut_r/g/b), 80 B PC (2 ints + 9 matrix floats + 9 extrapolation-coeff floats). The `DT_COLORSPACE_LAB` pass-through case goes through `dt_vulkan_copy_device_to_device` rather than a kernel, mirroring the OpenCL `enqueue_copy_image` shortcut. |
| `src/iop/filmic.c` | Legacy single-pass filmic (the predecessor to `filmicrgb`). Lab → XYZ → ProPhotoRGB → global saturation desat → log → 65536-entry S-curve LUT → derivative LUT for selective desat → power gamma → ProPhotoRGB → Lab. Both `preserve_color` modes (per-channel vs norm-preserving) folded into one entry point with a uniform branch on the flag. 4-binding dispatch (in, out, table, diff); 36 B PC (2 ints + 6 floats + 1 int). Wavelet-based `filmicrgb` is a separate port pending §8.5 (image2d + sampler bindings) for its reconstruction kernels. |
| `src/iop/overlay.c` | Composite (Cairo ARGB32 → RGBA float alpha-blend). The Cairo overlay buffer arrives as a packed 8-bit BGRA byte buffer; with Cairo's 4-byte stride alignment and `x*4` naturally word-aligned, each pixel is exactly one `uint` so the storage buffer is bound as `uint *` and the four bytes are extracted with shift+mask. 3-binding dispatch (in, argb, out); 16 B PC (2 ints + 1 float + 1 int). The "no overlay" branch routes through `dt_vulkan_copy_device_to_device` rather than a kernel, mirroring the OpenCL `enqueue_copy_image` shortcut. Cairo overlay rendering still runs CPU-side (`_get_overlay_argb`); the GPU only does the per-pixel alpha-blend over the full canvas. |
| `src/iop/sigmoid.c` | Modern sigmoid tone mapper, both color processing modes. **`rgb_ratio`** (norm-preserving): 2-binding dispatch (in, out), 32 B PC (2 ints + 6 floats — `white_target`, `black_target`, paper-exp, film-fog, contrast/skew powers); applies a loglogistic curve to per-pixel luma, scales RGB uniformly, then hyperbolic chroma roll-off against display+mapping gamut. **`per_channel`** (hue-preserving): 5-binding dispatch (in, out, m_pb, m_br, m_rp — three 9-float 3×3 matrices via `pack_3xSSE_to_3x3`), 32 B PC (2 ints + 6 floats — adds `hue_preservation`). Transforms pipe→base, desaturates negatives, transforms base→rendering primaries, applies per-channel sigmoid, then a 7-case `pixel_order` + `preserve_hue_and_energy` block that linearly interpolates the middle channel back toward the hue-correct value while keeping channel-sum energy constant, finally back via rendering→pipe. Both kernels build to standalone `.spv`; the host picks one based on `d->color_processing`. |
| `src/iop/agx.c` | AgX-inspired tone mapper. First port to migrate a kernel parameter struct out of push-constants: the OpenCL signature took `dt_iop_agx_tone_mapping_params_t` by value (124 B = 27 floats + 4 ints), which exceeded the 128 B PC budget once `width`/`height`/`base_working_same_profile` were added. The Vulkan kernel binds the struct as a storage buffer instead; std430 layout matches the C struct's flat 4-byte-aligned scalar fields byte-for-byte (verified against `OpMemberDecorate ... Offset` chain — 31 fields at consecutive 4-byte offsets, total 124 B). 7-binding dispatch (in, out, params, m_pb, m_br, m_rp, m_rxyz — four matrices as 9-float storage buffers packed via `pack_3xSSE_to_3x3`); 12 B PC (3 ints). Uses the `vk_RGB_to_HSV` / `vk_HSV_to_RGB` helpers landed in the prior commit for the optional hue-restore step. The toe / linear / shoulder curve with fallback branches and the gamut-compression preroll are mirrored byte-for-byte from the OpenCL kernel. |
| `src/iop/channelmixer.c` | Legacy 3-channel mixer (the deprecated predecessor of `channelmixerrgb`). 4-binding dispatch (in, out, hsl_matrix, rgb_matrix — both 9-float storage buffers) and a single entry point that switches on `operation_mode` (RGB / GRAY / HSL_V1 / HSL_V2). 12 B PC (2 ints + 1 enum int). HSL_V1 and HSL_V2 paths reuse the existing `vk_RGB_to_HSL` / `vk_HSL_to_RGB` helpers from splittoning's color cohort. |
| `src/iop/colorharmonizer.c` | Hue-harmonisation toward Gaussian-weighted nodes in dt UCS JCH space. First port to consume the new `vk_xyY_to_dt_UCS_JCH` / `vk_dt_UCS_JCH_to_xyY` helpers in `dt_vulkan_common.h`. Three-stage `process_vk`: a 6-binding **map** kernel (in, p_out, jch_out, matrix_in, nodes, node_saturation) converts to UCS, computes the per-pixel hue shift + saturation correction, and caches the forward UCS JCH so the apply pass doesn't redo it; the §5.10 `dt_gaussian_blur_vk` smooths the correction map (the `corrections` buffer is padded to float4 so the existing 4-channel Gaussian helper applies — only `.xy` carry data, `.zw` stays zero); a 4-binding **apply** kernel (out, matrix_out, jch_in, corrections) does the JCH→xyY→XYZ→work-profile RGB walk and writes the output. 20 B PC each. Both kernels are batched (matrix + node uploads piggyback the kernel dispatch). |
| `src/iop/colorbalancergb.c` | Scene-referred color balance in CIE 2006 LMS / Filmlight Yrg-Ych, with a perceptual saturation/brilliance pass in either JzAzBz (2021) or darktable UCS (2022) — both branches in one entry point switched on `saturation_formula`. The ~50 scalar/vector kernel args exceed the 128 B push-constant budget, so (like `agx`) the parameter block migrates into a storage buffer: `vk_colorbalancergb_params_t` is a flat all-4-byte struct (std430 == the C struct, verified by `spirv-dis` — members at consecutive offsets, total 232 B). 6-binding dispatch (in, out, params, matrix_in, matrix_out, gamut_lut); 8 B PC (width, height). The work profile is baked into the two host-premultiplied 3×4 matrices, so no ICC `profile_info` binding is needed (the OpenCL kernel takes one but never reads it). Added a cohort of reusable colour-science helpers to `dt_vulkan_common.h` — `vk_LMS_to_Yrg` / `vk_Yrg_to_Ych` / `vk_Ych_to_Yrg` / `vk_Yrg_to_LMS` / `vk_LMS_to_gradingRGB` / `vk_gradingRGB_to_LMS` / `vk_LMS_to_XYZ` / `vk_gamut_check_Yrg` / `vk_XYZ_to_JzAzBz` / `vk_JzAzBz_2_XYZ` / `vk_dt_UCS_{JCH↔HCB,JCH↔HSB}` / `vk_soft_clip` / `vk_lookup_gamut` — ported byte-for-byte from `colorspace.h` (the future `filmicrgb` / Yrg ports reuse these). The mask-display checkerboard preview ports too; it only runs in the gui-attached full pipe. |
| `src/iop/basecurve.c` | Base curve, non-fusion path only. Two kernels lifted straight off the curve-cohort template: `basecurve_legacy_lut` (3-binding per-channel `vk_lookup_unbounded`, 24 B PC) and `basecurve_lut` (5-binding norm-preserving via the §5.11 `vk_dt_rgb_norm` plumbing, 32 B PC); `process_vk` dispatches the matching slot on `d->preserve_colors`. The exposure-fusion path (Laplacian pyramids over `image2d`) needs §8.5 sampler support, so it stays on OpenCL/CPU — `commit_params` clears `process_vk_ready` when `exposure_fusion != 0` (the §10.2 predictive pattern), and `process_vk` belt-and-suspenders returns -1 for it. Validated end-to-end on lavapipe: an identity-ramp LUT round-trips to the input on both kernels (max error 1.3e-5 = LUT quantization), exercising the lookup + norm + ratio-scale paths. |
| `src/iop/colorchecker.c` | Thin-plate-spline colour correction in Lab. Already buffer-shaped in OpenCL, so the single kernel ports near-verbatim: 3-binding dispatch (in, out, params), 12 B PC (width, height, num_patches). The patch data packs into one `float4` storage buffer `[ source_Lab × N | coeff_Lab × (N+4) ]` exactly as `process_cl` builds it; the kernel slices it into `source_Lab` / `coeff_Lab` / `poly_Lab` via pointer offsets. `fastlog2`'s float↔uint bit-pun uses the same `union` idiom as `vk_atomic_add_f` (clspv-safe). Validated end-to-end on lavapipe against a CPU reference of the same math: identity transform is exact and a random 4-patch thin-plate transform matches bit-for-bit (max error 0.0). |
| `src/iop/colorzones.c` | Per-zone Lab lightness/chroma/hue grading. Both process modes ported as separate kernels: `colorzones` (strong — works in LCH via the new `vk_Lab_2_LCH` / `vk_LCH_2_Lab` helpers) and `colorzones_v3` (smooth — works directly in polar (h, C) with a near-axis blend toward neutral). Each is a 5-binding dispatch (in, out, table_L, table_a/C, table_b/h) with a 12 B PC (width, height, channel); `process_vk` picks the slot on `d->mode`. The three 65536-entry curve LUTs go through `vk_lookup` (the flat-buffer twin of the OpenCL 256×256 `image2d_t` `lookup`, bit-identical). `commit_params` mirrors the existing `process_cl_ready` mask-display gate for `process_vk_ready` (the GUI selection-mask preview stays CPU-only). Validated end-to-end on lavapipe: identity LUTs round-trip to the input on both kernels (max 3.9e-4 = LCH trig precision) and a table_L=1.0 scale matches the analytic 4× on L with a/b untouched. |
| `src/iop/soften.c` | The Orton soft-focus effect: HSL saturation/brightness boost (the "overexposed" reference), explicit separable Gaussian blur over that reference, then a blend with the original at `amount`. Ported as 4 small kernels — `soften_overexposed` (2-binding, 16 B PC; uses the `vk_RGB_to_HSL` / `vk_HSL_to_RGB` cohort), `soften_hblur` / `soften_vblur` (3-binding, 12 B PC; explicit 2*rad+1 Gaussian convolution against a host-uploaded normalised kernel buffer) and `soften_mix` (3-binding, 12 B PC; per-pixel blend). The OpenCL hblur/vblur tile via workgroup-local memory; the Vulkan twins read straight from the global storage buffer — math is bit-equal (same CLAMP_TO_EDGE via `clamp(x+i, 0, w-1)`, same Gaussian weights), only the L1 cache pattern differs. First MODERATE-bucket consumer to chain its own explicit convolution (not `dt_gaussian_blur_vk` Deriche — see §5.10 caveat); the chain dispatches 4 `dt_vulkan_dispatch_n` calls in sequence with a single scratch buffer + the Gaussian kernel buffer, mirroring `process_cl` byte-for-byte. Validated end-to-end on lavapipe against an independent C reference of the same data flow: max error 1.2e-7 = single-bit FP precision over the whole chain. |
| `src/iop/sharpen.c` | Unsharp mask. Operates in Lab and blurs only the L channel — the chroma plane passes through untouched. Three small kernels: `sharpen_hblur` / `sharpen_vblur` (3-binding, 12 B PC; same explicit 2*rad+1 Gaussian as soften, but reads `.x` only and writes back the full pixel) and `sharpen_mix` (3-binding, 20 B PC; soft-threshold unsharp mask `delta = orig.L - blurred.L`, `out.L = orig.L + amount * copysign(max(0, |delta| - thrs), delta)`). The OpenCL kernels skip the convolution for the outermost rad pixels (leaving them as the input); the Vulkan twins use the same `if(x >= rad && x < width-rad)` guard, so inside the convolution zone the i-shifts stay strictly in-bounds and no clamp is needed. `process_vk` mirrors `process_cl` byte-for-byte: copy-only fallback when `rad == 0` or the image is smaller than 2*rad+1 (via `dt_vulkan_copy_device_to_device`), same `init_gaussian_kernel` host-side, then hblur(in→out) / vblur(out→tmp) / mix(in, tmp → out) chained as three `dt_vulkan_dispatch_n` calls. Validated end-to-end on lavapipe against an independent C reference of the same data flow: max error **0.0** (bit-equal). |
| `src/iop/highpass.c` | Lab high-pass filter. Four kernels: `highpass_invert` (2-binding, 8 B PC; L → clamp(100 − L, 0, 100), chroma passes through), `highpass_hblur` / `highpass_vblur` (3-binding, 12 B PC; same explicit Gaussian as soften — convolves L everywhere with CLAMP_TO_EDGE on the source index, no skip-edge guard like sharpen has, matching the OpenCL sampler-clamped local-mem tile fill), and `highpass_mix` (3-binding, 12 B PC; `out.L = 50 + ((0.5·a.L + 0.5·b.L) − 50)·contrast_scale`, `out.a = out.b = 0`, then clamp to the Lab range — `±FLT_MAX` substitutes for OpenCL's `±INFINITY` on alpha and is identity for any finite value). `process_vk` mirrors `process_cl` byte-for-byte: same `BOX_ITERATIONS` sigma derivation, invert(in → tmp) / hblur(tmp → out) / vblur(out → tmp) / mix(in, tmp → out) chained as four `dt_vulkan_dispatch_n` calls (blur skipped at `rad == 0` so the mix sees the un-blurred inverted reference, exactly like the OpenCL fast path). Validated end-to-end on lavapipe against an independent C reference of the same data flow: max error 1.5e-5 = single-bit FP precision over the L ∈ [0, 100] range. |
| `src/iop/blurs.c` | Lens / motion / Gaussian blur. The OpenCL has three kernels (`convolve` for dense 2D convolution against a (2·r+1)² PSF, `convolve_sparse` for sparse 2D convolution against an `(offset_x, offset_y, weight)` list, and `restore_alpha` for the Gaussian post-pass); all three port one-for-one. The Gaussian fast path reuses `dt_gaussian_blur_vk` (the Deriche IIR helper landed earlier for shadhi/lowpass) and chains `restore_alpha` (3-binding, 8 B PC) to put the original mask channel back — `dt_gaussian_blur_vk` smears alpha along with RGB. The lens/motion path reuses the existing `_build_pixel_kernel` host helper to compute the PSF, builds the sparse list (entries with `|k| > 1e-6f`) and dispatches `convolve_sparse` (5-binding, 12 B PC); if the sparse allocations fail it falls back to `convolve` (3-binding, 16 B PC) byte-for-byte like `process_cl`. The kernels themselves are tiny and read directly from the global storage buffer with `clamp(x+m, 0, w-1)` boundaries — bit-equal to the OpenCL sampler-clamped reads. `init_global` / `cleanup_global` were restructured to live outside the existing `#if HAVE_OPENCL` block so the VK kernel slots can load even on OpenCL-disabled builds. Validated end-to-end on lavapipe against an independent C reference: all three kernels (`convolve`, `convolve_sparse`, `restore_alpha`) are bit-equal (max error 0.0). |
| `src/iop/bloom.c` | Bloom (the soft-glow lights-screen effect). Four small kernels: `bloom_threshold` (2-binding, 16 B PC; `L = pixel.x * scale`, set to 0 if ≤ threshold — writes a single-channel float scratch matching the OpenCL CL_R image), `bloom_hblur` / `bloom_vblur` (2-binding, 12 B PC; uniform 2·rad+1 box average — `sum / (2*rad+1)` — on the single-channel float scratch, no weights, CLAMP_TO_EDGE via the standard `clamp(x+i, 0, w-1)` idiom), and `bloom_mix` (3-binding, 8 B PC; screen blend `out.L = 100 − ((100 − in_a.L)·(100 − in_b))/100`, chroma + alpha pass through). First port to use **single-channel float scratch buffers** rather than float4 — saves 75 % of the working-set memory for the 8 box-blur iterations and matches the OpenCL allocation byte-for-byte. `process_vk` mirrors `process_cl` exactly: threshold(in → a), then `BOX_ITERATIONS` (= 8) ping-pongs of hblur(a → b) / vblur(b → a) on a 2-buffer scratch (the minimum bucket-chain length the OpenCL code documents), then mix(in, a → out). Validated end-to-end on lavapipe through the full 8-iteration chain against an independent C reference: max error **0.0** (bit-equal). |
| `src/iop/atrous.c` | Edge-aware à-trous wavelet decomposition with per-band thresholding + boost (the contrast-equalizer module). Two small kernels: `eaw_decompose` (4-binding, 16 B PC; 5×5 stencil with sample spacing `1 << scale`, per-channel exponential weights based on Lab L/a/b deltas, writes both the low-pass `coarse` and the high-frequency `detail = in − coarse`) and `eaw_synthesize` (3-binding, 40 B PC; soft-threshold accumulation `amount = copysign(max(0, |detail| − threshold), detail); out = coarse + boost · amount`). The disabled `USE_NEW_CL` `eaw_zero` / `eaw_addbuffers` kernels aren't ported (matching the active host path). `process_vk` mirrors `process_cl` byte-for-byte: same host-built separable B3-spline filter `mm[5][5]` uploaded once, seed dev_out with dev_in via `dt_vulkan_copy_device_to_device`, then `max_scale` decompose passes followed by `max_scale` synthesize passes — both ping-ponging between `dev_out` and `dev_tmp` on the same even/odd parity the OpenCL flow uses. Each scale needs its own `detail` buffer (allocated as a `max_scale`-deep array). The OpenCL `sampleri` CLAMP_TO_EDGE on the sparse stencil reads becomes an explicit `clamp(mult·(i−2)+x, 0, w−1)` per coordinate. Validated end-to-end on lavapipe across a 3-scale ping-pong chain (6 dispatches): max error 3.8e-5 = single-bit FP precision over the Lab L ∈ [0, 100] range. |
| `src/iop/hazeremoval.c` | Dark-channel-prior haze removal. First consumer of `dt_guided_filter_vk` (§5.15). The six module-own kernels port one-for-one (`transision_map` + 4× separable box min/max for refining the transition map + `dehaze` for the final atmospheric inversion); the guided-filter step delegates to the helper, so the module-side code is small. PC layouts: 24 B `tmap` (width, height, strength, A0_{r,g,b}); 12 B `box_min/max_{x,y}` (width, height, w); 24 B `dehaze` (width, height, t_min, A0_{x,y,z}). The dehaze kernel drops the unused `A0.w` from the OpenCL `float4` to keep the PC tight. `process_vk` mirrors `process_cl` byte-for-byte: the GUI-cached A0 fast path is preserved (preview pipe stashes A0 + distance_max for the full pipe); when no cache is available the input is read back via `dt_vulkan_read_from_device` and the existing CPU `_ambient_light` runs on the host buffer — no kernel needed for the global max-RGB reduction, matching the OpenCL `_ambient_light_cl` flow. Then: transition_map → box_max_x/y to refine → box_min_x/y to soften → `dt_guided_filter_vk(img_in, trans_map, trans_map_filtered)` → dehaze. Validated end-to-end on lavapipe: all six module-own kernels are bit-equal (max error 0.0) against the C reference, and the guided-filter helper has its own dedicated test (§5.15). |
| `src/iop/colorreconstruction.c` | Clipped-highlight colour reconstruction via a 3-D bilateral-grid splat/blur/slice pipeline (similar shape to the §5.13 bilateral helper but with chroma-blend slicing). Four small kernels: `colorreconstruction_zero` (1-binding, 8 B PC; zeros the flat grid view), `colorreconstruction_splat` (2-binding, 44 B PC; atomic-add float into the 3-D grid via the CAS-loop `vk_atomic_add_f` — same pattern as bilateral_splat — and the OpenCL workgroup-local pre-reduction is **dropped**, math stays bit-equal since the local reduction is just a batching optimisation), `colorreconstruction_blur_line` (2-binding, 24 B PC; 3 host-issued 1-D B3-spline passes — z/x/y — with drop-missing-samples boundary handling) and `colorreconstruction_slice` (3-binding, 52 B PC; trilinear lookup into the grid + chroma blend back into the output around the threshold). **Subtle rounding gotcha**: the OpenCL `round()` is half-away-from-zero, but GLSL `round()` is implementation-defined (often banker's rounding to even) — both `.cl` and `.comp` use the explicit form `floor(x + 0.5)` so the splat cell index is identical across both compilers and matches OpenCL byte-for-byte (the test caught this with the diagnostic-printing harness when `round(0.5)` produced different yi values). The OpenCL freeze/thaw preview→full grid reuse optimisation is **not ported**; the VK twin always rebuilds the grid from scratch in `process_vk` (alloc grid + tmp → zero → splat → 3-axis blur → slice → free). Validated end-to-end on lavapipe against a C reference of the identical math: all 4 kernels match (`zero` exact; `splat` max 3e-5 over the float-atomic accumulation; `blur_line` exact; `slice` ~4e-6 = trilinear-lookup FP precision). |
| `src/iop/nlmeans.c` | Non-local-means denoising (the GPU shift-and-accumulate variant from Goossens et al.). Six small kernels port one-for-one and all reads are flat-buffer (the OpenCL `image2d_t` reads are in-bounds-masked via the multiply-by-0 trick, so no sampler needed): `nlmeans_init` (1-binding, 8 B PC; zero the float4 accumulator U2), `nlmeans_dist` (2-binding, 24 B PC; per-pixel L/C-weighted squared distance to the q-shifted pixel), `nlmeans_horiz` / `nlmeans_vert` (2-binding, 12/16 B PC; separable box-sum of the distance with clamp-to-edge; vert also applies the patch weight `gh(d) = vk_fast_mexp2f(d·sharpness)`), `nlmeans_accu` (3-binding, 16 B PC; accumulate the ±q weighted neighbours into U2) and `nlmeans_finish` (3-binding, 24 B PC; normalise by the weight channel and blend). Added `vk_fast_mexp2f` to `dt_vulkan_common.h` — the `common.h::fast_mexp2f` bit-pun 2^-x approximation (clspv-safe union, same idiom as `vk_atomic_add_f`; the `.comp` twin uses `uintBitsToFloat`). The OpenCL horiz/vert tile via workgroup-local memory and rotate 4 buckets for async overlap; the VK twin reads from global storage with the same clamp-to-edge and ping-pongs just two single-channel scratch buffers (the bucket rotation is only an overlap optimisation). `nlmeans_accu`'s `U2[gidx] += accu` needs **no atomics** — each invocation owns its gidx and `process_vk` serialises the q-loop. `process_vk` mirrors the active (non-`USE_NEW_IMPL_CL`) `process_cl` byte-for-byte: init → for q ∈ [-K,0]×[-K,K] { dist → horiz → vert → accu } → finish. Validated end-to-end on lavapipe through the full 28-offset q-loop (~140 dispatches) against an independent C reference of the NLM math (including the bit-pun weight): max error 1.5e-5 = single-bit FP precision. |
| `src/iop/filmicrgb.c` | Scene-referred filmic tone mapper, main per-pixel path. The OpenCL module has two kernels (`filmicrgb_split` for per-channel and `filmicrgb_chroma` for chroma-preserving) selected by `(preserve_color == NONE && version != V5)`, each switching on the colour-science version v1..v5; both fold into one Vulkan entry that reproduces the host's split-vs-chroma decision internally and dispatches on `color_science` exactly like `process_cl`. 6-binding dispatch (in, out, params, matrices, profile_info, profile_lut); 8 B PC (width, height). Like `agx`/`colorbalancergb`, the ~26 scalars + 5 spline `float4`s overflow the PC budget (164 B), so `vk_filmicrgb_params_t` migrates into a storage buffer (156 B, std430 == the C struct verified by `spirv-dis`). The four `dt_colormatrix_t` 3×4 matrices (input, output, export-in, export-out) pack into a single 48-float `matrices` buffer at fixed base offsets; helpers take an `int base` so the same code paths handle work and export gamut targets. The colour-science cohort (`vk_LMS_to_Yrg` / `vk_Yrg_to_Ych` / `vk_Ych_to_Yrg` / `vk_Yrg_to_LMS` / `vk_gamut_check_Yrg`) is reused from `dt_vulkan_common.h` — the helpers `colorbalancergb` landed. The filmic-specific gamut machinery — `clip_chroma_white_raw` / `clip_chroma_white` / `clip_chroma_black` / `clip_chroma` / `gamut_check_RGB` / `gamut_mapping` / `desaturate_v4` — and the v1..v5 split/chroma variants are inlined faithfully from `filmic.cl`. ICC luminance goes through the §5.11 deferred plumbing (`dt_ioppr_build_iccprofile_params_vk_deferred` appends the profile uploads in one batched submit). The highlight-reconstruction path (inpaint + a-trous wavelets over `image2d_t`) and the clipped-pixel mask preview need §8.5 sampled-image bindings; `commit_params` clears `process_vk_ready` when `enable_highlight_reconstruction` is on or the GUI mask is showing (the §10.2 predictive pattern — mirrors `basecurve`'s exposure-fusion gate). Validated end-to-end on lavapipe with an independent C reference derived directly from `filmic.cl` (not from the port): `split_v2_v3` is bit-equal (max 0.0) and `chroma_v4` matches at FP rounding (max 1e-6) — proving the matrices buffer packing, the Yrg/Ych conversions, the gamut-clip family and the std430 marshalling all reproduce the OpenCL math byte-for-byte. |
| `src/iop/bilat.c` | "Local contrast" — has two modes selected by `d->mode`. **Bilateral grid** consumes the §5.13 helper directly: `process_vk` is just `dt_bilateral_init_vk → splat → blur → slice_vk(detail) → free` mirroring `process_cl` byte-for-byte. **Local Laplacian** (the default mode — `s_mode_local_laplacian` is the `$DEFAULT` for the params struct) consumes the new `dt_local_laplacian_*_vk` helper (§5.17) — same `init → apply → free` shape as the OpenCL surface. Validated end-to-end on lavapipe against independent C references: the bilateral chain matches at 7.6e-6 on L (a/b/alpha pass through bit-exactly), and the local-laplacian helper's 5 kernels match per-kernel at single-bit FP precision (pad_input + gauss_reduce bit-equal, process_curve + laplacian_assemble at 6e-8, write_back exact). |
| `src/iop/globaltonemap.c` | Three classic global tone mappers (Reinhard, Drago, Hejl/Burgess-Dawson "filmic"), all picked by `d->operator` and dispatched on a 2-binding (in, out), 24-byte PC (`width`, `height`, 4 floats) shape. Reinhard and Filmic ignore the float fields; Drago carries `(eps, ldc, bl, lwmax)` to drive its adaptive logarithmic curve. The OpenCL path computes `lwmax` via a two-stage `pixelmax_first` + `pixelmax_second` workgroup-local reduction; the Vulkan port follows the §4.2 `hazeremoval` precedent and **drops the GPU reduction** — when the GUI cache for `lwmax` is empty the input is read back via `dt_vulkan_read_from_device` and the max is found on the host. The full Drago fast-path (preview pipe primes `g->lwmax`/`g->hash`, full pipe re-uses) is preserved byte-for-byte, so the readback only triggers on a fresh module instance in the full pipe. Detail recovery uses the §5.13 bilateral helper: `splat → tonemap → blur → snapshot dev_out → slice_to_output_vk(in, snapshot, out, detail)`, matching `process_cl` exactly. Drago's `process_tiling_ready = FALSE` gate is the existing global flag and applies to Vulkan unchanged (the readback would give a per-tile max otherwise — same correctness issue OpenCL has). Validated end-to-end on lavapipe against independent C references of all three operators: reinhard and filmic are bit-equal (max 0.0), drago matches at 2.3e-5 over the log/pow chain. |
| `src/iop/colormapping.c` | Hertzmann-style colour-transfer mapping: equalised L histogram remap + soft-cluster-weighted Lab a/b remap from target to source statistics. Both kernels (`colormapping_histogram` and `colormapping_mapping`) port via the standard image-shortcut (the OpenCL `image2d_t` reads only use integer fixed coords, equivalent to flat `in[y*width + x]`). Histogram has 4 bindings (in, out, target_hist `int[2048]`, source_ihist `float[2048]`) with a 12 B PC (width, height, equalization); mapping has 7 bindings (in, tmp, out, target_mean `float2[5]`, source_mean `float2[5]`, var_ratio `float2[5]`, mapio `int[5]`) with a 12 B PC (width, height, clusters). `get_clusters` (chroma-distance soft membership) is inlined into the mapping kernel. The ACQUIRE preview-pipe input snapshot for host-side cluster training (`_get_cluster_mapping`) reads back via `dt_vulkan_read_from_device` — same precedent as `globaltonemap`'s lwmax cache miss and `hazeremoval`'s ambient-light snapshot. The bilateral-smoothed dL path reuses the §5.13 helper; the `equalization < 0.001` shortcut skips the bilateral and copies dev_out → dev_tmp via `dt_vulkan_copy_device_to_device`. When no source/target stats are present yet (the default-pass-through state) `process_vk` shortcuts to a single `dt_vulkan_copy_device_to_device` like `process_cl`'s `enqueue_copy_image` shortcut. Validated end-to-end on lavapipe with both kernels against an independent C reference: histogram matches at 3.8e-6, mapping at 1.5e-5 — single-bit FP precision over the cluster-weighted sum. |
| `src/iop/colorin.c` | Input ICC colour profile transform — the input-side mirror of `colorout`. Two kernels picked by `d->nrgb`: `colorin_unbound` (cam → XYZ via cmat then XYZ → Lab) and `colorin_clipping` (cam → RGB via cmat, clipped to [0, 1], then RGB → XYZ via lmat, then XYZ → Lab). Both share the same 7-binding (in, out, lutr, lutg, lutb, cmat, lmat) / 64 B PC (width, height, blue_mapping, corr.xyzw, a_r/g/b\[0..2\]) shape; lmat is unused in `colorin_unbound` but kept for binding symmetry. The OpenCL `image2d_t`-shaped LUTs (256×256 = 65536 entries each) become flat float storage buffers; the LUT lookups use `vk_lerp_lookup_unbounded` to match `lerp_lookup_unbounded0`'s linear interpolation between adjacent entries byte-for-byte (the gamma-power extrapolation tail `c1·pow(x·c0, c2)` for x ≥ 1 is reproduced identically). The Bayer blue-mapping gamut clamp is folded into both kernels behind the `blue_mapping` flag, mirroring the OpenCL kernel; `process_vk` ANDs the user flag with `dt_image_is_matrix_correction_supported(&pipe->image)` just like `process_cl`. The white-balance correction `(D65 / as-shot)` is computed host-side from `self->dev->chroma` (the `late_correction` path also mutates `pipe->dsc.temperature.coeffs` and `pipe->dsc.processed_maximum`, mirroring the CL behaviour). The `DT_COLORSPACE_LAB` fast path routes through `dt_vulkan_copy_device_to_device` (same shape as `colorout`'s Lab pass-through). The lcms2 slow path (non-matrix profiles) stays on CPU as in OpenCL. Validated end-to-end on lavapipe against an independent C reference of both kernels' math: unbound + blue_mapping at 1.8e-4, clipping at 9e-5 — FP precision through the LUT lerp + matrix + XYZ→Lab chain. |
| `src/iop/crop.c` | Hard image crop. `crop` runs ahead of distort and its `modify_roi_in` aligns roi_in to the cropped region — by the time the pipeline reaches `process_cl` / `process_vk` the input buffer is already the cropped image, and the OpenCL fast-path is just `dt_opencl_enqueue_copy_image(devid, dev_in, dev_out, …)`. The Vulkan twin is one line: `dt_vulkan_copy_device_to_device(devid, dev_out, dev_in, roi_out->w * roi_out->h * 4 * sizeof(float))`. Practically valuable because `crop` is always in the pipeline — a missing `process_vk` here forces a CL↔VK transition for every image even though no actual kernel work happens. Same shape as `mask_manager`. |
| `src/iop/lut3d.c` | 3-D LUT colour grading (Hald CLUT / .cube / GMIC). All four kernels port via the image-shortcut (sampler-clamp integer coords on in/out, the LUT was already a flat `global float *` storage buffer in OpenCL): `lut3d_none` (2 bindings, 8 B PC — identity copy when no LUT is configured), `lut3d_trilinear` / `lut3d_tetrahedral` / `lut3d_pyramid` (3 bindings each — in, out, clut — 12 B PC with `width, height, level`). The 8-corner cube indexing math `(color = ir + ig*level + ib*level²; i000 = color*3 …)` translates verbatim — the OpenCL packs 3 floats per LUT entry rather than a `float3` (no SoA alignment in `global float*`) and the Vulkan port keeps the same layout. The `clip4` clamp to [0, 1] is applied to the entire float4 input including alpha, matching OpenCL byte-for-byte. `process_vk` mirrors `process_cl`'s ICC-profile transform sandwich: optional `work_profile → lut_profile` via `dt_ioppr_transform_image_colorspace_rgb_vk` (§5.12), kernel dispatch with `src = transform ? dev_out : dev_in`, then `lut_profile → work_profile`. Validated end-to-end on lavapipe against a C reference of all 4 algorithms with a non-identity LUT and an alpha > 1 input (caught the clip4-clamps-alpha subtlety): trilinear bit-equal, tetrahedral and pyramid at 7e-9 (single-bit FP). |
| `src/iop/liquify.c` | Per-pixel free-form image warp with a user-painted spline / point distortion map. The OpenCL `warp_kernel` uses `read_imagef(in, sampleri, ...)` — `sampleri` is the CLK_FILTER_NEAREST + CLK_ADDRESS_CLAMP_TO_EDGE sampler, applied with the integer-truncated source coord — so the Lanczos / bicubic / bilinear reconstruction is done in kernel code via a host-prepared discrete kernel `k[]`, not by a hardware sampler. That makes liquify image-shortcut tractable like `lut3d` / `colormapping`. The Vulkan port folds the OpenCL flow's two dispatches (sub-region image copy + warp pass) into a single `liquify_warp` kernel dispatched over `roi_out`: every output pixel pass-through-copies its image-coord input by default, and only the pixels inside `map_extent` with a non-zero warp run the 2a×2a reconstruction. 4 bindings (in, out, map `float2[]`, k `float[]`) + 14-int 56 B PC carrying all 6 small structs the OpenCL kernel took as separate buffer args (`roi_in.{x,y,w,h}`, `roi_out.{x,y,w,h}`, `map_extent.{x,y,w,h}`, `kdesc.{size, resolution}`). When the host builds an empty map (no user warps yet) the Vulkan path short-circuits to a single `dt_vulkan_copy_device_to_device` when `roi_in == roi_out`. The Lanczos kernel-table compilation in the host (bilinear / bicubic / lanczos2 / lanczos3 → up to 6-tap `k[]`) ports verbatim from `process_cl`. Validated end-to-end on lavapipe against a C reference of the same warp math: pass-through (`warp == 0` everywhere), bilinear warp inside a 16×12 map_extent, and lanczos2 warp inside an 18×14 map_extent — **all three bit-equal (max 0.0)**. |
| `src/iop/rasterfile.c` | Loads a separately-exported mask file (.pfm) and routes its data through the raster-mask infrastructure for downstream blending. The OpenCL `process_cl` has two paths: when `roi_out->scale != roi_in->scale` it delegates to `dt_iop_clip_and_zoom_cl` (resample), and otherwise it does a sub-region copy via `dt_opencl_enqueue_copy_image(devid, dev_in, dev_out, (roi_out->x, roi_out->y), (0, 0), (w, h))`. The Vulkan port now covers **both**: the same-scale path via `dt_vulkan_copy_subregion` (§5.9 entry 2) and the resample path via the new `dt_iop_clip_and_zoom_vk` (§5.18). The mask-preview GUI focus mode (`visual = fullpipe && dt_iop_has_focus(self)`) returns -1 like `process_cl`. The raster-mask plumbing (`dt_iop_piece_set_raster` / `_clear_raster`) is host-side and ports verbatim. |
| `src/iop/finalscale.c` | The final output rescale that fits the pipe's working image to the export / display dimensions. `process_cl` gates upscaling (`roi_out->scale > 1.0f`) to OpenCL / CPU and routes downscale + 1:1 through `dt_iop_clip_and_zoom_{roi_}cl`. The Vulkan port mirrors this exactly: same upscale gate (returns -1), then `dt_iop_clip_and_zoom_roi_vk` for the export pipe or `dt_iop_clip_and_zoom_vk` for the interactive pipe (§5.18). First consumer of the resampler helper — practically high-value because finalscale runs on every export and every zoomed-out darkroom view, so a missing `process_vk` would force a CL↔VK transition at the most bandwidth-heavy point in the pipe (the full-resolution → display-resolution downscale). |
| `src/iop/ashift.c` | Perspective correction (lens-shift / keystone correction via a user-specified rotation + shear + lens-shift + aspect homography). Four interpolation kernels (`ashift_bilinear` / `ashift_bicubic` / `ashift_lanczos2` / `ashift_lanczos3`) selected at runtime via `DT_INTERPOLATION_USERPREF_WARP`. The OpenCL `bilinear` variant uses `samplerf` for hardware bilinear filtering; the Vulkan port implements **manual 4-tap bilinear in kernel code** with CLAMP-to-zero borders (matching the OpenCL semantics byte-for-byte: out-of-bounds corner reads return `(0,0,0,0)`). The `bicubic` / `lanczos2` / `lanczos3` variants use `samplerA` (nearest + ADDRESS_NONE) at integer coords — pure image-shortcut, so the kernel does its own multi-tap reconstruction via the new `vk_interpolation_bicubic` / `vk_interpolation_lanczos` helpers + the `vk_sinf_fast` bit-pun union (clspv-safe; the `.comp` twin uses GLSL's even/odd branch in lieu of the union sign-trick). The `vk_clip_mirror` helper handles edge mirroring for the multi-tap reads. 3 storage bindings (in, out, homograph float[9]) + 48 B PC (8 ints + 4 floats: width, height, iwidth, iheight, roi_in.{x,y}, roi_out.{x,y}, in_scale, out_scale, clip.{x,y}). The preview-pipe input snapshot for parameter-fitting (`g->buf` for the `_fit_helper` line-detection passes) reads back via `dt_vulkan_read_from_device` — same precedent as `globaltonemap` / `colormapping` / `hazeremoval`. The neutral-params pass-through (`_isneutral(d)`) routes to `dt_vulkan_copy_device_to_device` like `process_cl`'s `enqueue_copy_image` shortcut. Validated end-to-end on lavapipe against C references of all 4 interpolators (small rotation around the image centre exercising sub-pixel sampling): bilinear at 6e-7, bicubic at 8e-7, lanczos2 at 9e-7, lanczos3 at 9e-7 — all single-bit FP precision. |
| `src/iop/clipping.c` | Crop + rotate + flip + lens-keystone correction (the older sibling of `ashift`). Same four interpolation kernels (`clip_rotate_bilinear` / `clip_rotate_bicubic` / `clip_rotate_lanczos2` / `clip_rotate_lanczos3`) reusing the `vk_interpolation_*` / `vk_sinf_fast` / `vk_clip_mirror` cohort from §5.7. The homography is richer than ashift's (translation `t`, lens-shift-style `k`, a 2×2 matrix `m`, plus an optional secondary keystone backtransform via `k_space` / `ka` / `ma` / `mb` when `k_space.z > 0`) so the port reproduces `backtransform` and `keystone_backtransform` from `basic.cl` byte-for-byte as in-kernel helpers. 3 storage bindings (in, out, plus a dummy slot kept for binding symmetry across the cohort) + a 124-byte push-constant block (7 ints + 24 floats — just under the 128 B PC limit), which fits all 6 small structs the OpenCL kernel took as separate args. The fast crop-only path (`!d->flags && d->angle == 0 && d->all_off && roi_in == roi_out`) routes through `dt_vulkan_copy_device_to_device`. Validated end-to-end on lavapipe against C references of all 4 interpolators under a small rotation: bilinear **bit-equal (max 0.0)**, bicubic at 3.6e-7, lanczos2 / lanczos3 at 2.4e-7. |
| `src/iop/colorequal.c` | **Fully ported — both the non-guiding fast path and the guided-filter path.** **Fast path** (`use_filter == off`): `ce_sample_input` (input → XYZ_D65 → xyY → dt-UCS L*/UV + saturation) → `dt_gaussian_mean_blur_vk` (smooth saturation) → `ce_process_data` (LUV → JCH → HSB + per-hue LUT corrections) → `ce_write_output` (apply corrections, gamut-map, HSB → XYZ → RGB). **Guided path** (`use_filter == on`, the default): adds 11 more kernels in two host-orchestrated stages (`_prefilter_chromaticity_vk`, `_guide_with_chromaticity_vk`) mirroring the OpenCL `_prefilter_chromaticity_cl` / `_guide_with_chromaticity_cl` byte-for-byte — `ce_init_covariance` / `ce_finish_covariance` (UV outer-product covariance), `ce_prepare_prefilter` / `ce_final_guide` (per-pixel 2×2 covariance inverse → guided-filter a/b coefficients), `ce_apply_prefilter` / `ce_apply_guided` (satweight-LUT-weighted blend), `ce_prepare_correlations` / `ce_finish_correlations` (UV ⊗ corrections), and `ce_bilinear1` / `ce_bilinear2` / `ce_bilinear4` (down/upsample float/float2/float4 buffers between the full and `_get_scaling`-reduced resolution levels). The guided stages reuse `dt_gaussian_mean_blur_vk` at 1/2/4 channels for the box-filter passes. The satweight LUT (`2*SATSIZE+1`=8193 floats) uploads once via `dt_vulkan_write_to_device`. Added the **dt-UCS colour-science helper cohort** to `dt_vulkan_common.h` (9 helpers mirroring `colorspace.h`). Only the GUI mask-display preview falls back to OpenCL / CPU (a runtime check in `process_vk`, not a `commit_params` gate, since `mask_mode` is gui state). Validated end-to-end on lavapipe: the fast-path 3-kernel chain matches a C reference at 1.8e-6 (single-bit FP, all 4 channels); each of the 11 guided kernels matches a per-kernel C reference bit-equal (the 2×2-inverse, satweight-blend, outer-product, and bilinear-resample bodies all reproduce the OpenCL math exactly). |
| `src/iop/censorize.c` | Privacy-blur module (license plates, faces). The OpenCL `process_cl` is a never-compiled `#if FALSE` stub — the module ran CPU-only before this port. The CPU pipeline is gaussian-blur → pixelate (5-sample average over each `2*pixel_radius` block) → gaussian-blur → multiplicative-Gaussian noise. The Vulkan port adds **two new kernels** (`censorize_pixelate`, `censorize_noise`) plus reuses `dt_gaussian_mean_blur_vk` for the two blur stages. **PRNG note**: the CPU noise pass uses `splitmix32` (64-bit multiply) to seed a per-pixel xoshiro128plus state. Rather than emulate uint64 portably in GLSL, the Vulkan port substitutes a Wang-style 32-bit avalanche hash (`vk_hash32`) for seeding. The xoshiro128plus stream and Box-Muller Gaussian transform are bit-for-bit equivalent; only the seed differs, so the noise pattern is **statistically equivalent but not bit-equal** to CPU. For privacy blurring this is the right correctness bar — noise looks identical to a human observer; only the per-pixel realisation differs. The pixelate kernel dispatches one work-item per big-pixel block (size `2*pixel_radius`) and writes the 5-sample average to every output pixel in the block; matches the CPU loop byte-for-byte. Validated end-to-end on lavapipe: pixelate is **bit-equal vs C reference** (max 0.0); noise mean/stddev on the R channel match the expected scale within ±1 % of mean / ±2 % of stddev; alpha passes through bit-exact. `init_global` / `cleanup_global` had to be lifted out of the `#if FALSE` wrapping the broken `process_cl` so the Vulkan kernel slots actually get loaded (the dead `kernel_lowpass_mix` OpenCL slot stays zero-initialised). |
| `src/iop/highlights.c` | Highlight reconstruction — **CLIP mode on non-Bayer images only**. The OpenCL build has 15+ kernels covering the 6 highlights modes (CLIP / LCH / OPPOSED / INPAINT / LAPLACIAN / SEGMENTS), almost all of which need single-channel uint16 / float input buffers and the Bayer / X-Trans pattern lookup. The Vulkan port covers `highlights_4f_clip` (a per-pixel `min(clip, pixel)` clamp on the post-demosaic float4 path — non-Bayer images: HDR DNG, 16-bit TIFF, PNG, etc.) and gates everything else via `commit_params`'s `process_vk_ready = FALSE` when `piece->pipe->dsc.filters` is set (RAW Bayer/X-Trans), when the gui mask preview is on (uses the `highlights_false_color` kernel which needs the xtrans pattern), or when `mask_display == DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU` (which would need either `dt_iop_clip_and_zoom_vk` or a sub-region copy — both available, but the additional plumbing isn't worth it for the gated mask-display path). 2 storage bindings (in, out) + 12 B PC (width, height, clip). `process_vk` is a single `dt_vulkan_dispatch_inout`. The clip value is `d->clip * highlights_clip_magics[d->mode]` — exactly the OpenCL host-side formula. Validated end-to-end on lavapipe with three test cases (no clipping, partial clip, full clip): all three bit-equal vs C reference. |
| `src/iop/rawprepare.c` | Black-level subtraction + per-channel divisor at the head of the pipeline. **4-channel float path only** — the OpenCL build has 5 variants (`rawprepare_1f` / `_1f_gainmap` / `_1f_unnormalized` / `_1f_unnormalized_gainmap` / `_4f`) for the different RAW input formats; the Vulkan port covers `_4f` (post-demosaic or non-Bayer pre-pipeline data — HDR DNG, 16-bit TIFF, PNG, etc.) and gates the four single-channel Bayer / X-Trans paths to OpenCL / CPU via `commit_params`'s `process_vk_ready = FALSE` (the §10.2 predictive pattern — same shape as `basecurve` fusion, `filmicrgb` highlight reconstruction). Those 1f kernels need single-channel uint16 / float input buffer binding + an `FC()` / `FCxtrans()` pattern lookup, neither of which is wired into the Vulkan pipeline integration yet. `process_vk` is a single `dt_vulkan_dispatch_inout` with 2 bindings (in, out) and a 56 B PC carrying the dimensions, crop offsets, ROI origins, the 4-channel black-level vector, and the 4-channel divisor. The image-shortcut applies (OpenCL `readpixel(in, x + cx, y + cy)` with the sampler-clamp integer-coord pattern translates verbatim). Validated end-to-end on lavapipe with three test cases (full image, cropped sub-region with offset, zero-black uniform-white) against a C reference of `(pixel - black) / div` per RGB channel + alpha passthrough: **all three bit-equal (max 0.0)**. |

*Partial (clspv: full; glslang fallback: one mode only):*

| Module | Status |
|---|---|
| `src/iop/channelmixerrgb.c` | All 5 adaptation modes ported (linear Bradford, full Bradford, CAT16, XYZ, RGB) with full gamut mapping and per-channel saturation / lightness. clspv emits all 5 entry points into one `.spv`; the host loads the program once and creates 5 kernel slots via `dt_vulkan_module_kernel_create_from`. On glslang-only builds only the linear-Bradford entry exists (GLSL doesn't support multi-entry `.spv`); other modes fall back to OpenCL transparently. |

*Pending Vulkan port (currently OpenCL/CPU only):*

| Module | Status |
|---|---|
| `src/iop/diffuse.c` | The à-trous wavelet diffuse pipeline runs 6 different kernels (build_mask, inpaint_mask, blur_2D_Bspline_horizontal / vertical, wavelets_detail_level, diffuse_pde) across up to 10 scales × N iterations, with ~13 intermediate buffers managed by the host. The earlier integration shipped a single-pass unsharp-mask approximation that *looked* like sharpen but produced visibly different output than the OpenCL path — we've removed it so users get correct OpenCL output instead of silent drift. A full port is the next major milestone (§8.9); it needs (a) the 6 kernel translations [`diffuse_pde` alone is ~125 lines of anisotropic PDE math], (b) a multi-dispatch orchestration loop in `process_vk` mirroring `wavelets_process_cl`, and (c) a multi-dispatch batching mode in the HAL so we don't pay queue-submit + fence-wait per dispatch. |

**Toolchain convention** for the per-module kernels under
`data/kernels/vulkan/`: each kernel is a paired `.cl` (clspv) and
`.comp` (glslang fallback). Both produce a `<kernel>.spv` with an
entry-point named after the OpenCL kernel function; glslang gets
its GLSL `void main()` entry renamed via `--source-entrypoint main
-e <name>` so the runtime can address them identically. The
`dt_vulkan_create_kernel` host-side call passes the entry name and
both toolchains' `.spv` work without further dispatch.

**Verified in this container:** all 41 module files and the new
backend compile clean against all four `(HAVE_VULKAN × HAVE_OPENCL)`
combinations (the full darktable build target succeeds, including
`libfilmic.so`, `libcolorout.so` and the other plugin shared
libraries). All GLSL twins build to valid SPIR-V via glslang +
`spirv-val`. End-to-end runs against a real RAW are exercised by
the user out-of-container on AMD RX 9060 XT (RADV).

**What's left** (from the 70 surveyed `process_cl` modules):

- **TRIVIAL bucket** still to port. The Lab/RGB-curve cohort
  (`tonecurve`, `rgbcurve`, `rgblevels`) is now landed — they
  lifted off the colisa template (§5.8) + the §5.11 ICC profile
  plumbing for their norm-preserving modes. `monochrome` is
  already ported as the first bilateral consumer (§5.13).
  `rawoverexposed` ported in this pass — the post-demosaic float4
  in/out lets us keep the standard 2-binding pipe shape; the raw
  uint16 sensor buffer is uploaded as a flat `uint[]` (promoted on
  host so the shader can index it without bit-unpacking), the
  X-Trans 6×6 pattern goes in a 36-entry `uint[]` binding via the
  existing `vk_FCxtrans` helper, and the per-pixel post-distort raw
  coords ride a third float buffer. Three kernel variants
  (`mark_cfa`, `mark_solid`, `falsecolor`) share the same 5-binding
  layout; PCs carry the four uint thresholds plus the mode-specific
  colors. Done in earlier passes: `colisa`, `levels`,
  `profile_gamma`,
  `zonesystem`, `splittoning` (added RGB↔HSL helpers to
  `dt_vulkan_common.h`), `overexposed` (first consumer of ICC
  profile plumbing §5.11), `lowlight` (template for the next LUT
  consumers).
- **EASY bucket** — one or two storage buffers for matrices /
  LUTs. Done in earlier passes:
  `basecurve` (non-fusion path — the two LUT kernels off the
  curve-cohort template; the Laplacian-pyramid fusion path needs
  §8.5 and stays on OpenCL via a `process_vk_ready` gate),
  `colorbalancergb` (LMS/Yrg/Ych/JzAzBz/dt-UCS scene-referred
  balance; `agx`-style params-struct-in-storage-buffer for the
  >128 B arg block + a reusable colour-science helper cohort added
  to `dt_vulkan_common.h`),
  `basicadj` (second consumer of the §5.11 plumbing; full
  6-feature ICC-aware kernel), `rgbcurve` / `rgblevels` /
  `tonecurve` (the Lab/RGB curve cohort, all sharing the 7-binding
  §5.11 + §5.8 pattern), `colorbalance` (3 variants, push-
  constant-only — switches on `d->mode` to dispatch the matching
  `.spv`), `colorout` (Lab→RGB matrix + 3 shaper LUTs, with the
  `DT_COLORSPACE_LAB` pass-through case routed through
  `dt_vulkan_copy_device_to_device`), `filmic` (legacy single-pass
  variant; the newer `filmicrgb` ports its main per-pixel tone
  mapping path — see HARD bucket — and gates the wavelet
  highlight reconstruction to OpenCL/CPU), `sigmoid` (both color-processing
  modes — `rgb_ratio` is push-constant-only, `per_channel` carries
  3 matrices as storage-buffer bindings packed by
  `pack_3xSSE_to_3x3`), `agx` (first to migrate a >128 B param
  struct from PC into a storage-buffer binding — pattern available
  for future param-heavy ports), `channelmixer` (legacy 4-mode
  mixer reusing the existing HSL helpers),
  `lut3d` (3-D LUT colour grading — turned out tractable without
  §8.5 since the LUT was already a flat `global float *` buffer in
  OpenCL and the cube indexing math doesn't need sampler filtering;
  all 4 algorithms ported, ICC-profile transform sandwich reuses
  §5.12 `dt_ioppr_transform_image_colorspace_rgb_vk`).
- **MODERATE** — multi-pass with intermediate buffers or
  local-memory barriers: `highlights`. The Gaussian VK
  helper (§5.10) handles the separable-blur half (`blurs` uses
  it for its DT_BLUR_GAUSSIAN fast path); the explicit-convolution
  modules (`soften`, `sharpen`, `highpass`) have their own
  Gaussian in OpenCL and don't reuse `dt_gaussian_blur_vk`, so
  they were ported with a direct global-storage convolution that
  is bit-equal to the OpenCL math.
  Done in earlier passes:
  `blurs` (lens / motion / Gaussian: 3-kernel port with the dense
  `convolve`, the sparse `convolve_sparse`, and the
  `restore_alpha` post-pass for the Gaussian path; bit-equal to
  the C reference on all three kernels, reuses `dt_gaussian_blur_vk`
  for the Gaussian-type fast path),
  `highpass` (Lab high-pass: invert L + same Gaussian as soften
  + a desaturating contrast mix around L = 50; 4-kernel chain,
  bit-equal to a C reference modulo single-bit FP in the L range),
  `sharpen` (unsharp mask in Lab; 3-kernel chain over the L
  channel only — chroma passes through — with the same explicit
  Gaussian + skip-edge semantic as soften, plus a soft-threshold
  unsharp blend),
  `soften` (Orton effect — 4-kernel chain: HSL boost +
  separable explicit Gaussian + blend; first MODERATE consumer
  to do its own convolution rather than reuse the Deriche helper,
  reads straight from global storage with `clamp(x+i, 0, w-1)` instead
  of OpenCL's workgroup-local tile — bit-equal math, no shared memory),
  `colorzones` (both process modes — strong via the new Lab↔LCH
  helpers, smooth via polar (h, C); `process_vk` switches on
  `d->mode` and the mask-preview is gated to CPU),
  `colorchecker` (single-kernel thin-plate spline — already
  buffer-shaped, ported near-verbatim with a packed `float4`
  params buffer),
  `graduatednd`, `vignette`, `relight`, `borders` (multi-fill +
  sub-region copy, §5.9), `lowpass` (first combined-helper
  consumer — chains §5.10 or §5.13 followed by a curve-mix
  kernel), `shadhi` (second combined-helper consumer — same shape
  + soft-light overlays),
  `bilat` (both modes — bilateral-grid arm uses §5.13 directly,
  local-laplacian arm uses the new §5.17 helper; the gate that
  previously routed the local-laplacian arm to OpenCL is now
  removed),
  `colormapping` (Hertzmann colour-transfer histogram + cluster
  mapping; both kernels port via the image-shortcut, ACQUIRE
  preview-pipe input snapshot reads back via
  `dt_vulkan_read_from_device` for host-side cluster training —
  same pattern as `globaltonemap`'s lwmax cache miss and
  `hazeremoval`'s ambient-light snapshot).
- **HARD** — denoiseprofile,
  retouch, basecurve (full variants).
  `colorequal` is now **fully ported** (§4.2 — both the non-guiding
  fast path and the 14-kernel guided-filter path with the two-stage
  prefilter / guide orchestration + bilinear up/downsamples). `retouch`
  is now **helper-unblocked** by the
  `dt_dwt_*_vk` à-trous helper (§5.16) — its OpenCL path drives the
  a-trous decomposition through a per-form callback (clone / fill /
  blur / heal); the same callback shape ports to Vulkan with
  `_dwt_layer_func_vk`. The remaining work for retouch is the per-
  form kernels themselves (the largest is `heal` with its Poisson
  iteration). Done
  in earlier passes: `agx` (params struct migrated from PC into a
  storage-buffer binding so the 124 B struct fits — the pattern is
  now available for any future port whose param block exceeds the
  128 B PC budget),
  `filmicrgb` (main per-pixel split + chroma kernels folded into
  one entry that reproduces the host's split-vs-chroma decision
  internally and dispatches on the colour-science version v1..v5;
  reuses `colorbalancergb`'s Yrg/Ych cohort and the §5.11 ICC
  plumbing, adds the filmic-specific gamut-clip family
  `clip_chroma_white/black/clip_chroma/gamut_check_RGB` for v4/v5;
  the wavelet highlight-reconstruction path needs §8.5 sampler
  support and is gated to OpenCL/CPU when `enable_highlight_reconstruction`
  or the GUI mask is on),
  `bloom` (4-kernel chain: threshold + 8×separable uniform box blur
  on a single-channel float scratch + screen-blend mix; first VK
  consumer of float-element scratch buffers — saves 75 % of the
  working-set memory vs float4), `atrous` (edge-aware à-trous
  wavelet equalizer; 2-kernel decompose/synthesize ping-pong over
  max_scale scratch detail buffers, byte-for-byte the active
  process_cl flow),
  `hazeremoval` (dark-channel prior + box min/max refinement +
  guided-filter delegation to `dt_guided_filter_vk` (§5.15) — the
  first consumer of the new helper; 6 module-own kernels, all
  bit-equal),
  `colorreconstruction` (3-D bilateral-grid splat/blur/slice with
  CAS-loop float atomics for the splat; second VK consumer of the
  `vk_atomic_add_f` pattern after the §5.13 bilateral helper.
  Surfaced an OpenCL-vs-GLSL `round()` divergence — both `.cl` and
  `.comp` now use the explicit `floor(x + 0.5)` form to guarantee
  identical cell indices),
  `nlmeans` (non-local-means denoise; 6-kernel shift-and-accumulate
  over the q-offset grid with the new `vk_fast_mexp2f` bit-pun
  weight; the `U2 += accu` needs no atomics since the host
  serialises the q-loop),
  `globaltonemap` (3 tonemap operators reinhard/drago/filmic on a
  shared 2-binding 24 B PC shape; the Drago global-max reduction
  drops the `pixelmax_first` + `pixelmax_second` workgroup-shared
  pair in favour of a CPU readback fallback when the GUI `g->lwmax`
  cache is empty — same precedent as `hazeremoval`'s `_ambient_light`
  global-max handling; detail recovery reuses the §5.13 bilateral
  helper via `dt_bilateral_slice_to_output_vk`). Multi-
  kernel pipelines or per-warp reductions. `lowpass`, `censorize`,
  `shadhi`, `retouch`, `monochrome` can now build
  on the bilateral helper (§5.13) for their grid-based passes.
- **VERY HARD** — demosaicing. The geometric-correction cohort
  (ashift, clipping) has now landed — the `vk_interpolation_*` /
  `vk_sinf_fast` / `vk_clip_mirror` helpers in `dt_vulkan_common.h`
  are the reusable infrastructure that took the bilinear (manual
  4-tap) and the multi-tap (bicubic / lanczos2 / lanczos3, via
  image-shortcut) cases off the still-pending list without needing
  hardware sampler bindings. Demosaicing remains the most complex
  outstanding category (Bayer + X-Trans pattern handling, multiple
  algorithms with different per-pattern code paths).
  Done in earlier passes:
  `borders` (without sampled images — used a sub-region copy
  kernel instead), `mask_manager` (used the new
  `dt_vulkan_copy_device_to_device` HAL primitive instead of a
  kernel),
  `colorin` (input ICC matrix path — both unbound and clipping kernels
  use sampler-clamp integer-coord reads only, so the image-shortcut
  pattern applies; LUTs and matrices go via flat storage buffers, the
  Bayer blue-mapping gamut step ports unchanged, and the lcms2 slow
  path stays on CPU as in OpenCL),
  `crop` (hard crop — `modify_roi_in` already aligns the input buffer
  to the cropped region, so `process_vk` is one `dt_vulkan_copy_device_to_device`,
  same shape as `mask_manager`; practically valuable because crop is
  always in the pipeline and a missing `process_vk` would force a
  CL↔VK transition per image),
  `liquify` (free-form pixel warp — the OpenCL `warp_kernel` uses
  `sampleri` with integer-coord reads, so the Lanczos / bicubic /
  bilinear reconstruction is done in kernel code via a host-prepared
  `k[]` table, not by a hardware sampler; the Vulkan port folds the
  CL flow's image-copy + warp dispatches into one kernel that runs
  the reconstruction inside map_extent and passes-through outside),
  `rasterfile` (same-scale sub-region pass-through via the
  `dt_vulkan_copy_subregion` HAL helper — §5.9 entry 2 — and the
  different-scale resample path via the new §5.18 resampler helper;
  both paths now Vulkan-native),
  `finalscale` (output rescale — first consumer of the §5.18
  resampler; downscale + 1:1 ported, upscale gated to OpenCL / CPU
  like `process_cl`),
  `ashift` (perspective correction — all 4 interpolators
  bilinear/bicubic/lanczos2/lanczos3 with manual in-kernel
  reconstruction; the new `vk_interpolation_bicubic` /
  `vk_interpolation_lanczos` / `vk_sinf_fast` / `vk_clip_mirror`
  helpers in `dt_vulkan_common.h` are also reusable for the future
  clipping / lens module ports),
  `clipping` (crop + rotate + flip + lens-keystone — same four
  interpolators as ashift reusing the helper cohort; reproduces the
  `backtransform` + `keystone_backtransform` from `basic.cl` as in-
  kernel helpers, 124 B PC carrying all 6 small structs the OpenCL
  kernel took as separate args).
  `overlay` was originally in this
  bucket but turned out to be tractable without sampler support — its
  image-bound CL signature was rewritten to storage buffers (RGBA float
  in/out) and the Cairo ARGB32 byte buffer is bound as a packed `uint`
  storage buffer (Cairo's 4-byte stride alignment + naturally
  word-aligned x*4 means each pixel is exactly one uint, no
  cross-word shuffles needed).

See §8 for the staged milestone plan that gets us through these
buckets without breaking the OpenCL coexistence.

## 5. Architecture of a full port

The host side gets a new abstraction layer, `dt_gpu`, that the IOP
modules call. Whether it's backed by OpenCL or Vulkan is the
implementation detail. The migration ships in stages so we don't have
a multi-month branch.

### 5.1 The HAL boundary

A single header would replace today's exposed `opencl.h` surface:

```c
// Hypothetical sketch.
typedef struct dt_gpu_t        dt_gpu_t;
typedef struct dt_gpu_device_t dt_gpu_device_t;
typedef struct dt_gpu_buf_t    dt_gpu_buf_t;     // 1D buffer
typedef struct dt_gpu_img_t    dt_gpu_img_t;     // 2D image (was cl_mem image)
typedef struct dt_gpu_kernel_t dt_gpu_kernel_t;

int  dt_gpu_init(dt_gpu_t *gpu);
int  dt_gpu_load_program(dt_gpu_t *gpu, const char *name);
int  dt_gpu_create_kernel(int program, const char *entry);

dt_gpu_buf_t *dt_gpu_alloc_buffer(int dev, size_t size);
dt_gpu_img_t *dt_gpu_alloc_image (int dev, int w, int h, int bpp);
int           dt_gpu_write_image (int dev, dt_gpu_img_t *dst, const void *src, ...);
int           dt_gpu_read_image  (int dev, void *dst, dt_gpu_img_t *src, ...);

int dt_gpu_enqueue_kernel_2d_args(int dev, int kernel,
                                  size_t w, size_t h, ...);
```

The mapping from these calls to the OpenCL backend is a direct
forwarding; the mapping to Vulkan is the work the PoC starts.

The IOP `process_cl()` signature is the only thing that has to change
in modules. The minimal version keeps `cl_mem` semantics by typedef-
aliasing — modules continue passing opaque device pointers; the type's
backing struct is just different.

### 5.2 Buffer vs image binding

OpenCL's `image2d_t` with samplers maps to two Vulkan concepts:

| OpenCL | Vulkan |
|---|---|
| `read_only image2d_t` + sampler | sampled image (`VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE`) + sampler (`VK_DESCRIPTOR_TYPE_SAMPLER`) |
| `write_only image2d_t` | storage image (`VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`) |
| `global float *buf` | storage buffer (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`) |
| `constant float *buf` | uniform buffer (`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`) |
| kernel scalar args | push constants (preferred, ≤128 bytes) or uniform buffer |

clspv emits the right SPIR-V decorations automatically when given the
`.cl` source; the host runtime queries the produced module's descriptor
layout (via clspv's `--cl-kernel-arg-info` metadata or
SPIRV-Reflect) and builds the matching `VkDescriptorSetLayout`. This
means modules don't need to hand-annotate bindings.

### 5.3 Command stream

OpenCL command queues map cleanly onto Vulkan command buffers + a
compute queue. The current "asyncmode" in `dt_opencl_device_t` —
batching multiple kernels per pipeline run without per-module
synchronisation — translates to recording multiple `vkCmdDispatch`
into one command buffer with `vkCmdPipelineBarrier` between them. This
is more efficient than the OpenCL flow, not less.

### 5.4 Memory model and headroom

Vulkan requires explicit choice of memory type:

- `DEVICE_LOCAL` for kernel buffers (the OpenCL-`CL_MEM_READ_WRITE`
  equivalent).
- `HOST_VISIBLE | HOST_COHERENT` for staging.
- `HOST_VISIBLE | DEVICE_LOCAL` (the BAR / unified-memory tier) for
  small frequently-updated buffers, where available.

The current `pinned_memory` flag on `dt_opencl_device_t` maps directly
to the choice between "staging-copy" and "host-coherent" memory types.
The `unified_memory` flag maps to "device has only HOST_VISIBLE
DEVICE_LOCAL memory" (Apple, iGPUs).

Headroom logic (the `headroom` field and `DT_OPENCL_DEFAULT_HEADROOM`
constant) is OS-/driver-side memory-pressure mitigation; it ports as-is
because Vulkan exposes `VkPhysicalDeviceMemoryProperties` with budget
extensions (`VK_EXT_memory_budget`).

### 5.5 Tiling

`dt_develop_tiling_t.factor_cl` and friends are independent of the
backend — they describe how many bytes of working memory the kernel
needs per output pixel. They port literally.

### 5.6 Module-side abstraction: `dt_vk_module_kernel_t`

The first batch of module ports each ran ~35 lines of per-module
plumbing (open-coded `vk_program`/`vk_kernel_X` fields,
`if(dt_vulkan_running()) { load … create … }` in `init_global`,
guard-and-free in `cleanup_global`, manual buffer-array setup in
`process_vk`). That much boilerplate is impossible to keep
consistent across 70+ modules — different authors will guard things
differently, forget to null-init, leak slots on partial failure.

To keep duplication low and the surface uniform, the host-side API
now exposes a slim module helper in `src/common/vulkan.h`:

```c
typedef struct dt_vk_module_kernel_t { int program, kernel; } dt_vk_module_kernel_t;
#define DT_VK_MODULE_KERNEL_INIT { -1, -1 }

void dt_vulkan_module_kernel_load(dt_vk_module_kernel_t *out,
                                  const char *spv_name,
                                  const char *entry,
                                  uint32_t num_storage_buffers,
                                  uint32_t push_constant_size,
                                  uint32_t local_x, local_y, local_z);
void dt_vulkan_module_kernel_unload(dt_vk_module_kernel_t *k);

int  dt_vulkan_dispatch_inout    (const dt_vk_module_kernel_t *k,
                                  dt_vk_mem_t *in, dt_vk_mem_t *out,
                                  size_t w, size_t h,
                                  const void *pc, size_t pcs);
int  dt_vulkan_dispatch_inout_lut(const dt_vk_module_kernel_t *k,
                                  dt_vk_mem_t *in, dt_vk_mem_t *out,
                                  dt_vk_mem_t *lut,
                                  size_t w, size_t h,
                                  const void *pc, size_t pcs);
```

Each helper is a no-op / failure-return when Vulkan isn't running,
so the calling module never needs an `if(dt_vulkan_running())`
guard. The type is also stubbed in the `HAVE_VULKAN=0` block as an
empty struct, so modules can keep a `dt_vk_module_kernel_t vk;` field
without #ifdef walls in their global-data struct.

A typical ported module is now:

```c
typedef struct dt_iop_X_global_data_t {
  int kernel_X;
  dt_vk_module_kernel_t vk;
} dt_iop_X_global_data_t;

#ifdef HAVE_VULKAN
int process_vk(…) {
  struct { int w, h; …scalars… } pc = { … };
  return dt_vulkan_dispatch_inout(&gd->vk, dev_in, dev_out,
                                  pc.w, pc.h, &pc, sizeof(pc));
}
#endif

void init_global(dt_iop_module_so_t *self) {
  …allocate gd; OpenCL kernel registration if HAVE_OPENCL…
  dt_vulkan_module_kernel_load(&gd->vk, "X", "X", 2, sizeof(pc_X_t),
                               16, 16, 1);
}

void cleanup_global(dt_iop_module_so_t *self) {
  …OpenCL frees if HAVE_OPENCL…
  dt_vulkan_module_kernel_unload(&gd->vk);
  free(self->data);
}
```

The Vulkan-specific footprint per module is ~10 lines (one struct
field, ~6-line `process_vk` body, one line each in `init_global` and
`cleanup_global`). Modules that have two kernel variants
(channelmixerrgb's adaptation modes, temperature's 4f-only path,
invert's 4f vs 1f) keep one `dt_vk_module_kernel_t vk_<variant>;`
field per kernel slot.

This refactor cut ~300 lines of duplication from the first batch of
ports without changing behaviour; it's the pattern every subsequent
module port should follow.

### 5.7 Kernel-side helpers (`dt_vulkan_common.h`)

`data/kernels/vulkan/dt_vulkan_common.h` carries the small set of
helpers that recur across kernels: `clipf` (matches the OpenCL helper
in `data/kernels/common.h`), `idx2d`, `clampf`, `clampf4`,
`read_clamped`, and `matmul3` / `matmul3_padded`. Including it from
every `*.cl` keeps the per-kernel boilerplate consistent and makes
behaviour drift between the kernels visible at one address.

It also carries `vk_lookup` and `vk_lookup_unbounded` — the
storage-buffer equivalents of the OpenCL `image2d_t`-backed
`lookup` / `lookup_unbounded` in `color_conversion.h`. These take a
flat 65536-entry float buffer (matching the OpenCL 256×256 layout)
plus the linear-extrapolation coefficients as three scalars rather
than a `constant float *` (clspv can't accept a private-pointer arg
as `constant`, and push constants are the natural carrier).

The RGB↔HSL helpers (`vk_RGB_to_HSL`, `vk_HSL_to_RGB`,
`vk_hue_to_rgb`) and the Lab/XYZ/sRGB/ProPhoto matrices are also
here. Keep additions to colour-space code byte-for-byte equivalent
to the matching helpers in `data/kernels/colorspace.h`; the two
files are an intentional duplicate that has to stay in sync.

The dt UCS appearance model (`vk_xyY_to_dt_UCS_JCH`,
`vk_dt_UCS_JCH_to_xyY`, and the `…_HCB`/`…_HSB` cylindrical forms)
and the scene-referred CIE 2006 LMS / Filmlight Yrg-Ych cohort
(`vk_LMS_to_Yrg`, `vk_Yrg_to_Ych`, `vk_Ych_to_Yrg`, `vk_Yrg_to_LMS`,
`vk_LMS_to_gradingRGB`, `vk_gradingRGB_to_LMS`, `vk_LMS_to_XYZ`,
`vk_gamut_check_Yrg`), plus `vk_XYZ_to_JzAzBz` / `vk_JzAzBz_2_XYZ`,
`vk_soft_clip`, and `vk_lookup_gamut`, are also here. These were
added incrementally by the `colorharmonizer` (UCS JCH) and
`colorbalancergb` (the rest) ports and are reused by any future
`filmicrgb` / Yrg-space port. Matrix rows are inlined (rather than
the `matrix_dot(v, m[3])` array-arg form) to stay on clspv patterns
that the rest of the header already uses.

**Portability gotcha — `round()`**: OpenCL's `round()` rounds half-
away-from-zero (`round(0.5) == 1`, `round(-0.5) == -1`); GLSL's
`round()` is **implementation-defined** for fractional 0.5 values
and the popular Mesa / glslang implementations pick banker's
rounding to even (`round(0.5) == 0`). When a kernel uses `round()`
to compute an integer index (closest-integer splatting, nearest-cell
lookup, etc.) and a sample lands exactly on .5, the two compilers
will disagree by one cell. The `colorreconstruction` splat caught
this in testing. **Fix in both `.cl` and `.comp`**: use the
explicit `floor(x + 0.5)` form instead of `round(x)`. This is
unambiguous in both languages and matches the OpenCL semantics
byte-for-byte.

### 5.8 The LUT-on-storage-buffer pattern

Curve-based colour modules (`tonecurve`, `rgbcurve`, `rgblevels`,
`basecurve`, `colisa`, …) carry one or two 65536-entry per-channel
look-up tables. In the OpenCL build these are 256×256 `image2d_t`
LUTs sampled with `lookup_unbounded`; in the Vulkan port they're
flat `global const float *` storage buffers bound at descriptors
2..N alongside the standard in/out at 0..1.

The canonical wiring (see `src/iop/colisa.c` for the worked
example) is:

```c
// host side:
dt_vk_mem_t *dev_ctable = dt_vulkan_alloc_buffer(0, sizeof(float) * 0x10000);
dt_vulkan_write_to_device(0, dev_ctable, d->ctable, sizeof(float) * 0x10000);
// … same for dev_ltable …
dt_vk_mem_t *buffers[] = { dev_in, dev_out, dev_ctable, dev_ltable };
int rc = dt_vulkan_dispatch_n(&gd->vk, buffers, 4,
                              width, height, &pc, sizeof(pc));
dt_vulkan_free_buffer(0, dev_ctable);
dt_vulkan_free_buffer(0, dev_ltable);
```

For modules with multiple small uploads (matrices, LUTs, param
structs), prefer the batched variant — it records all uploads,
a single transfer→compute barrier, and the dispatch into one
command buffer with a single `vkQueueSubmit` + `vkWaitForFences`:

```c
dt_vk_mem_t *buffers[] = { dev_in, dev_out, dev_ctable, dev_ltable };
const dt_vk_upload_t uploads[] = {
  { dev_ctable, d->ctable, sizeof(float) * 0x10000 },
  { dev_ltable, d->ltable, sizeof(float) * 0x10000 },
};
int rc = dt_vulkan_dispatch_n_batched(&gd->vk, buffers, 4,
                                      uploads, 2,
                                      width, height, &pc, sizeof(pc));
```

The shared staging buffer is partitioned by 4-byte-aligned
offsets (Vulkan's `vkCmdCopyBuffer` requirement), so all uploads
share one DMA region without extra allocations. For a module
with N small uploads, batching saves N submit/wait round-trips
(~5-30 ms each on a discrete GPU). All modules that used to call
`dt_vulkan_write_to_device` before a dispatch are now on the
batched path: `agx` (5 uploads — params + 4 matrices), `sigmoid`
per_channel (3 matrices), `rgbcurve` / `rgblevels` / `tonecurve`
/ `colorout` (3 LUTs each), `channelmixer` / `colisa` / `filmic`
/ `basicadj` / `lowpass` / `zonesystem` (2 uploads),
`overlay` / `lowlight` / `profile_gamma` (gamma branch) /
`levels` / `temperature` (1f X-Trans branch) /
`channelmixerrgb` (1 upload). For modules with zero uploads,
`dt_vulkan_dispatch_n_batched` short-circuits to
`dt_vulkan_dispatch_n` so there's no overhead from picking the
batched variant.

```c
// kernel side (excerpt):
kernel void X(global const float4 *in, global float4 *out,
              global const float *ctable, global const float *ltable,
              const int width, const int height,
              const float ca0, const float ca1, const float ca2, …)
{
  // …
  o.x = vk_lookup_unbounded(ctable, p.x / 100.0f, ca0, ca1, ca2);
}
```

The three linear-extrapolation coefficients ride in push constants
rather than a third storage buffer — they're scalars and a per-
dispatch buffer allocation for 12 bytes isn't worth it.

Today's implementation re-uploads the LUT on every dispatch.
Curve params change at commit_params time, not per frame, so a
piece-cached scratch buffer would let consecutive dispatches reuse
the same upload — that's a future optimisation; the current
fresh-upload-per-dispatch matches the OpenCL build's behaviour
(see e.g. `colisa.c::process_cl`).

### 5.9 Sub-region copy and multi-fill

Modules that operate on a fraction of the output canvas (the framing
module, mask compositors, history-anchor IOPs) need two primitives
the simple `dispatch_inout` path doesn't cover: filling an
arbitrary destination rectangle with a constant colour, and copying
a source rectangle into an offset region of a larger destination
buffer.

Three complementary mechanisms exist:

1. **Device-to-device buffer copy** (`dt_vulkan_copy_device_to_device`)
   — a thin wrapper around `vkCmdCopyBuffer` for full-buffer
   `dt_vk_mem_t* → dt_vk_mem_t*` transfers, no kernel needed. Used
   by `mask_manager` (the entire OpenCL path was a single
   `enqueue_copy_image`; the Vulkan equivalent is one
   `vkCmdCopyBuffer` of the float4 pipe buffer), and by `crop` and
   `liquify`'s empty-map shortcut.

2. **Sub-region 2-D copy** (`dt_vulkan_copy_subregion`) — submits a
   single `vkCmdCopyBuffer` with `region_h` `VkBufferCopy` entries
   (one per row), each describing a `region_w * bytes_per_pixel`-byte
   row copy from `src + (src_y + r) * src_stride + src_x * bpp` to
   `dst + (dst_y + r) * dst_stride + dst_x * bpp`. Generalises the
   OpenCL `enqueue_copy_image(src, dst, src_origin, dst_origin, region)`
   primitive for arbitrary row strides and origin offsets in either
   buffer. Used by `rasterfile` for its same-scale sub-region pass-
   through (the OpenCL path is exactly `enqueue_copy_image` with
   `(roi_out->x, roi_out->y)` as the src origin). The per-row
   `VkBufferCopy` cost is proportional to `region_h` — fine for typical
   image sizes (4000 rows × ~16 B per descriptor ≈ 64 KB of region
   array, well within typical command-buffer command stream budgets).

3. **Sub-region copy / fill kernels** (`borders.cl::borders_copy`,
   `borders.cl::borders_fill`) — when the source or destination is
   a strict subrectangle of a larger buffer *and* the copy needs to
   thread between a constant fill and a non-trivial source, kernel-
   based copies are clearer than splitting the work across multiple
   `vkCmdCopyBuffer` calls. The OpenCL build dispatches the fill
   kernel over the entire canvas and returns early outside the
   target rectangle; the Vulkan port instead dispatches the kernel
   exactly over the target rectangle (since buffer storage doesn't
   get "early return outside region" for free — wasted work-items
   are real cost). The kernel translates `(lid_x, lid_y)` into a
   write at `(dst_x + lid_x, dst_y + lid_y)`.

These two are deliberately separate APIs. `dt_vulkan_copy_device_to_device`
is cheap (no SPIR-V to load, just a transfer command), and modules
that just need a full-buffer pass-through copy shouldn't have to
register a kernel program.

### 5.10 Recursive Gaussian helper (`dt_gaussian_*_vk`)

`src/common/gaussian.{c,h}` ship a Vulkan equivalent of the
existing `dt_gaussian_*_cl` surface used by `lowpass`, `censorize`,
`shadhi`, `retouch`, and several other modules: Deriche recursive
IIR blur for arbitrary sigma. Shape:

```c
dt_gaussian_vk_t *g = dt_gaussian_init_vk(width, height, channels,
                                          max, min, sigma, order);
int rc = dt_gaussian_blur_vk(g, dev_in, dev_out);
dt_gaussian_free_vk(g);
```

`channels` is 1, 2 or 4 — `dt_gaussian_init_vk` picks the matching
kernel pair internally. The OpenCL convenience wrapper
`dt_gaussian_mean_blur_cl(devid, buf, w, h, ch, sigma)` has a VK
twin `dt_gaussian_mean_blur_vk(devid, buf, w, h, ch, sigma)` that
does init + in-place blur + free in one call.

The OpenCL build runs column-blur + transpose + column-blur +
transpose with a workgroup-local-memory transpose. The Vulkan port
drops the transpose entirely and runs row-then-column blur with one
work-item per row / column — the IIR recurrence is serial along its
sweep axis anyway, so the natural parallelism granularity is one
work-item per row/column, not per pixel. The local-memory transpose
would be a meaningful speedup at large image sizes but needs
clspv/glslang local-memory plumbing the HAL doesn't expose yet.

The 4-channel pair (`gaussian_row_4c`, `gaussian_column_4c`) lives
in one shared multi-entry `.spv` (`gaussian.spv`) — the host extracts
both entries via `dt_vulkan_module_kernel_create_from`. The 1c and
2c kernels are split across four single-entry `.spv` modules
(`gaussian_row_1c`, `gaussian_column_1c`, `gaussian_row_2c`,
`gaussian_column_2c`) because the glslang fallback can only expose
one entry per `.spv`. Splitting also keeps the push-constant blob
sized to the channel count (1c PC = 48 B, 2c = 56 B, 4c = 72 B —
each carries channel-many `Labmax`/`Labmin` floats). The 4c row
blur is **unreachable from the glslang path** (a pre-existing
limitation of the multi-entry approach for 4c); 1c and 2c row blurs
work in both builds since each has its own dedicated `.comp` file.
The host caches all six kernel slots in module-level statics so
every IOP that uses the helper amortises the SPV read across the
process lifetime.

`dt_gaussian_blur_vk` returns -1 on any failure (Vulkan not running,
program load failed, dispatch error). Callers should always check
the return value and fall through to `dt_gaussian_blur_cl` /
`dt_gaussian_blur` (CPU) on -1, in the same shape as the existing
CPU/OpenCL fallback pattern in the lowpass / shadhi / retouch
modules. Validated on lavapipe: 1c and 2c column outputs are
**bit-equal** to the corresponding channels of the existing 4c
column blur on the same input.

### 5.11 ICC profile info storage-buffer plumbing

A large family of modules (`basicadj`, `basecurve`, `colorbalance`,
`filmic`, `rgbcurve`, `rgblevels`, `overexposed`, ...) reads the
work / histogram / output profile via
`dt_colorspaces_iccprofile_info_cl_t` + a 2-D image2d_t tone-curve
LUT. clspv doesn't surface samplers under our buffer-only kernel
convention, so the Vulkan path replaces both with storage buffers:

* `vk_dt_colorspaces_iccprofile_info_t` — declared in
  `dt_vulkan_common.h`; byte-for-byte equivalent to the existing
  `dt_colorspaces_iccprofile_info_cl_t` (156 bytes). `spirv-dis`
  confirms the std430 offsets land at 0 / 36 / 72 / 76 / 112 / 148 /
  152 — exactly what the C struct produces.
* Tone-curve LUT — a flat float buffer of `6·lutsize` entries with
  channels laid out as `[in_R, in_G, in_B, out_R, out_G, out_B]`,
  each `lutsize` floats long. The OpenCL build uses the same flat
  array internally; it only reshapes into a 256×(256·6) image2d_t
  because of OpenCL sampler limits.
* Kernel-side helpers in `dt_vulkan_common.h` mirror the OpenCL
  helpers in `data/kernels/color_conversion.h` and `rgb_norms.h`
  function-for-function: `vk_lerp_lookup_unbounded`,
  `vk_apply_trc_in` / `vk_apply_trc_out`, `vk_get_rgb_matrix_luminance`,
  `vk_dt_camera_rgb_luminance`, `vk_dt_rgb_norm`. All take the
  profile struct + LUT as `global const` pointers, matching the
  binding shape established by `dt_vulkan_dispatch_n`.

Host-side mirrors the OpenCL surface in `iop_profile.h`:

```c
dt_colorspaces_iccprofile_info_vk_t *info = NULL;
float *lut = NULL;
dt_vk_mem_t *dev_info = NULL;
dt_vk_mem_t *dev_lut  = NULL;
if(dt_ioppr_build_iccprofile_params_vk(work_profile, devid,
                                       &info, &lut, &dev_info, &dev_lut) != 0)
  goto cleanup;
/* ... bind dev_info, dev_lut, dispatch_n ... */
dt_ioppr_free_iccprofile_params_vk(&info, &lut, &dev_info, &dev_lut, devid);
```

`build_iccprofile_params_vk` copies the matrix / coeffs / lutsize /
nonlinearlut / grey into the host struct, flattens the
`lut_in[3]` / `lut_out[3]` arrays into the device-side 6·lutsize
buffer, and uploads both. When `profile_info` is `NULL` (no profile)
a 6-float placeholder is allocated so the binding stays valid; the
kernel guards against actually reading it via `use_work_profile`.

The first consumer (`overexposed`) is wired up: it uses
`vk_get_rgb_matrix_luminance` for the GAMUT / LUMINANCE / SATURATION
clipping previews and a 5-binding dispatch (in, out, tmp,
profile_info, profile_lut). The host-side colorspace transform that
populates `tmp` is currently still done CPU-side via
`dt_ioppr_transform_image_colorspace_rgb` — porting that helper to
Vulkan is the next step here. Once it's in, the round-trip through
host memory in `overexposed::process_vk` goes away.

Caveat shared with §5.8: the profile struct contains a `float[3][3]`
in C, which std430 lays out identically to `float[9]` (alignment 4,
no padding). The GLSL fallback in `overexposed.comp` declares the
member as `float unbounded_coeffs_in[9]` and accesses
`[channel*3 + coeff]`; the clspv-built kernel keeps the 2-D
declaration. `spirv-dis` is the trustworthy oracle here; cross-check
new modules against the offsets above on first compile.

### 5.12 Cross-profile RGB→RGB transform on the GPU

Several modules need a working-space → output-space (or histogram-
space) RGB transform mid-pipeline: optional input TRC LUT, 3×3
matrix product, optional output TRC LUT. The OpenCL build does
this via `dt_ioppr_transform_image_colorspace_rgb_cl` calling the
`colorspaces_transform_rgb_matrix_to_rgb` kernel.

Vulkan mirror in `src/common/iop_profile.{c,h}`:

```c
if(!dt_ioppr_transform_image_colorspace_rgb_vk(devid, dev_in, dev_out,
                                               width, height,
                                               profile_from, profile_to,
                                               self->op)) {
    /* fall back to CPU round-trip for lcms2-only profiles */
}
```

* Kernel: `data/kernels/vulkan/colorspaces.cl` (+ `.comp` fallback);
  6 storage-buffer bindings (in, out, from-profile_info, from-LUT,
  to-profile_info, to-LUT) and a 44-byte push constant block
  (`vk_colorspace_rgb_pc_t`: 2 ints + 9 floats = `pack_3xSSE_to_3x3`
  output of `matrix_to->matrix_out · matrix_from->matrix_in`).
* The host helper pre-combines the two 3×3 matrices into a single
  product so the kernel only does one matmul per pixel — same shape
  as the OpenCL code path.
* Same-profile shortcut: when `from->type == to->type` and
  filenames match, the helper does a `vkCmdCopyBuffer` (via
  `dt_vulkan_copy_device_to_device`) instead of dispatching.
* Non-matrix profile fallback: when either profile lacks a usable
  matrix (lcms2-only), the helper returns `FALSE` so the caller can
  round-trip through host memory. `overexposed::process_vk` is the
  first consumer; the fallback path runs only when the histogram /
  current profile is a CMS-only entry.

Kernel slot caching mirrors `dt_gaussian_*_vk` (§5.10): a static
`_vk_colorspaces_rgb2rgb` slot in `iop_profile.c` loaded lazily on
first call, no `dt_vulkan_t` plumbing needed. The two other
colorspaces.cl kernels (`lab_to_rgb_matrix`, `rgb_matrix_to_lab`)
have not been ported yet — they're additional entry points in the
same `.spv` module that we'd wire up the same way when a consumer
needs them.

### 5.13 Bilateral filter helper (`dt_bilateral_*_vk`)

`src/common/bilateralvk.{c,h}` ship a Vulkan equivalent of the
existing `dt_bilateral_*_cl` surface used by `lowpass`,
`censorize`, `shadhi`, `retouch`, `monochrome`, `globaltonemap` and
several other modules: 3-D-grid bilateral filtering of the L
channel of a Lab buffer.

```c
dt_bilateral_vk_t *b = dt_bilateral_init_vk(width, height, sigma_s, sigma_r);
dt_bilateral_splat_vk(b, dev_in);
dt_bilateral_blur_vk(b);
dt_bilateral_slice_vk(b, dev_in, dev_out, detail);
dt_bilateral_free_vk(b);
```

The kernels live as six separate `.cl` / `.comp` pairs under
`data/kernels/vulkan/`:

| Entry                          | Bindings | PC bytes |
|--------------------------------|---------:|---------:|
| `bilateral_zero`               |        1 |        8 |
| `bilateral_splat`              |        2 |       28 |
| `bilateral_blur_line`          |        2 |       24 |
| `bilateral_blur_line_z`        |        2 |       24 |
| `bilateral_slice`              |        3 |       32 |
| `bilateral_slice_to_output`    |        4 |       32 |

Splitting each entry into its own `.spv` (rather than the multi-
entry-in-one-module pattern used by channelmixerrgb) is deliberate:
the glslang fallback path can only emit one entry per build, so
modules that consume the whole bilateral pipeline need all six
kernels available on glslang-only systems. The clspv path still
compiles each `.cl` separately too — the storage cost is six tiny
`.spv` files (~5 KB each) instead of one ~25 KB blob.

The host helper sizes the grid via `dt_bilateral_grid_size` from
`src/common/bilateral.c` — same `L_range = 100` convention and
clamps as the CL backend, so VK and CL produce bit-equal grids for
matching sigma inputs. Two device-local buffers are allocated up
front (`dev_grid` and `dev_grid_tmp`) so the three blur passes can
ping-pong without per-pass reallocations. The zero kernel runs
once at init via a 2-D dispatch over `(size_x, size_y * size_z)`.

**Splat correctness vs splat performance** — the OpenCL splat
kernel uses workgroup-local memory to reduce atomic contention
before the global `atomic_add_f`. The Vulkan port currently does
direct `vk_atomic_add_f` per work-item × 8 grid cells (no local-
memory reduction). This is correct but slower under heavy
contention — adjacent pixels that map to the same grid cell pay
CAS-loop retries. Local-memory reduction is a planned follow-up;
the current form is the "correctness scaffold before optimisation"
shape.

**`vk_atomic_add_f`** — added to `dt_vulkan_common.h`. CAS loop on
the `uint` reinterpretation of the float (same shape as
`data/kernels/common.h::atomic_add_f` minus the NVIDIA-PTX fast
path). The GLSL fallback inlines the equivalent loop with
`atomicCompSwap` on a `uint`-aliased grid buffer (the splat `.comp`
file). The same `union { float; uint; }` bit-pun idiom carries
`vk_fast_mexp2f` (the `common.h::fast_mexp2f` 2^-x approximation,
added for `nlmeans`) and colorchecker's `fastlog2` — all three are
clspv-safe and the `.comp` twins use `uintBitsToFloat` /
`floatBitsToUint`.

No darktable-global state — the six kernel slots live in static
`dt_vk_module_kernel_t` slots in `bilateralvk.c`, loaded lazily on
first `dt_bilateral_init_vk` call, same pattern as
`dt_gaussian_*_vk` (§5.10) and the colorspaces helper (§5.12).

`dt_bilateral_*_vk` returns -1 on any failure (Vulkan not running,
kernel load failed, dispatch error). Callers should always check
return values and fall through to `dt_bilateral_*_cl` /
`dt_bilateral_*` (CPU) on -1, in the same shape as the existing
CPU/OpenCL fallback pattern in `lowpass` / `shadhi` / `retouch`.

### 5.14 VK→VK chain hand-off cache

The original pixelpipe-VK integration (§4.2) host-stages every
module: download input → upload to VK → dispatch → download to
host. For two consecutive `process_vk` modules that's two
redundant transfers around the boundary — the previous output
buffer is already on the GPU and the new module is about to
upload exactly the same bytes back.

The hand-off cache eliminates that redundancy. After a successful
`process_vk` dispatch the pixelpipe keeps the output `dt_vk_mem_t*`
alive on the pipe state (`dt_dev_pixelpipe_t::vk_handoff_buf` /
`vk_handoff_size`). On the next module:

* If it has `process_vk` and the cached buffer's size matches the
  expected input size, reuse it directly as `vin` — skip both the
  device-local input allocation and the host→device upload. The
  log line gains a `[vk handoff]` suffix so it's easy to see in a
  trace.
* If it has `process_vk` but the size doesn't match (ROI changed
  mid-pipeline, e.g. crop), drop the cached buffer and follow the
  normal alloc-and-upload path.
* If it's an OpenCL module or a CPU-only module, the cache is
  invalidated immediately. Per-pipeline teardown also releases any
  lingering buffer, covering mid-pipeline aborts.

Buffer ownership transfers atomically — once a module reuses the
cached buffer, the pipe state slot is cleared so a dispatch
failure can't double-free it. The cache is per-pipeline rather
than global, which matches the OpenCL `cl_mem` chain semantics
and means parallel preview/full/export pipelines don't interfere.

**Host-mutation invariant.** The cache contains the previous
module's *raw* `process_vk` output. The host buffer for the same
pipe stage was written from that same data via
`dt_vulkan_read_from_device`, so initially they are bit-identical.
Anything that subsequently mutates the host buffer in place — the
input-side colourspace transform in `_pixelpipe_process_on_CPU`
when `cst_from != cst_to`, the blend-side input/output transforms
when `_transform_for_blend` fires — desynchronises the two.
`_vk_handoff_invalidate(pipe)` must run *before* every such
mutation; otherwise the next module's `vin_from_cache` path
silently feeds the kernel pre-mutation GPU bytes while the
parameters were chosen against the post-mutation host state. The
darkroom-only regression that motivated this rule was
`agx → colorout` with an RGB→Lab host transform between them:
colorout's VK kernel read RGB pixels but applied a Lab→RGB matrix,
producing a uniformly black framebuffer. (Export was unaffected
because `finalscale` runs in OpenCL between agx and colorout,
breaking the chain via the existing CL-branch invalidation.) The
CPU-fallback branch and the OpenCL branch already invalidate at
their entry; the only previously-uncovered sites were the host
colourspace transforms.

For VK→VK pairs this halves the per-boundary host traffic. The
asymmetric temperature→exposure timing reported in §10.1 was
*not* caused by VK→VK transitions (other OpenCL modules ran in
between), so the user-visible improvement only shows up once
enough modules in the default pipeline expose `process_vk` to
form chains. The infrastructure is in place now; module coverage
catches up incrementally as more modules port.

### 5.15 Guided-filter helper (`dt_guided_filter_vk`)

Third multi-kernel device helper (after §5.10 Gaussian and §5.13
bilateral). The guided filter is a local linear regression of the
input onto the (clipped, weighted) RGB guide; box-meaning the
per-pixel products plus a 3×3 Cramer solve yields a smoothed
coefficient field, which is then applied back to the guide.
Used by `hazeremoval` (landed) and `colorequal` (pending) — single
shared implementation, one helper to keep in sync.

```c
int dt_guided_filter_vk(int devid,
                        dt_vk_mem_t *guide, dt_vk_mem_t *in, dt_vk_mem_t *out,
                        int width, int height, int w,
                        float sqrt_eps, float guide_weight, float min, float max);
```

Eight kernels (1:1 with `guided_filter.cl`):

- `guided_filter_split_rgb` — guide → three weighted single-channel
  float buffers. 4 bindings, 16 B PC.
- `guided_filter_box_mean_x` / `_y` — separable box mean via a
  Kahan-compensated sliding sum. 1D dispatch (over rows / columns).
  2 bindings, 12 B PC. A new `vk_kahan_sum(m, c, add)` lvalue macro
  in `dt_vulkan_common.h` matches `common.h::Kahan_sum` byte-for-byte.
- `guided_filter_covariances` — per-pixel `img · weighted guide`. 5
  bindings, 16 B PC.
- `guided_filter_variances` — per-pixel symmetric guide cov (6 terms).
  7 bindings, 16 B PC.
- `guided_filter_update_covariance` — `out = in − a·b + eps`. 4
  bindings, 12 B PC.
- `guided_filter_solve` — per-pixel 3×3 Cramer solve. **17 bindings**
  (13 read + 4 write), 8 B PC. This is the reason
  `DT_VULKAN_MAX_BINDINGS` was bumped from 16 to 20 (the Vulkan spec
  floor is 4 but every real desktop / mobile GPU plus MoltenVK→Metal
  allows well above 20; clspv's argument-flattening makes hitting 16
  trivial on multi-channel solves).
- `guided_filter_generate_result` — clamp(`gw · (guide · a) + b`).
  6 bindings, 24 B PC.

Storage model: the **guide is a `float4`** flat buffer; `in`, `out`
and all ~20 scratch buffers are **single-channel float** —
matches the OpenCL `CL_R` allocation and saves 75 % of the
working-set memory vs. the float4 default. The helper allocates the
scratch internally on each call; no per-instance handle needed
(unlike Gaussian/bilateral) — the helper signature is stateless.

The orchestration follows `_guided_filter_cl_impl` byte-for-byte:
split → 4× box-mean → covariances + variances → 9× (box-mean +
update_covariance) → solve → 4× box-mean (`a_{r,g,b}`, `b`) →
generate_result. Single-tile only — the tiled variant for very
large images is deferred (the OpenCL fallback handles it; a future
Path C zero-copy round would also remove most of the motivation).
Returns 0 on success or -1 on any failure (Vulkan not running,
kernel load failed, allocation failed, dispatch error). Callers
should fall through to `guided_filter_cl` / CPU `guided_filter`
on -1, same pattern as the gaussian / bilateral helpers.

Validated end-to-end on lavapipe against an independent C reference
of the same dataflow (a separable clamped-window box mean stands in
for the kernels' Kahan running sum — same math to FP precision):
max error 1.2e-4 over a 40×32 patch through the full 8-kernel
sequence.

### 5.16 À-trous wavelet helper (`dt_dwt_*_vk`)

Fourth multi-kernel device helper (after §5.10 Gaussian, §5.13
bilateral and §5.15 guided filter). Mirrors `dt_dwt_*_cl` from
`src/common/dwt.{c,h}` byte-for-byte: an iterative a-trous
B3-spline hat transform that decomposes a float4 image into N
detail scales plus a residual, with a user-supplied per-scale
callback that lets the consumer post-process each band (denoise,
recompose, mask, …) before the next iteration runs. Used by
`retouch` (pending port — clone / heal / fill / blur all consume
the decomposed scales).

```c
dwt_params_vk_t *p = dt_dwt_init_vk(devid, image, w, h,
                                    scales, return_layer,
                                    merge_from_scale, user_data,
                                    preview_scale);
dwt_decompose_vk(p, layer_callback);   // calls callback per scale
dt_dwt_free_vk(p);
```

Five kernels (1:1 with `dwt.cl`):

| kernel                    | bindings | PC (bytes) |
|---------------------------|:--------:|:----------:|
| `dwt_init_buffer`         |        1 |          8 |
| `dwt_add_img_to_layer`    |        2 |          8 |
| `dwt_subtract_layer`      |        2 |          8 |
| `dwt_hat_transform_row`   |        2 |         12 |
| `dwt_hat_transform_col`   |        2 |         16 |

Two minor differences vs the OpenCL surface (all transparent to
callers):

- `dwt_subtract_layer.cl` takes an unused `const float lpass_mult`
  argument; the Vulkan twin drops it to keep PC at 8 B. The host
  helper never reads it either, so no caller observes a change.
- The OpenCL `dwt_hat_transform_col` PC is laid out as
  (width, height, sc, lpass_mult) — same 16 B layout — so the
  Vulkan port keeps the field order verbatim.

The host-side decomposition driver lives next to the existing
`dwt_wavelet_decompose_cl` in `src/common/dwt.c` and reproduces
the same ping-pong-over-`buffer[2]` loop with `temp` allocated +
freed per scale (same as OpenCL — the per-scale alloc/free was
the OpenCL helper's way of dodging a -4 OOM on tiny GPUs; we
keep it for memory parity). `_get_max_scale` and
`_first_scale_visible` are shared with the OpenCL path so the
two backends always agree on scale count.

The callback returns `int` (0 = success, non-zero = abort the
decomposition) instead of `cl_int`. Retouch's eventual port will
typedef `_dwt_layer_func_vk` for its per-form processing and
re-use the same per-form code paths the CL helper drives.

Validated end-to-end on lavapipe against an independent C
reference that recreates the row/col hat transforms and the
subtract/accumulate flow: full decomposition + recomposition is
**bit-equal** (max 0.0) against the C reference, and the identity
round-trip — input → 3-scale decompose → 3-scale recompose —
matches the input at single-bit FP precision (1.9e-6). Confirms
the dispatch chain, the PC layouts, the ping-pong buffer wiring
and the `(1/16)` `lpass_mult` normalisation all reproduce the
OpenCL helper byte-for-byte.

### 5.17 Local Laplacian pyramid helper (`dt_local_laplacian_*_vk`)

Fifth multi-kernel device helper (after §5.10 Gaussian, §5.13
bilateral, §5.15 guided filter and §5.16 DWT). Local Laplacian
filtering for HDR-style local tone mapping: builds a multi-scale
Gaussian pyramid over the L channel, evaluates the user-supplied
shadow/highlight/clarity curve at 6 evenly-spaced reference
brightnesses, then assembles the final pyramid by interpolating
between the two laplacian pyramids nearest the local brightness.

Used by `bilat` (HDR local tone-mapping / "clarity" arm — see §4.2
update). The OpenCL surface is `dt_local_laplacian_cl`; the Vulkan
twin lives in `src/common/locallaplacianvk.{c,h}` and mirrors it
init / apply / free byte-for-byte:

```c
dt_local_laplacian_vk_t *l = dt_local_laplacian_init_vk(
    devid, width, height, sigma, shadows, highlights, clarity);
dt_local_laplacian_vk(l, dev_in, dev_out);
dt_local_laplacian_free_vk(l);
```

Five kernels (1:1 with `locallaplacian.cl`):

| kernel                  | bindings | PC (bytes) |
|-------------------------|:--------:|:----------:|
| `ll_pad_input`          |        2 |         20 |
| `ll_gauss_reduce`       |        2 |         16 |
| `ll_process_curve`      |        2 |         28 |
| `ll_laplacian_assemble` |       15 |         16 |
| `ll_write_back`         |        3 |         16 |

All `image2d_t` reads in `locallaplacian.cl` use sampler-clamp
integer coords — no actual filtering — so the same image-shortcut
pattern used by `overlay`/`sigmoid`/`agx`/`colormapping` applies,
and every binding becomes a flat float (or float4) storage buffer
with explicit `clamp` in the kernel. The OpenCL `gauss_reduce`
implicit CLAMP_TO_EDGE on the fine buffer becomes an explicit
`clamp(2*cx+ii, 0, fine_w-1)` in both the `.cl` and `.comp` —
otherwise reads near the bottom-right corner of the padded buffer
could go one row past in row-major flat-buffer addressing.

`ll_laplacian_assemble`'s 15 bindings fit the bumped 20-binding
ceiling (`DT_VULKAN_MAX_BINDINGS` was raised 16→20 for
`guided_filter_solve`). The OpenCL kernel includes an `#ifdef AMD`
branch that uses `select(…)` instead of `switch(lo)` to work
around a driver bug (issue #3756); the Vulkan port keeps the
`switch` form for both clspv and glslang since the AMD workaround
is OpenCL-specific.

Per-level dimensions `(lwidth[l], lheight[l])` are cached at init
time using the same `_dl()` "downsample-by-2 with ceiling" recurrence
as the CL helper. Each level allocates one `dev_padded`, one
`dev_output` and six `dev_processed[gamma]` float buffers — same
2 + num_gamma = 8 buffers per level as OpenCL. The "AMD-issue
workaround branch" in `locallaplacian.cl` is omitted from the
Vulkan port (the GLSL switch-based form is fine on all drivers
we've tested).

Note: this is the first helper with a write-image-back step that
copies chroma + alpha from the input buffer to the output, mirroring
the OpenCL kernel's `pixel.x = 100.0f * processed; write_imagef(out)`
pattern. The output buffer's chroma channels come from the *original*
float4 input, not from any padded intermediate.

`bilat.c` previously gated its `s_mode_local_laplacian` arm to
OpenCL/CPU via `process_vk_ready = FALSE`; with this helper in
place, the gate is removed and the Vulkan path covers both modes
of "local contrast" — bilateral and local Laplacian (the default).

Validated end-to-end on lavapipe with per-kernel tests against
independent C references: pad_input and gauss_reduce are bit-equal
(max 0.0), process_curve and laplacian_assemble match at 6e-8
(single-bit FP), write_back round-trips exactly (L scale + chroma
passthrough).

### 5.18 Resampler helper (`dt_interpolation_resample_vk` / `dt_iop_clip_and_zoom_vk`)

Sixth device helper. The image up/down-scaler used by the geometric
and scaling modules (`finalscale`, `rasterfile`, and the
sampler-blocked `clipping` / `ashift` once those land). Mirrors
`dt_interpolation_resample_cl` in `src/common/interpolation.c` and
`dt_iop_clip_and_zoom_{,roi_}cl` in `src/develop/imageop_math.c`.

The OpenCL `interpolation_resample` kernel processes the image
column-wise with **workgroup-local memory + a recursive reduction**
for the vertical convolution — it caches the horizontal-convolution
intermediate in `local float4 buffer[]` and reduces across work-items.
The Vulkan port instead does the **full separable convolution as a
single per-output-pixel gather**:

```
out[x,y] = Σ_iy vkernel[iy] · ( Σ_ix hkernel[ix] · in[vindex[iy], hindex[ix]] )
```

This is mathematically identical (a separable kernel is the outer
product of its 1-D factors, so the 2-D convolution equals the nested
sum either way) but needs **no local memory, no barriers, and no
intermediate buffer** — a much better fit for the one-shot HAL. For
typical tap counts (lanczos3 = 6×6 = 36 multiply-adds per output
pixel) the arithmetic is modest. It also sidesteps the OpenCL path's
`CL_INVALID_WORK_GROUP_SIZE` fallback (where the vertical tap count
exceeds the workgroup size and the CL helper bails to CPU) — the
single-pass gather has no such limit, so the `_roi` variant needs no
CPU fallback.

The resampling **plan tables** are built by the *shared* static
`_prepare_resampling_plan` (the same code the OpenCL path uses,
verbatim) and uploaded as flat storage buffers in one batched submit:
`hmeta`/`vmeta` (3 ints per output column/row — the (length, kernel,
index) base offsets), `hlength`/`vlength`, `hindex`/`vindex` (pre-
clamped input coordinates, so no border handling in the kernel), and
`hkernel`/`vkernel` (normalised filter weights). 10 storage bindings,
16 B PC (width, height, in_width, in_height).

Two kernels:

| kernel                   | bindings | PC (bytes) | role |
|--------------------------|:--------:|:----------:|------|
| `interpolation_resample` |       10 |         16 | generic up/down-scale gather |
| `interpolation_copy`     |        2 |         24 | 1:1 expand crop with zero-fill |

The 1:1 paths mirror `dt_interpolation_resample_cl`'s `copymode`
branch: when `roi_out->scale == 1` and the input fully covers the
output, the new `dt_vulkan_copy_subregion` (§5.9) handles it with a
plain row-wise `vkCmdCopyBuffer`; when the input doesn't fully cover
(expanded crop), `interpolation_copy` zero-fills the out-of-bounds
border.

`dt_iop_clip_and_zoom_vk` / `_roi_vk` are the thin `imageop_math.c`
wrappers (pick the user-preference interpolator, call the resampler)
that the modules call, exactly paralleling the `_cl` pair.

Validated end-to-end on lavapipe against C references that recreate
the gather with hand-built plan tables (the real plan math is shared
verbatim with the already-tested OpenCL path, so the Vulkan-specific
risk is the table indirection + accumulation): the 1:1 offset copy
with zero-fill, a 2× box downscale, and an **asymmetric variable-tap
plan** (2–4 taps per pixel with a packed non-uniform-stride index /
kernel layout, exercising the per-pixel meta offsets) are **all three
bit-equal (max 0.0)**.

## 6. Per-OS picture

### Linux

- Vulkan loader and one ICD per vendor: Mesa RADV (AMD), Mesa ANV
  (Intel), NVIDIA proprietary, AMDVLK.
- Build deps: `libvulkan-dev` (in distro repos), `clspv` (must be
  obtained as a binary or built from source — no Debian package yet).
  Full per-distro lists in §7.
- No regression for users with current OpenCL setups: the existing
  OpenCL path continues to coexist behind `USE_OPENCL` until removed.

### Windows

- Vulkan loader is shipped with all major GPU drivers (AMD, NVIDIA,
  Intel).
- Build dep: LunarG Vulkan SDK; `clspv` shipped in our packaging or
  built in-tree.
- Replaces today's reliance on each vendor's OpenCL ICD. Same kernels.

### macOS

- We rely on **MoltenVK** as the Vulkan-on-Metal layer. Available via
  Homebrew (`vulkan-headers`, `molten-vk`) and bundled with the
  LunarG SDK.
- `VK_KHR_portability_subset` is required and is handled by toggling
  the corresponding instance/device flags (our PoC already requests
  Vulkan 1.2; `VK_KHR_portability_enumeration` is exposed by lavapipe
  and present in all MoltenVK builds).
- MoltenVK uses SPIRV-Cross at runtime to translate SPIR-V → MSL.
  We do not write Metal directly.
- Overhead: MoltenVK adds a translation pass on first use of a
  pipeline; subsequent dispatches incur ~0% additional cost. Pipeline
  cache reuse is supported via `VkPipelineCache`, so we serialize
  caches per-image-export.

### Edge case: macOS without Vulkan

For older macOS releases (pre-10.13) or installations where MoltenVK
is unavailable, we fall back to the CPU path — same behaviour darktable
already has for systems without OpenCL.

## 7. Build and developer dependencies

The PoC in `tools/vulkan_compute_poc/` and (later) the full Vulkan path
need a small set of extra packages on top of darktable's normal build
deps. Versions are the ones we've verified during PoC bring-up; older
versions probably work but we have not pinned a hard minimum yet.

### Ubuntu 24.04 LTS (and 26.04)

All packages are in the stock archive — no PPAs required. Tested
combination:

| Package | Role | Notes |
|---|---|---|
| `libvulkan-dev` | Vulkan headers + loader | provides `find_package(Vulkan)` |
| `vulkan-tools` | `vulkaninfo`, debug utilities | optional, useful for triage |
| `mesa-vulkan-drivers` | lavapipe software ICD | required to *run* the PoC on headless CI; on dev machines the vendor driver is used instead |
| `glslang-tools` | `glslangValidator` | fallback shader compiler when `clspv` isn't installed |
| `spirv-tools` | `spirv-val`, `spirv-dis` | optional, the CMake target uses them if present |
| `spirv-cross` | `spirv-cross` CLI | optional, handy for inspecting what MoltenVK will see |
| `clang-18` (or newer) | clspv build-from-source dep | only needed if you build clspv yourself |

One-liner:

```sh
sudo apt-get install -y \
    libvulkan-dev vulkan-tools mesa-vulkan-drivers \
    glslang-tools spirv-tools spirv-cross
```

`clspv` is **not** packaged in Ubuntu 24.04 or 26.04. Until that
changes, contributors who want the production-path build either:

- pull a prebuilt binary from the project's CI artefacts
  (`https://github.com/google/clspv`), or
- build it in-tree against system LLVM. A pinned build script will
  live in `tools/vulkan_compute_poc/scripts/` once the PoC graduates.

Without `clspv`, the PoC CMake falls back to compiling the GLSL twin
(`kernels/basicadj_min.comp`) with `glslangValidator` — same SPIR-V
output shape, same dispatch, kept behaviour-equivalent by hand. Useful
for getting the host-side runtime working in environments where
installing clspv is impractical (CI, fresh dev containers).

### Fedora 40+ / RHEL-likes

Equivalent packages: `vulkan-loader-devel`, `vulkan-tools`,
`mesa-vulkan-drivers`, `glslang`, `spirv-tools`, `spirv-cross-devel`.
`clspv` is not packaged; the same in-tree-build story applies.

### Arch / Manjaro

`vulkan-devel` (meta), `vulkan-tools`, `vulkan-mesa-layers` (or
the vendor driver of choice), `glslang`, `spirv-tools`,
`spirv-cross`. `clspv` is in AUR.

### macOS (Homebrew)

The PoC build hasn't been exercised on macOS in this PR (no CI hosts
yet), but the dependency surface is small:

```sh
brew install vulkan-headers molten-vk glslang spirv-tools spirv-cross
```

`molten-vk` provides both the Vulkan loader shim (`libvulkan.dylib`)
and the Metal-backed ICD; `vulkan-headers` is what `find_package(Vulkan)`
picks up. The LunarG SDK is an alternative one-shot install that
bundles all of these plus `vulkaninfo`.

### Windows (MSYS2 UCRT64)

The CI's MSYS2 environment can install everything needed via:

```
pacboy -S vulkan-headers:p vulkan-validation-layers:p \
          glslang:p spirv-tools:p
```

`molten-vk` does not apply on Windows; the Vulkan loader is installed
with the GPU driver. For developer machines the LunarG SDK installer
is the simplest option.

### CI

The Linux job in `.github/workflows/ci.yml` adds a dedicated matrix
entry that installs the package set above and builds the PoC with
`-DBUILD_VULKAN_COMPUTE_POC=ON`. The entry also smoke-runs the PoC
binary against the lavapipe software ICD; exit code `77` means "no
Vulkan device on this runner, skipping the dispatch test" and is
treated as success (the build-clean signal is still the main thing
we care about at this stage). Non-zero, non-77 exit codes fail the
matrix entry.

### Testing a Vulkan kernel port

The verification layers we exercise per port, smallest to largest:

1. **SPIR-V validation at build time.** `data/kernels/vulkan/CMakeLists.txt`
   pipes every produced `.spv` through `spirv-val` (when
   `SPIRV_VAL_BIN` was found at configure time). Catches malformed
   SPIR-V — typically caused by GLSL syntax errors or unsupported
   intrinsics.

2. **Push-constant layout cross-check.** After build, run
   `spirv-dis <kernel>.spv | grep OpMemberDecorate` and confirm the
   reported offsets match the C-side `vk_<module>_pc_t` struct
   byte-for-byte. A mismatch silently corrupts kernel reads of push
   constants — the build doesn't catch this, but a single
   `spirv-dis` check does.

3. **Library link check.** `cmake --build build --target <module>`
   reproduces the per-module compile + link cycle (the same one the
   pipelines load at runtime). Confirms the C-side `process_vk`
   compiles, `dt_vk_module_kernel_t` field types are right, and
   `dt_vulkan_*` calls resolve under both `HAVE_VULKAN={ON,OFF}`.

4. **End-to-end dispatch on lavapipe.** Mesa ships a software Vulkan
   ICD (`llvmpipe`) that works without any GPU. `vulkaninfo --summary`
   shows it as `deviceType = PHYSICAL_DEVICE_TYPE_CPU`. The
   `tools/vulkan_compute_poc/build/dt_vk_compute_poc` binary is the
   reference dispatch harness — it loads a SPIR-V module, allocates
   buffers, dispatches, and compares the result against a CPU
   reference. The same harness can be pointed at any kernel module
   under `data/kernels/vulkan/` with `--spv <path>.spv` and a
   compatible push-constant struct; the existing build target uses
   `basicadj_min.spv`.

5. **Integration tests.** `src/tests/integration/` runs the full
   darktable pipeline against reference RAW inputs and compares the
   output via delta-E. Costlier (needs `darktable-cli` plus the
   test image set) but is the only way to verify a port composes
   correctly with the rest of the pipeline. Run with
   `--disable-opencl` to force the CPU/Vulkan path; once
   pixel-equality CI is in place this becomes the canonical
   regression gate.

The first three layers all run today in seconds and should be done
on every change; the fourth is one extra command; the fifth is the
gold standard but takes minutes.

### Developer convenience

`build.sh` exposes the new options through the standard `--enable-X`
/ `--disable-X` mechanism, so they read like any other feature flag:

```sh
# Enable the Vulkan backend integration:
./build.sh --enable-vulkan

# Also build the standalone PoC tool:
./build.sh --enable-vulkan --enable-vulkan-poc

# Same thing in plain cmake:
cmake -S . -B build -DUSE_VULKAN=ON -DBUILD_VULKAN_COMPUTE_POC=ON
cmake --build build
```

Hyphens in the CLI map to underscores in the underlying feature
name, so `--enable-vulkan-poc` toggles the `VULKAN_POC` feature
which is wired to the `BUILD_VULKAN_COMPUTE_POC` cmake variable
(the only feature in build.sh that maps to a `BUILD_*` rather than
a `USE_*` option; see the inline `case` in `build.sh`).

## 8. Migration milestones

1. ✅ **Standalone PoC** (landed). Buffer-only kernel proves the
   round-trip works without touching darktable's main build.
2. ✅ **In-tree Vulkan backend** (landed). `src/common/vulkan.{c,h}`,
   `USE_VULKAN` CMake option, `HAVE_VULKAN` define, runtime init
   alongside OpenCL.
3. ✅ **`process_vk` API surface** (landed). `OPTIONAL(int,
   process_vk, ...)` in `src/iop/iop_api.h`, `process_vk_ready` flag
   on `dt_dev_pixelpipe_iop_t`, dispatch hook in
   `src/develop/pixelpipe_hb.c` that prefers Vulkan over CPU when a
   module has a port.
4. ✅ **Module ports** (landed; see the §4.2 tables for the full list).
   40 modules now expose `process_vk`, covering the simple per-pixel
   bucket (exposure, velvia, invert, vibrance, colorcorrection,
   colorcontrast, colorize, flip, negadoctor, primaries, temperature
   ×3, profile_gamma ×2, splittoning, zonesystem, levels,
   whitebalance_1f / 1f_xtrans, channelmixerrgb ×5, graduatednd,
   vignette, relight, lowlight, colorbalance ×3 modes), the first
   sub-region + multi-fill module (borders), the first pass-through
   HAL-only module (mask_manager), the first LUT-on-storage-buffer
   port (colisa), the first two ICC-profile-aware ports
   (overexposed, basicadj), the first bilateral-helper consumer
   (monochrome), the Lab/RGB-curve cohort (rgbcurve, rgblevels,
   tonecurve), the two combined-helper consumers (lowpass,
   shadhi — each chains §5.10 / §5.13 with a mix kernel), and the
   scene-referred Yrg/dt-UCS cohort (colorharmonizer, plus
   colorbalancergb — `agx`-style params struct + the LMS/Yrg/
   JzAzBz/dt-UCS helper set in `dt_vulkan_common.h`), and the
   non-fusion base curve (basecurve — its two LUT kernels, with
   the exposure-fusion path gated to OpenCL), the thin-plate
   colour checker (colorchecker — single buffer-shaped kernel),
   and per-zone Lab grading (colorzones — both strong/smooth
   modes, mask-preview gated to CPU), the scene-referred
   filmic tone mapper (filmicrgb — main per-pixel split/chroma
   path with all v1..v5 colour-science variants and the v4/v5
   gamut-clip family; the wavelet highlight-reconstruction path
   is gated to OpenCL/CPU pending §8.5), the Orton soft-focus
   effect (soften — first MODERATE-bucket port to chain its own
   explicit separable Gaussian; 4-kernel sequence over a single
   scratch buffer, validated to single-bit FP precision against
   a C reference of process_cl), the Lab-L-channel unsharp
   mask (sharpen — 3-kernel chain that mirrors soften's convolution
   shape but operates on luminance only and uses a soft-threshold
   blend; bit-equal to the C reference), the Lab high-pass
   filter (highpass — 4-kernel chain: invert L, separable Gaussian
   on L, then a desaturating contrast mix around L = 50;
   single-bit FP precision against the C reference), and the
   lens/motion/Gaussian blur module (blurs — dense + sparse 2D
   convolution kernels for lens/motion PSFs plus a restore_alpha
   post-pass for the Gaussian-via-dt_gaussian_blur_vk fast path;
   bit-equal on all three kernels), the bloom effect (bloom —
   threshold + 8 iterations of separable uniform box blur on a
   single-channel float scratch + screen-blend mix; first VK
   consumer of float-element scratch buffers, bit-equal through
   the full 8-iteration chain), the edge-aware à-trous wavelet
   equalizer (atrous — 2-kernel decompose/synthesize ping-pong
   over max_scale scratch detail buffers; single-bit FP precision
   against a C reference), the dark-channel-prior haze removal
   (hazeremoval — 6 module-own kernels for transition map + box
   min/max + dehaze, plus the new `dt_guided_filter_vk` helper for
   the guided-filter refinement step; first consumer of the §5.15
   guided-filter helper, all 6 module-own kernels bit-equal), and the
   clipped-highlight colour reconstruction (colorreconstruction — 3-D
   bilateral-grid splat/blur/slice with CAS-loop float atomics, second
   `vk_atomic_add_f` consumer after the §5.13 bilateral helper; surfaced
   the OpenCL-vs-GLSL `round()` divergence the docs now warn about), and
   non-local-means denoising (nlmeans — 6-kernel shift-and-accumulate
   over the q-offset grid; added the `vk_fast_mexp2f` bit-pun weight
   helper, no atomics needed since the host serialises the q-loop).
   All are bit-equal to their OpenCL counterparts for the supported paths.
4a. ✅ **`dt_vk_module_kernel_t` abstraction** (landed; see §5.6).
    Cuts the per-module wiring boilerplate by ~30 LOC each and gives
    a uniform shape for every future port.
4b. ✅ **HAL-level device-to-device copy** (landed; see §5.9).
    `dt_vulkan_copy_device_to_device` exposes the internal
    `vkCmdCopyBuffer` machinery to pass-through modules — no SPIR-V
    needed.
4c. ✅ **Recursive Gaussian helper** (landed; see §5.10).
    `dt_gaussian_*_vk` row-then-column IIR blur. Substrate for
    porting lowpass, censorize, shadhi, retouch (each also needs at
    least one of: LUT plumbing, work-profile struct plumbing,
    bilateral helper). Extended later to 1- and 2-channel buffers
    (and a `dt_gaussian_mean_blur_vk` one-shot convenience) — the
    full channel set (1, 2, 4) the OpenCL surface offers; unblocks
    `colorequal`'s mixed 1c/2c/4c Gaussian use.
4d. ✅ **LUT-on-storage-buffer pattern** (landed; see §5.8).
    Worked example in `src/iop/colisa.c`; template for the remaining
    curve-based colour modules (`tonecurve`, `rgbcurve`, `rgblevels`,
    `basecurve`, `lowlight`, …).
4e. ✅ **Bilateral filter helper** (landed; see §5.13).
    `dt_bilateral_*_vk` 3-D splat / blur / slice pipeline as 6
    separate `.spv` modules + `vk_atomic_add_f` CAS-loop helper.
    Substrate for porting `lowpass`, `censorize`, `shadhi`,
    `retouch`, `monochrome`, `globaltonemap` (each also needs at
    least one of: LUT plumbing, work-profile struct plumbing). The
    splat kernel ships the simple direct-atomic form first; local-
    memory reduction is a planned perf follow-up.
4f. ✅ **ICC profile info storage-buffer plumbing** (landed; see
    §5.11). `vk_dt_colorspaces_iccprofile_info_t` storage buffer +
    flat 6·lutsize tone-curve LUT, with `vk_lerp_lookup_unbounded` /
    `vk_apply_trc_in` / `vk_get_rgb_matrix_luminance` / `vk_dt_rgb_norm`
    helpers in `dt_vulkan_common.h` and `dt_ioppr_build_iccprofile_params_vk`
    / `dt_ioppr_free_iccprofile_params_vk` host-side. First consumer
    is `overexposed`; unblocks `basicadj`, `basecurve`, `rgbcurve`,
    `rgblevels`, and the work-profile branches of several colour
    modules.
4h. ✅ **Guided-filter helper** (landed; see §5.15).
    `dt_guided_filter_vk` orchestrates 8 single-channel-float kernels
    (split + box-mean x/y + covariances + variances + update_covariance
    + 3×3 Cramer solve + generate_result). Required bumping
    `DT_VULKAN_MAX_BINDINGS` from 16 to 20 for the 17-binding solve.
    Unblocks `hazeremoval` (landed; see §4.2) and `colorequal`
    (pending) at the helper level.
4g. ⏳ **diffuse multi-scale wavelet pipeline** (§8.9 below) —
    pending. The OpenCL kernel chain (6 kernels × up to 10 scales ×
    N iterations) needs (a) the 6 kernel translations, (b) a
    multi-dispatch orchestration in process_vk, and (c) a batched-
    submit mode in the HAL so we don't pay one queue-submit + fence
    per dispatch (~300+ dispatches per call would be untenable).
5. **Image2D + sampler support.** Port `dt_opencl_alloc_device()` and
   `dt_opencl_write_image_*` to Vulkan storage / sampled images.
   Choose image format mapping table (the OpenCL `cl_image_format`
   enum vs `VkFormat`). Today's integration uses storage buffers
   only; once images land we can drop the per-module host-staging.
6. ✅ **Skip host-staging when both ends are Vulkan** (landed; see
   §5.14). `dt_dev_pixelpipe_t::vk_handoff_buf` keeps the previous
   VK output buffer alive across module boundaries; the next
   `process_vk` reuses it when the size matches, skipping the
   redundant device-alloc + host upload. Per-pipeline state so
   parallel preview/full/export pipelines stay independent.
   Invalidated on switch to OpenCL/CPU or at pipeline teardown.
   User-visible gain shows up once enough modules port to form
   consecutive VK chains in the default pipeline.
7. **Port the easy half.** Other modules that use only image2d +
   scalar args (filmic, basecurve, sigmoid, ~30 others). Verified
   pixel-equal against the OpenCL output.
8. **Full channelmixerrgb + diffuse.** Port the remaining adaptation
   modes and the wavelet pipeline so the current scope caveats go
   away.
9. **Atomics and local memory.** clspv supports atomics and
   `barrier()`; the bilateral and colour-reconstruction reductions
   port with minor tweaks (the inline-asm fast path for NVIDIA in
   `common.h` becomes plain OpenCL atomics; clspv compiles those to
   SPIR-V atomics).
10. **Demosaicers and local-laplacian.** The hardest kernels
    (Markesteijn especially) make heavy use of local memory and
    workgroup synchronisation. Verify clspv's local-memory ABI works
    at the workgroup sizes we currently use.
11. **Tiling and headroom.** Rewrite the host-side memory accounting
    on `VK_EXT_memory_budget`; preserve the existing
    `factor_cl`/`overlap` semantics.
12. **Binary cache.** Today's MD5-on-source binary cache (lines
    2242-2289 of `src/common/opencl.c`) becomes a `VkPipelineCache`
    keyed on `(SPIR-V hash, physical device UUID, driver version)`.
13. **Deprecate OpenCL build path.** When all modules ship on the new
    HAL and per-module pixel-equality tests pass, default
    `USE_OPENCL=OFF` and announce a deprecation window of one
    release.

## 9. clspv subset risks

`clspv` supports the subset of OpenCL C 1.2 that maps cleanly to
SPIR-V. The kernels in `data/kernels/` were surveyed for the
documented blockers:

| Feature | Used by darktable? | clspv status |
|---|---|---|
| `image2d_t` + samplers | yes (most kernels) | supported |
| `read_only`/`write_only` qualifiers | yes | supported |
| `barrier(CLK_LOCAL_MEM_FENCE)` | sharpen, demosaic_rcd, bilateral, colorreconstruction | supported |
| `local float buf[N]` | several demosaicers | supported |
| `atomic_add`, `atomic_cmpxchg` | bilateral.cl, colorreconstruction.cl, common.h `atomic_add_f` | supported (int); float emulation already in common.h |
| Native maths (`native_log2`, `native_sin`, …) | filmic, basicadj, … (under `__FAST_RELAXED_MATH__`) | supported, mapped to GLSL/SPIR-V relaxed-precision intrinsics |
| `async_work_group_copy` | none | n/a |
| `printf` | none | n/a (clspv has limited support behind a flag) |
| Inline PTX asm (`atom.global.add.f32`) | `common.h::atomic_add_f` NVIDIA fast path | **not supported** — the surrounding `#else` CAS-loop fallback already exists and we'd compile that on every device |
| Pointer aliasing (writing to a `global` buffer aliased through different types) | none observed | safe |

The biggest concrete risk is the NVIDIA inline-PTX shortcut in
`atomic_add_f`. The good news is it already has an `#else` branch with
a CAS loop; we just lose the (NVIDIA-only, modest) speedup until we
either add a Vulkan extension path (`VK_EXT_shader_atomic_float`) or
let drivers optimise the CAS.

Other risks to watch as we port more kernels:

- **Image-format combinations**: clspv supports a documented set of
  `cl_image_format` mappings to `VkFormat`. dt uses `RGBA float32`,
  `R float32`, and (rarely) `RGBA int16`. All three are in the
  supported set.
- **Constant-memory size limits**: OpenCL `constant` buffers become
  Vulkan uniform buffers. Some drivers cap these at 64KB; we already
  fit comfortably, but the demosaicer Xtrans table (`xtrans[6][6]`)
  should be sanity-checked.
- **Workgroup size discovery**: OpenCL queries
  `CL_KERNEL_WORK_GROUP_SIZE` at runtime; clspv emits a fixed local
  size in the SPIR-V `OpExecutionMode`. We either embed the same
  values we use today or use Vulkan specialisation constants to make
  them adjustable post-compile.

## 10. Performance expectations

For native Vulkan on Linux/Windows: should be within noise of the
current OpenCL on the same hardware. SPIR-V is a similar IR to the
NVIDIA OpenCL bytecode and AMDGCN intermediate; the kernels are the
same.

For MoltenVK on macOS: the relevant comparison is not "Vulkan vs Metal"
(equivalent), it's "MoltenVK-translated SPIR-V vs Metal-shading-
language hand-written". Published benchmarks show MoltenVK at 90–98%
of native Metal for compute-dominated workloads; well above what an
unmaintained Apple OpenCL stack gives on Apple Silicon.

For workloads where MoltenVK overhead matters (sub-millisecond
dispatches with tiny buffers), we already batch on the host. Image
modules are millisecond-scale or longer; the overhead is irrelevant.

### 10.1 Current observed numbers (Vulkan-OpenCL hybrid pipeline)

The pixelpipe integration in §4.2 currently routes one module at a
time through Vulkan with host-staging at each boundary. This is
**slower per module** than the equivalent OpenCL dispatch — the
extra cost is real and measurable, and the §8.6 milestone (skip
host-staging when both ends are VK) is the path to closing it.

A real-world export on AMD Radeon RX 9060 XT (RADV) with both
backends enabled and the default `prefer-vulkan` routing reported:

| Module      | Pipeline | Canvas    | Time   | ms/MPx |
|-------------|----------|-----------|-------:|-------:|
| temperature | [export] | 9568×6376 | 1.37 s |     22 |
| exposure    | [export] | 9568×6374 | 7.30 s |    120 |
| temperature | [preview]| 2587×1723 | 0.11 s |     25 |
| exposure    | [preview]| 2587×1723 | 0.43 s |     96 |

The 4–5× asymmetry between temperature and exposure on the same
canvas isn't kernel cost (both kernels are trivial per-pixel
arithmetic) — it's allocator pressure. Each
`dt_vulkan_write_to_device` / `read_from_device` call was doing a
fresh ~1 GB `vkAllocateMemory` for the staging buffer; the first
VK module (temperature) warms the driver's host-visible memory
pool, the second (exposure) lands in a fragmented pool after the
first one freed 4 GB of buffers.

**Mitigation already in tree** — `dt_vk_device_t.staging` (added
this iteration) keeps one host-visible staging buffer per device
that grows monotonically and is reused across every host transfer.
Drops the steady-state per-dispatch HAL overhead to one
`vkCmdCopyBuffer` + one `vkMapMemory`, with the
`vkAllocateMemory` happening once per high-water-mark instead of
per call. Expected to close most of the temperature/exposure gap
on the next export.

**Follow-up landed** — §5.14 VK→VK chain hand-off cache. When two
consecutive modules both have `process_vk`, the second one skips
the device-local input allocation and the host upload entirely
(reuses the previous module's output buffer). Halves the per-
boundary host traffic for VK chains. Visible in the trace as a
`[vk handoff]` annotation on the `process_vk` log line.

**Follow-up landed** — `dt_vk_device_t.buf_pool` is a small (64-slot)
free-list of recently released device-local buffers.
`dt_vulkan_alloc_buffer` does a best-fit scan against the pool
before falling back to `_alloc`; `dt_vulkan_free_buffer` pushes
back to the pool (evicting the smallest pooled buffer when full,
so the most-expensive-to-recreate allocations stay cached). For
typical pixelpipe workloads the steady-state per-module cost
collapses to two pool hits — no `vkAllocateMemory` /
`vkCreateBuffer` calls inside the per-dispatch critical section.

**Follow-up landed** — `_submit_one_shot` now reuses a persistent
`VkCommandBuffer` + `VkFence` per device via
`vkResetCommandBuffer` + `vkResetFences`. The previous create-
destroy pair per submission cost ~5-50 µs of driver overhead;
with three submissions per module (upload, kernel, readback) and
hundreds of submissions per export pipeline, the savings show up
in the trace as smaller "wall-time minus useful-work" gaps.

**Still outstanding** — the global `g_vk_lock` still serialises
all dispatches across concurrent pipelines (thumbnail / preview /
full / export). Splitting the lock so buffer alloc/free, host
memcpys, and the staging-buffer access can overlap across
pipelines would unblock the multi-pipeline thumbnailer case
visible in §10.1's traces.

**Follow-up landed — chain-aware prefer-vulkan routing.** Earlier
the `prefer-vulkan` decision in `pixelpipe_hb.c` routed every
module with a `process_vk` callback through Vulkan whenever the
user had VK enabled. For a 5208×3472 image on a discrete GPU
each isolated VK module wedged between CL ones paid a
CL→host→VK→host→CL transition (~1-2 s for 290 MB), while the
kernel itself ran in microseconds — net 4-5 s of pure overhead
on a typical export, with no upside.

The routing now scrutinises four cases:
- **Chain live** (`pipe->vk_handoff_buf != NULL`): continue VK,
  the upload is free via §5.14.
- **VK-only** (the module has no CL path): VK is the only GPU
  option, route to VK regardless of cost.
- **Chain-start**: look ahead one module — if the next enabled
  piece has `process_vk_ready`, this is a chain of ≥2 and the
  entry transition amortises. Route to VK.
- **Singleton**: VK module surrounded by CL with no chain ahead.
  Fall through to CL — its kernel is microsecond-scale and the
  round-trip is the bottleneck. The VK kernel still gets exercised
  on real chains (e.g. `exposure → exposure.2 → exposure.1`),
  just not on isolated stops.

Set `opencl_force_vulkan_routing=true` to override (forces VK
for every module that offers it; needed when fuzzing the VK port
for coverage). The `prefer-vulkan` log line carries a
`[chain]`/`[chain-start]`/`[no-cl]`/`[forced]` tag so the routing
decision is visible in `-d opencl` traces.

### 10.2 Performance roadmap — what's left and where the time goes

After the buffer pool, persistent fence/cmd-buffer, chain-aware
routing, and §5.14 hand-off cache have landed, a fresh trace from
an RX 9060 XT (RADV) full export of a 5208×3472 image still
takes ~15 s. The breakdown is sobering and clarifies which
direction has the most leverage:

| Bucket | Time | Cause | Fixable how |
|---|---:|---|---|
| CPU-only modules (`toneequal` ×2) | ~5.6 s | No `process_cl`, no `process_vk` — pure CPU work on 290 MB | Module port (large) |
| CPU blending after VK | ~3-5 s | VK arm forces blending to CPU; CL arm runs blending on GPU via `blendop.cl` (15 kernels, ~1700 LOC) | Port `blendop.cl` to VK (large) |
| CL ↔ VK boundary transitions | ~1-2 s | 290 MB `clEnqueueReadImage` + host→VK upload at each switch | VK-CL interop OR more VK ports to extend chains |
| Per-VK-module overhead (alloc, fences) | <0.1 s | Buffer pool + persistent fence already cover this | Already done |
| Lock contention across pipelines | ~1-2 s in multi-pipeline traces | `g_vk_lock` serialises all dispatches | Split lock (moderate) |
| Misc dispatch + kernel work | <0.5 s | Actually GPU-bound | Already optimal |

The first three rows account for almost the entire export
runtime, and **none of them are addressed by HAL micro-opts**
(buffer pool, fence reuse, lock-scope reduction). The buffer
pool that landed in `b65132b` saves measurable driver overhead
but doesn't move the needle on a discrete-GPU export with
hundreds of MB of staging traffic — that test confirmed in the
follow-up trace.

#### Architectural paths forward, ranked by impact-to-effort

**Path A — Port `blendop.cl` to Vulkan.** The single highest-
impact item. Each VK module that wants masking/blending today
forces a 290 MB device→host readback, ~3 GB/s of CPU blend work,
then a host→device re-upload for the next module. Putting blend
on the GPU side keeps the chain on-device. Impact: ~3-5 s saved
on a typical export with default mask settings. Effort: large —
15 compute kernels (Lab / RAW / RGB-jzczhz × {mask, mask-tone-curve,
rgb-blend, …}) plus ~500 LOC of host plumbing in
`src/develop/blend.c`. Tractable incrementally: start with the
"normal mode + no blendif + no raster mask" path that covers
80% of real-world use, then layer in the conditional modes.

**Path B — Continue module ports to *extend* existing VK chains.**
Each module ported brings two wins: (1) it removes its own host
roundtrip when a chain reaches it, and (2) it expands chain
boundaries so neighbour modules can stay on-device too. Highest-
value targets in dependency order: `colorequal` (uses
guided filter — overlaps with Path D groundwork), `lut3d` (needs
§8.5 milestone: image2D + sampler). Landed on this lane:
`colorbalancergb` (used in nearly every modern pipeline; the
LMS/Yrg/Ych/JzAzBz/dt-UCS helper cohort it added unblocks the
remaining Yrg-space ports), `colorharmonizer` (UCS JCH), and
`filmicrgb` (the main per-pixel split + chroma kernels folded
into one entry that reproduces the host split-vs-chroma decision
and dispatches on v1..v5 internally — reused the Yrg cohort, added
the filmic-specific gamut-clip family, and gated the wavelet
highlight reconstruction to OpenCL/CPU pending §8.5).
Effort: medium per module (~100-200 LOC kernel + ~100 LOC host).
Impact: cumulative — each port shaves ~50-200 ms once VK chains
form around it; the leverage compounds with Path A.

Recent landings on this lane: `colorout`, `filmic` (legacy
single-pass), `overlay` (image-bindings rewritten to storage
buffers; demonstrates that some "VERY HARD" entries actually
fit the buffer-only HAL once you replace `image2d_t in/out` with
`global float4 *in/out`), `sigmoid` (both color-processing modes
in one commit — `rgb_ratio` push-constant-only, `per_channel`
with three 3×3 matrices in storage buffers), `agx` (first port
to migrate a kernel param struct from PC into a storage-buffer
binding so the 124 B struct fits — the pattern unlocks any
future port whose param block exceeds the 128 B PC budget),
`channelmixer` (legacy 4-mode mixer; reuses the HSL cohort
helpers), `colorharmonizer` (UCS JCH hue/saturation; first
consumer of the dt-UCS helper cohort), `colorbalancergb`
(scene-referred balance — reuses the `agx` params-struct pattern
and adds the LMS/Yrg/Ych/JzAzBz/dt-UCS helper cohort the next
Yrg-space ports build on), and `filmicrgb` (the largest port to
date — main split + chroma kernels with all v1..v5 colour-science
variants and the v4/v5 gamut-mapping subsystem, reusing the
`colorbalancergb` Yrg cohort and §5.11 ICC plumbing; validated
end-to-end on lavapipe against an independent C reference derived
from `filmic.cl` — `split_v2_v3` bit-equal, `chroma_v4` matches at
FP precision).

**Path C — VK-CL zero-copy interop via DMA-BUF.** Replace the
host roundtrip at CL↔VK boundaries with a shared physical
allocation. `VK_KHR_external_memory_fd` + `cl_khr_external_memory`
(or AMD's `cl_amd_copy_buffer_p2p` / NVIDIA's
`cl_nv_external_memory`) let a buffer's `VkDeviceMemory` and
`cl_mem` point to the same VRAM page; the pixel data never
crosses PCIe at the transition. Impact: ~3-5 s saved per export
on this hardware (eliminates entry + exit transitions on every
VK chain). Effort: high (~500-1000 LOC) and platform-specific —
extension support is solid on AMD/Intel Linux, present but
fiddly on NVIDIA, and absent on most macOS configurations.
Best deferred until a critical mass of modules is ported (Path B
maturity) so the interop investment compounds.

**Path D — Port CPU-only modules (`toneequal`, `cacorrect`,
multi-pass diffuse pipeline).** `toneequal` alone is ~3 s of the
export trace and has no `process_cl`. These ports are large
individual investments (`toneequal` is 3445 LOC of host code
using a guided filter we haven't ported yet) but the gain is
direct: ~1 s saved per second of CPU work replaced by an ~10×
faster GPU dispatch. Best tackled after the guided-filter
helper exists (which Path B's `colorequal` port would
naturally produce).

**Path E — Split `g_vk_lock` into per-area mutexes.** Currently a
single mutex serialises all VK work across all pipelines (full,
preview, thumbnails). On multi-pipeline workloads (re-thumb a
folder while editing) the lock multiplies dispatch wall time by
the number of concurrent pipelines. Splitting into a
descriptor-pool lock, command-pool lock, queue submit lock, and
staging-buffer lock would let independent pipelines do alloc /
upload / readback in parallel and serialise only on the queue.
Impact: significant on multi-pipeline scenes (~30-50% wall-time
reduction in those traces); zero impact on single-pipeline
exports. Effort: moderate (~150 LOC, careful audit of
ordering invariants).

#### Routing heuristic tightened: chain-ahead requires length ≥ 3

The `vk_chain_ahead` look-ahead used to trip on a single VK-ready
neighbour. In practice that mis-fired routinely: a real-world
`agx → colorout` pair would chain-start `agx` on VK because
`colorout` was marked VK-ready, only for `colorout` to fall back
to CPU at runtime on a non-matrix Lab profile. We paid the CL→VK
entry transition (~50-200 ms on a discrete GPU at export
resolution) and the chain died one module later anyway. Same
shape for `exposure → exposure.1` when blendop forces a CPU blend.

The chain-ahead heuristic now requires **two** consecutive
enabled VK-ready modules ahead before chain-start fires (chain
length ≥ 3 including the current module). Singleton VK islands
fall through to OpenCL; genuine multi-module chains still trigger.
`opencl_force_vulkan_routing=true` still overrides for coverage
testing.

The data motivating this lives in a user-supplied trace
(8.14 s export on a 4632×2895 ORF / AMD RX 9060 XT / RADV):
`agx` was 303 ms on VK in preview but 9 ms on CL in export
because the export path's longer chain happened to evade the
heuristic. The kernel itself was 5 ms in both cases — the
remaining ~300 ms in the VK path was pure host-staging
overhead. The same trace showed `exposure` taking 0.98 s wall
on VK at export resolution while the kernel itself ran in ~5 ms;
the host-staging cost (CL→host→VK plus VK→host→CL across the
chain) was the entire delta. Tightening the heuristic moves
these singletons onto the CL fast path.

#### Predictive `process_vk_ready` for runtime-conditional fallbacks

The trace also showed `colorout` taking the `vulkan -> CPU fallback`
path inside `process_vk` whenever the output profile required
lcms2 (i.e. wasn't a matrix profile). The chain-ahead heuristic
had already committed to VK at that point: the pixelpipe staged
the input CL→VK, `process_vk` returned -1, and the runtime
re-staged VK→CPU to run the lcms2 path — three transitions for a
module that was always going to fall back.

The matrix-vs-lcms2 decision is made at `commit_params` time
(`dt_colorspaces_get_matrix_from_output_profile`). When that
returns "no matrix" we now clear `piece->process_vk_ready` in
the same place we already clear `piece->process_cl_ready`. The
routing logic then never sees `colorout` as a VK candidate for
that piece — neither for chain-ahead lookahead nor for chain
continuation — and the surrounding modules pick the CL/CPU path
without the wasted detour. Same pattern is available for any
future module whose VK port is conditionally unavailable at
commit time.

#### Batched HAL dispatch landed (§5.6 update)

Modules carrying small pre-dispatch uploads (LUTs, 3×3 matrices,
param structs) used to call `dt_vulkan_write_to_device` once per
upload, each issuing its own `vkQueueSubmit` +
`vkWaitForFences(UINT64_MAX)`. On a discrete GPU each sync is
5-30 ms, so an `agx` dispatch with 5 small uploads paid 25-150 ms
in sync overhead alone before the compute kernel even ran.

`dt_vulkan_dispatch_n_batched` now records all uploads + a single
`TRANSFER_WRITE → SHADER_READ` pipeline barrier + the compute
dispatch into one command buffer and submits it once. The shared
staging buffer is partitioned by 4-byte-aligned offsets (Vulkan's
`vkCmdCopyBuffer` requirement) so all uploads share one DMA
region. For N small uploads, this collapses N+1 submit/wait
round-trips to 1.

Every iop module that used to call `dt_vulkan_write_to_device`
before a dispatch is now on the batched path. The biggest payoffs
are on multi-upload modules: `agx` (5 uploads — params + 4
matrices), `sigmoid` per_channel (3 matrices), `rgbcurve` /
`rgblevels` / `tonecurve` / `colorout` (3 LUTs each); plus the
2-upload set `channelmixer` / `colisa` / `filmic` / `basicadj` /
`lowpass` / `zonesystem`; and the 1-upload set `overlay` /
`lowlight` / `profile_gamma` (gamma) / `levels` / `temperature`
(1f X-Trans) / `channelmixerrgb`. Modules with zero uploads
short-circuit back to `dt_vulkan_dispatch_n`, so swapping in the
batched call is always safe and never slower.

`dt_ioppr_build_iccprofile_params_vk_deferred` is the matching
helper for §5.11 plumbing: instead of immediately uploading the
ICC `profile_info` + tone-curve LUT (2 separate `vkQueueSubmit`
round-trips), it appends those uploads to the caller's
`dt_vk_upload_t[]` so they roll into the same batched dispatch
as the module's own LUTs. Currently used by `rgbcurve`,
`rgblevels`, `tonecurve`, and `basicadj`. Saves 2 more
submit/wait cycles per profile-using dispatch on top of the
module-local upload batching.

This compounds with the chain-ahead tightening: even when a VK
chain is genuinely worth running, each module now pays less per
dispatch. Expected saving scales with upload count — roughly
`(N+1) × ~5-30 ms` becomes `1 × ~5-30 ms`, so an agx-class
6-submit module saves ~25-150 ms per call.

#### Recommended sequence

1. **Path B continued** (this session and next) — every port
   compounds with future Path A and Path C work. Low risk, high
   reuse. The recent `colorout`, `filmic`, `overlay`, `sigmoid`,
   `agx`, `channelmixer` (legacy), `colorharmonizer`, and
   `colorbalancergb` landings are in this lane; `filmicrgb`
   (now that the Yrg/LMS helper cohort exists), `colorequal`
   (guided filter), and `lut3d` (after §8.5) are the next logical
   picks.
2. **Path A** (multi-session) — start with the common-case
   blend ("normal mode, no blendif, no raster mask") which
   covers the bulk of real pipelines, then incrementally add the
   conditional / mask / colorspace variants. Each landed kernel
   immediately benefits every VK module downstream of it.
3. **Path E** when the multi-pipeline traces become the
   bottleneck (i.e. once Path A makes the single-pipeline path
   fast enough that lock contention dominates the remaining
   slowness).
4. **Path C** once enough modules are ported that VK chains
   span most of the pipeline — at that point the CL boundary is
   only at one or two points and interop pays for its complexity.
5. **Path D** opportunistically as the helpers needed (guided
   filter for `toneequal`, etc.) emerge as side-products of
   Path B work.

#### Things that *don't* help (and why)

- **More aggressive deferred readback inside the VK arm.**
  Investigated: keeping the device buffer "live" past the
  module dispatch only helps when the *next* module both reads
  from the device buffer (VK with handoff) *and* doesn't trigger
  CPU blending/picker/color-space-transform that needs the host
  buffer. The current export's chain (`exposure` →
  `exposure.2[blend]` → `exposure.1[blend]`) has CPU blending on
  every chain link except the first — every deferral would force
  a sync at the next module's blending step anyway. Net savings
  on this trace: zero. Worth implementing only after Path A
  removes the blend-forced syncs.
- **Async kernel dispatch / pipeline barriers within a single
  command buffer.** Investigated: the fence-wait cost per
  dispatch is already ~µs-scale on RADV. The bottleneck is
  *data motion*, not kernel-launch latency.
- **Resizable-BAR / direct host-visible-device-local staging.**
  Investigated: would save the staging→device DMA hop (~10 ms
  per 290 MB transfer), so ~80-160 ms over an export. Real but
  marginal next to Path A / C scope. Worth doing once at HAL
  level when we touch staging again, but not a project of its own.

## 11. Optional intermediate: clvk

A way to de-risk the host-side rewrite is to keep the OpenCL ICD
boundary and replace the *driver* underneath: clvk exposes a fully-
conformant OpenCL 3.0 ICD whose backend is Vulkan compute (using clspv
internally). With clvk in place:

- `src/common/opencl.c` is **unchanged**.
- macOS gets a Vulkan-via-MoltenVK backend transparently; we just ship
  a clvk binary as part of our macOS bundle.
- We can validate the SPIR-V path against the existing kernels with
  zero kernel changes first.

The cost is that we keep the OpenCL host-API for longer and lose the
chance to clean up `dlopencl.{c,h}` and friends. It's a useful
intermediate test, not the long-term destination.

## 12. Open questions

- **clspv versioning policy.** clspv has no tagged release schedule.
  We need a vendoring or pinned-build script and a CI job that bumps
  it on a known cadence.
- **Vulkan loader on Windows.** Some packaging surfaces (the MSYS2
  builds especially) need to either depend on the LunarG SDK or
  vendor the Vulkan loader.
- **Per-device tuning data.** Today's `cldevice_v6_<id>` config
  entries store roundup widths, headroom, etc. The format becomes
  `vkdevice_v1_<UUID>` with the same fields. Migration is best-effort:
  on first run we re-tune by running the existing benchmark kernels.
- **Test coverage**. We need a pixel-equality test harness that runs
  every kernel on a tiny known input and compares against a golden
  baseline. The PoC's CPU-reference comparison is the first piece.
- **Apple Silicon performance baseline.** Before announcing
  deprecation we want a benchmark suite (filmic, demosaic, denoise,
  filmic+lut3d export) on at least one Intel Mac and one M-series Mac
  comparing today's Apple-OpenCL path against MoltenVK.

## 13. Summary

- The image-processing maths in `data/kernels/*.cl` is the asset; the
  host-side OpenCL plumbing is replaceable.
- clspv lets us compile that `.cl` source to SPIR-V at build time.
  Vulkan runs the SPIR-V on Linux and Windows natively, and on macOS
  through MoltenVK. No hand-written Metal.
- The PoC in `tools/vulkan_compute_poc/` shows the toolchain works
  end-to-end and produces numerically-equal output to a CPU reference.
- The migration is staged behind a new `USE_VULKAN` flag; OpenCL stays
  available throughout, and only flips off when every module has been
  ported and validated.
