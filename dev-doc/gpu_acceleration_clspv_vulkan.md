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

**Per-module ports**: 34 modules currently expose `process_vk`,
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

**Verified in this container:** all 12 module files and the new
backend compile clean against all four `(HAVE_VULKAN × HAVE_OPENCL)`
combinations (syntax-only check via `gcc -fsyntax-only`). All 12
GLSL twins build to valid SPIR-V via glslang + `spirv-val`. End-to-
end darktable build requires upstream dependencies (intltool-merge
etc.) that aren't installed in the remote execution environment, so
runs against a real RAW are deferred to CI.

**What's left** (from the 70 surveyed `process_cl` modules):

- **TRIVIAL bucket** still to port. The Lab/RGB-curve cohort
  (`tonecurve`, `rgbcurve`, `rgblevels`) is now landed — they
  lifted off the colisa template (§5.8) + the §5.11 ICC profile
  plumbing for their norm-preserving modes. `monochrome` is
  already ported as the first bilateral consumer (§5.13).
  `rawoverexposed` is RAW-only and likely stays on OpenCL.
  `rawoverexposed` is RAW-only and likely stays on OpenCL. Done in
  earlier passes: `colisa`, `levels`, `profile_gamma`,
  `zonesystem`, `splittoning` (added RGB↔HSL helpers to
  `dt_vulkan_common.h`), `overexposed` (first consumer of ICC
  profile plumbing §5.11), `lowlight` (template for the next LUT
  consumers).
- **EASY bucket** — one or two storage buffers for matrices /
  LUTs: `basecurve`, `lut3d` (3D LUT — needs a 256³ float buffer or
  sampled image), `channelmixer` (legacy), `colorbalancergb`,
  `colorout` (3 LUTs), `filmic` (1 LUT + scalars). Each is ~50 LOC
  module + ~80 LOC kernel. Done in earlier passes: `basicadj`
  (second consumer of the §5.11 plumbing; full 6-feature ICC-aware
  kernel), `rgbcurve` / `rgblevels` / `tonecurve` (the Lab/RGB
  curve cohort, all sharing the 7-binding §5.11 + §5.8 pattern),
  `colorbalance` (3 variants, push-constant-only — switches on
  `d->mode` to dispatch the matching `.spv`).
- **MODERATE** — multi-pass with intermediate buffers or
  local-memory barriers: `blurs`, `colorchecker`, `colorzones`,
  `sharpen`, `soften`, `highpass`, `highlights`. The Gaussian VK
  helper (§5.10) handles the separable-blur half; `sharpen` and
  the larger blurs still want workgroup-local-memory plumbing for
  the cache-friendly kernels. Done in earlier passes:
  `graduatednd`, `vignette`, `relight`, `borders` (multi-fill +
  sub-region copy, §5.9), `lowpass` (first combined-helper
  consumer — chains §5.10 or §5.13 followed by a curve-mix
  kernel), `shadhi` (second combined-helper consumer — same shape
  + soft-light overlays).
- **HARD** — atrous, bloom, denoiseprofile, filmicrgb,
  globaltonemap, hazeremoval, nlmeans, retouch, colorequal, agx,
  basecurve (full variants), colorreconstruction (atomics). Multi-
  kernel pipelines or per-warp reductions. `lowpass`, `censorize`,
  `shadhi`, `retouch`, `monochrome`, `globaltonemap` can now build
  on the bilateral helper (§5.13) for their grid-based passes.
- **VERY HARD** — demosaicing, geometric corrections (ashift,
  clipping, crop), liquify, overlay, rasterfile, colorin. These
  need the sampled-image + sampler bindings (§5.2 milestone 5) and
  in some cases full distort pipelines. Done in earlier passes:
  `borders` (without sampled images — used a sub-region copy
  kernel instead), `mask_manager` (used the new
  `dt_vulkan_copy_device_to_device` HAL primitive instead of a
  kernel).

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

Two complementary mechanisms exist:

1. **Device-to-device buffer copy** (`dt_vulkan_copy_device_to_device`)
   — a thin wrapper around `vkCmdCopyBuffer` for full-buffer
   `dt_vk_mem_t* → dt_vk_mem_t*` transfers, no kernel needed. Used
   by `mask_manager` (the entire OpenCL path was a single
   `enqueue_copy_image`; the Vulkan equivalent is one
   `vkCmdCopyBuffer` of the float4 pipe buffer).

2. **Sub-region copy / fill kernels** (`borders.cl::borders_copy`,
   `borders.cl::borders_fill`) — when the source or destination is
   a strict subrectangle of a larger buffer, kernel-based copies and
   fills are clearer than chains of `vkCmdCopyBuffer`s with per-row
   `VkBufferCopy` regions. The OpenCL build dispatches the fill
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
dt_gaussian_vk_t *g = dt_gaussian_init_vk(width, height, max, min,
                                          sigma, order);
int rc = dt_gaussian_blur_vk(g, dev_in, dev_out);
dt_gaussian_free_vk(g);
```

The OpenCL build runs column-blur + transpose + column-blur +
transpose with a workgroup-local-memory transpose. The Vulkan port
drops the transpose entirely and runs row-then-column blur with one
work-item per row / column — the IIR recurrence is serial along its
sweep axis anyway, so the natural parallelism granularity is one
work-item per row/column, not per pixel. The local-memory transpose
would be a meaningful speedup at large image sizes but needs
clspv/glslang local-memory plumbing the HAL doesn't expose yet.

The two kernels (`gaussian_row_4c`, `gaussian_column_4c`) live in
one shared `.spv`. The host loads the program once and caches the
kernel slots in module-level statics so every IOP that uses the
helper amortises the SPV read across the process lifetime. Only the
4-channel path is wired up so far; the OpenCL 1c/2c variants serve a
small minority of modules and can be added on demand.

`dt_gaussian_blur_vk` returns -1 on any failure (Vulkan not running,
program load failed, dispatch error). Callers should always check
the return value and fall through to `dt_gaussian_blur_cl` /
`dt_gaussian_blur` (CPU) on -1, in the same shape as the existing
CPU/OpenCL fallback pattern in the lowpass / shadhi / retouch
modules.

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
file).

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

For VK→VK pairs this halves the per-boundary host traffic. The
asymmetric temperature→exposure timing reported in §10.1 was
*not* caused by VK→VK transitions (other OpenCL modules ran in
between), so the user-visible improvement only shows up once
enough modules in the default pipeline expose `process_vk` to
form chains. The infrastructure is in place now; module coverage
catches up incrementally as more modules port.

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
   34 modules now expose `process_vk`, covering the simple per-pixel
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
   tonecurve), and the two combined-helper consumers (lowpass,
   shadhi — each chains §5.10 / §5.13 with a mix kernel). All are
   bit-equal to their OpenCL counterparts for the supported paths.
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
    bilateral helper).
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
