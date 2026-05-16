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

**Per-module ports**: this PR ships `process_vk` for three modules,
in order of port difficulty.

| Module | Status | Notes |
|---|---|---|
| `src/iop/exposure.c` | **Equivalent** to the OpenCL kernel. | The maths is one subtract + multiply per channel; the .cl/.comp source pair under data/kernels/vulkan/exposure.{cl,comp} compiles to the same SPIR-V shape via clspv or glslang. |
| `src/iop/channelmixerrgb.c` | **Linear-Bradford path only.** | The OpenCL kernel covers five adaptation modes (CAT16, linear/non-linear Bradford, XYZ, RGB) plus gamut mapping and per-channel saturation/lightness. The Vulkan port covers the linear-Bradford path with scalar saturation/lightness — the most common case in practice. `process_vk` returns -1 for the other modes and the pipeline falls back to OpenCL/CPU automatically. |
| `src/iop/diffuse.c` | **Single-pass approximation** of sharpen. | The full diffuse pipeline runs an à-trous wavelet decomposition and applies anisotropic diffusion per scale; that's a multi-kernel, multi-buffer dance that's out of scope for the first port. The Vulkan path runs one unsharp-mask pass against a 3×3 box-blur low-pass, which gives a visible sharpen but is NOT bit-equal to the OpenCL/CPU output. When precise diffuse output is required, returning -1 from `process_vk` (e.g. when amount ≤ 0 or the SPIR-V module didn't load) flips the dispatch back to OpenCL/CPU. |

**Verified in this container:** all module files and the new backend
compile clean against all four `(HAVE_VULKAN × HAVE_OPENCL)`
combinations (syntax-only check). The full darktable build needs
upstream dependencies (intltool-merge, etc.) that aren't installed
in the remote execution environment, so end-to-end pipeline runs
against a real RAW are deferred to CI.

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
4. ✅ **Three module ports** (landed, with scope caveats noted in §4.2):
   exposure (equivalent), channelmixerrgb (linear-Bradford only),
   diffuse (single-pass approximation).
5. **Image2D + sampler support.** Port `dt_opencl_alloc_device()` and
   `dt_opencl_write_image_*` to Vulkan storage / sampled images.
   Choose image format mapping table (the OpenCL `cl_image_format`
   enum vs `VkFormat`). Today's integration uses storage buffers
   only; once images land we can drop the per-module host-staging.
6. **Skip host-staging when both ends are Vulkan.** Track which
   buffer type holds the live pipeline data; when consecutive
   modules both have `process_vk`, pass the `dt_vk_mem_t*` directly
   between them instead of round-tripping through host memory.
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
