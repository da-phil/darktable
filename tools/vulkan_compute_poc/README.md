# Vulkan compute proof-of-concept

A small standalone target that demonstrates the proposed
`OpenCL → clspv → SPIR-V → Vulkan` pipeline on a stripped-down darktable
kernel. See `dev-doc/gpu_acceleration_clspv_vulkan.md` for the design
rationale this code supports.

## What it does

1. Compiles `kernels/basicadj_min.cl` (the production path, via `clspv`)
   or `kernels/basicadj_min.comp` (fallback path, via `glslangValidator`)
   into a SPIR-V module.
2. Validates the module with `spirv-val` if available.
3. Builds `dt_vk_compute_poc`, a Vulkan-compute host driver that loads
   the SPIR-V, allocates two storage buffers, dispatches the kernel,
   reads back, and compares against a CPU reference.

The same SPIR-V module will dispatch unchanged on:

- Linux (Mesa RADV/ANV, AMDVLK, NVIDIA proprietary),
- Windows (vendor ICDs),
- macOS (MoltenVK, which translates SPIR-V → MSL → Metal at runtime).

## Build (standalone)

```sh
cd tools/vulkan_compute_poc
cmake -S . -B build
cmake --build build
./build/dt_vk_compute_poc --spv build/basicadj_min.spv
```

Expected output:

```
vulkan device: <your GPU>
loaded N words of SPIR-V from build/basicadj_min.spv
PoC OK: 256x256 kernel result matches CPU reference
```

Exit code 77 means "no Vulkan device available, skipping" (useful in CI).

## Build (via darktable top-level)

```sh
cmake -S . -B build -DBUILD_VULKAN_COMPUTE_POC=ON
cmake --build build --target dt_vk_compute_poc
```

The option is `OFF` by default; standard darktable builds are
unaffected.

## Run options

```
--spv <path>       SPIR-V module to dispatch (required)
--validation       Enable VK_LAYER_KHRONOS_validation
--width N          Test image width  (default 256)
--height N         Test image height (default 1080 in 1080p check)
```

## clspv vs glslang fallback

If `clspv` is in `PATH`, the build uses it on `basicadj_min.cl` —
this is the path the real port will take. If only `glslangValidator`
is available, the build falls back to the GLSL twin
(`basicadj_min.comp`); the two must stay behaviour-equivalent (see
the comment block at the top of each file).
