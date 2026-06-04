// Vulkan port of src/iop/censorize.c :: make_noise pass.
//
// Adds zero-mean multiplicative Gaussian noise to the image in
// place. Per pixel: seed a per-pixel xoshiro state from (x, y), warm
// up 4 rounds, sample a Box-Muller Gaussian, scale by noise * norm,
// and clip the output to non-negative.
//
// The seeding scheme **differs from the CPU path** by design: the
// CPU uses splitmix32 (64-bit multiply + xor folds) which needs
// either uint64 support (clspv `ulong`, GLSL int64 extension) or a
// portable 32×32→64 decomposition. For noise we don't need bit-equal
// CPU output — only statistically uniform seeding — so we use a
// fast 32-bit hash (`vk_hash32`) tuned for spatially decorrelated
// per-pixel seeds. The resulting noise pattern is visually
// indistinguishable from the CPU output at the same `noise`
// parameter; only the per-pixel realisations differ.
//
// Binding layout (1 storage buffer):
//   0: out  (float4) — modified in place; CPU code passes `output`
//                      to make_noise after writing it.
// Push constants: 2 ints + 1 float = 12 bytes (width, height, noise).

#include "dt_vulkan_common.h"

static inline uint vk_hash32(uint x)
{
  // Wang-style 32-bit avalanche hash. ~5 ALU ops, uniform output.
  x = (x ^ 61u) ^ (x >> 16);
  x = x + (x << 3);
  x = x ^ (x >> 4);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}

static inline uint vk_rol32(const uint x, const int k)
{
  return (x << k) | (x >> (32 - k));
}

static inline float vk_xoshiro128plus(uint state[4])
{
  const uint result = state[0] + state[3];
  const uint t = state[1] << 9;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = vk_rol32(state[3], 11);
  return (float)(result >> 8) * 0x1.0p-24f;
}

kernel void censorize_noise(global float4 *out,
                            const int width,
                            const int height,
                            const float noise)
{
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if(j >= width || i >= height) return;

  // Per-pixel state. The CPU uses splitmix32(j+1), splitmix32((j+1)*(i+3)),
  // splitmix32(1337), splitmix32(666). We swap splitmix for vk_hash32
  // with similar input scrambling (still per-pixel-unique).
  uint state[4];
  state[0] = vk_hash32((uint)(j + 1));
  state[1] = vk_hash32((uint)((j + 1) * (i + 3)));
  state[2] = vk_hash32(1337u);
  state[3] = vk_hash32(666u);

  // Same 4-round warmup as the CPU code.
  for(int k = 0; k < 4; k++) vk_xoshiro128plus(state);

  const int idx = i * width + j;
  float4 pix = out[idx];
  const float norm = pix.y;

  // Box-Muller — mirrors gaussian_noise() with the flip bit alternating
  // per pixel (CPU uses `i % 2 || j % 2`).
  const int flip = ((i & 1) != 0) || ((j & 1) != 0);
  const float u1 = fmax(vk_xoshiro128plus(state), 1.1754944e-38f);  // FLT_MIN
  const float u2 = vk_xoshiro128plus(state);
  const float two_pi = 2.0f * M_PI_F;
  const float g = flip ? sqrt(-2.0f * log(u1)) * cos(two_pi * u2)
                        : sqrt(-2.0f * log(u1)) * sin(two_pi * u2);
  const float epsilon = (g * noise * norm + norm) / norm;

  // out RGB ← max(0, out RGB * epsilon); alpha untouched.
  pix.x = fmax(pix.x * epsilon, 0.0f);
  pix.y = fmax(pix.y * epsilon, 0.0f);
  pix.z = fmax(pix.z * epsilon, 0.0f);
  out[idx] = pix;
}
