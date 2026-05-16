/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Driver for the clspv/Vulkan proof-of-concept.

    Steps:

      1. Generate a synthetic 4-channel image on the host.
      2. Compute the CPU reference (the same exposure/black-point step
         a darktable IOP would do).
      3. Upload to a device buffer, dispatch the SPIR-V kernel, read
         back.
      4. Compare reference vs. GPU result.

    The SPIR-V module path is passed on the command line. The build
    system produces one from kernels/basicadj_min.cl (clspv) when clspv
    is present, otherwise from kernels/basicadj_min.comp (glslang). Both
    paths must produce numerically equivalent results.
*/

#include "vk_compute.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct PushConstants {
    int32_t width;
    int32_t height;
    float   black_point;
    float   scale;
};

void cpu_reference(const float* in, float* out, int w, int h, float black, float scale) {
    for (int i = 0; i < w * h; ++i) {
        out[4*i + 0] = (in[4*i + 0] - black) * scale;
        out[4*i + 1] = (in[4*i + 1] - black) * scale;
        out[4*i + 2] = (in[4*i + 2] - black) * scale;
        out[4*i + 3] = in[4*i + 3];
    }
}

void make_test_image(std::vector<float>& buf, int w, int h) {
    buf.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = ((size_t)y * w + x) * 4;
            buf[i + 0] = 0.05f + 0.001f * (float)x;
            buf[i + 1] = 0.12f + 0.0005f * (float)y;
            buf[i + 2] = 0.22f + 0.0002f * (float)(x + y);
            buf[i + 3] = 1.0f;
        }
    }
}

bool buffers_close(const float* a, const float* b, size_t n, float eps) {
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            std::fprintf(stderr, "mismatch at %zu: cpu=%f gpu=%f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const char* spv_path = nullptr;
    bool validation = false;
    int w = 256, h = 256;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spv" && i + 1 < argc) spv_path = argv[++i];
        else if (a == "--validation") validation = true;
        else if (a == "--width"  && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) h = std::atoi(argv[++i]);
        else if (a == "--help") {
            std::printf("usage: %s --spv <path> [--validation] [--width N] [--height N]\n", argv[0]);
            return 0;
        }
    }

    if (!spv_path) {
        std::fprintf(stderr, "error: --spv <path> is required\n");
        return 2;
    }

    dt::vkpoc::Device dev;
    std::string err;
    if (!dev.init(validation, err)) {
        // Treat as skip rather than a CI red — useful when the runner
        // has no Vulkan ICD installed.
        std::fprintf(stderr, "vulkan init failed: %s\n", err.c_str());
        std::fprintf(stderr, "skipping PoC dispatch (no usable Vulkan device)\n");
        return 77; // automake-style "skip"
    }
    std::printf("vulkan device: %s\n", dev.device_name().c_str());

    std::vector<uint32_t> spirv;
    if (!dt::vkpoc::load_spirv(spv_path, spirv, err)) {
        std::fprintf(stderr, "load_spirv: %s\n", err.c_str());
        return 1;
    }
    std::printf("loaded %zu words of SPIR-V from %s\n", spirv.size(), spv_path);

    std::vector<float> input;
    make_test_image(input, w, h);
    std::vector<float> cpu_out(input.size(), 0.0f);
    const float black = 0.04f;
    const float scale = 1.7f;
    cpu_reference(input.data(), cpu_out.data(), w, h, black, scale);

    const VkDeviceSize bytes = input.size() * sizeof(float);
    auto buf_in  = dev.alloc_buffer(bytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    auto buf_out = dev.alloc_buffer(bytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
    dev.upload(buf_in, input.data(), bytes);

    // The kernel uses (16,16,1) workgroups in the GLSL twin; the clspv
    // output uses (1,1,1) by default which the runtime would still
    // dispatch correctly, just at lower occupancy. We pass the values
    // the build script captured into a small descriptor sidecar; for the
    // PoC we hardcode and rely on the dispatch rounding.
    const uint32_t lx = 16, ly = 16, lz = 1;

    auto pipe = dev.create_pipeline(spirv, "main",
                                    /*storage buffers*/ 2,
                                    /*push constants  */ sizeof(PushConstants),
                                    lx, ly, lz);

    dev.bind_buffers(pipe, { &buf_in, &buf_out });

    PushConstants pc{};
    pc.width       = w;
    pc.height      = h;
    pc.black_point = black;
    pc.scale       = scale;
    dev.dispatch(pipe, &pc, (uint32_t)w, (uint32_t)h, 1u);

    std::vector<float> gpu_out(input.size(), 0.0f);
    dev.download(buf_out, gpu_out.data(), bytes);

    dev.destroy_pipeline(pipe);
    dev.free_buffer(buf_in);
    dev.free_buffer(buf_out);

    // 1 ULP-ish: clspv and glslang may emit fmuladd vs separate ops
    // depending on driver, so we tolerate ~1e-5.
    if (!buffers_close(cpu_out.data(), gpu_out.data(), gpu_out.size(), 1e-5f)) {
        std::fprintf(stderr, "PoC FAILED: CPU/GPU mismatch\n");
        return 1;
    }
    std::printf("PoC OK: %dx%d kernel result matches CPU reference\n", w, h);
    return 0;
}
