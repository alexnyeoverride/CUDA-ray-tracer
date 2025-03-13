#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// 4 instead of 3 for SIMD.
using Pixel = uchar4;

struct Image {
    int16_t height;
    int16_t width;

    Pixel* pixels;

    __host__ __device__ Pixel& at(const int x, const int y) const {
        return pixels[y * width + x];
    }

    void save() const;

    static Image create(const int16_t width, const int16_t height) {
        return {
            height,
            width,
            new Pixel[width * height]
        };
    }

    void destroy() const {
        delete[] pixels;
    }
};