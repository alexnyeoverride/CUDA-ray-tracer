#include "image.cuh"
#include <cstdio>
#include <iostream>

void Image::save() const {
    const auto file = fopen("output.ppm", "wb");
    if (file == nullptr) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return;
    }

    fprintf(file, "P3\n%d %d\n255\n", width, height);

    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            const auto color = at(j, i);
            fprintf(file, "%d %d %d\n", color.x, color.y, color.z);
        }
    }
}
