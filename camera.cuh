#pragma once

#include "vector.cuh"
#include "image.cuh"

struct Scene;

struct Camera {
	double focal_length = 1.0;
	double4 center = { 0.0, 0.0, 0.0, 0.0 };

	__host__ Image capture(const Scene& scene) const;
};
