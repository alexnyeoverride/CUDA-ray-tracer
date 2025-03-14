#pragma once

#include "vector.cuh"

struct Ray {
	// double4 instead of double3, because SIMD.
	double4 origin;
	double4 direction;
	double4 color;

	__device__ __host__ double4 at(const double t) const {
		return origin + direction * t;
	}
};