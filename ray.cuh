#pragma once

#include "vector.cuh"

struct Ray {
	// double4 instead of double3, because SIMD.
	double4 origin;
	double4 direction;

	__device__ __host__ double4 at(const double t) const {
		return origin + direction * t;
	}

	// TODO: rename colorize and call from camera code
	__device__ __host__ uchar4 color() const;
};