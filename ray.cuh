#pragma once

#include "vector.cuh"

struct Ray {
	// double4 instead of double3, because SIMD.
	double4 origin;
	double4 direction;

	__device__ __host__ double4 at(const double t) const {
		return origin + direction * t;
	}

	__device__ __host__ bool hit_sphere(const double4& center, const double radius) const {
		const auto oc = center - origin;
		const auto a = length(direction);
		const auto b = dot(direction, oc) * -2.0;
		const auto c = dot(oc, oc) - radius * radius;
		const auto discriminant = b * b - 4 * a * c;
		return discriminant >= 0.0;
	}

	__device__ __host__ uchar4 color() const {
		if (hit_sphere({0.0, 0.0, -1.0, 0.0}, 0.5)) {
			return uchar4{0, 0, 0, 0};
		}

		const auto alpha = 0.5 * (normalize(direction).y + 1.0);
		constexpr auto white = double4{1.0, 1.0, 1.0, 1.0};
		constexpr auto blue = double4{0.5, 0.7, 1.0, 1.0};
		const auto color = white * (1.0-alpha) + blue * alpha;
		return {
			static_cast<unsigned char>(color.x * 255.9999),
			static_cast<unsigned char>(color.y * 255.9999),
			static_cast<unsigned char>(color.z * 255.9999),
			1
		};
	}
};