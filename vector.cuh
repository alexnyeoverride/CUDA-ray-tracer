#pragma once

#include <cuda_runtime.h>

__device__ __host__ inline double4 operator+(const double4& a, const double4& b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__ __host__ inline double4 operator-(const double4& a, const double4& b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__device__ __host__ inline double4 operator*(const double4& a, const double& scale) {
	return {a.x * scale, a.y * scale, a.z * scale, a.w * scale};
}

__device__ __host__ inline double dot(const double4& a, const double4& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __host__ inline double length(const double4& a) {
	return sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2) + pow(a.w, 2));
}

__device__ __host__ inline double4 operator/(const double4& a, const double4& b) {
	return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

__device__ __host__ inline double4 operator/(const double4& a, const double scale) {
	return {a.x / scale, a.y / scale, a.z / scale, a.w / scale};
}

__device__ __host__ inline double4 normalize(const double4& a) {
	return a / length(a);
}