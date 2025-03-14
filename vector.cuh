#pragma once

#include <array>
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


struct Matrix {
	std::array<double4, 4> matrix;

	__host__ __device__ Matrix operator*(const Matrix& other) const {
		auto result = Matrix {};
		for (int i = 0; i < 4; ++i) {
			const auto&[x, y, z, w] = matrix[i];

			result.matrix[i].x
			= x * other.matrix[0].x
			+ y * other.matrix[1].x
			+ z * other.matrix[2].x
			+ w * other.matrix[3].x
			;
			result.matrix[i].y
			= x * other.matrix[0].y
			+ y * other.matrix[1].y
			+ z * other.matrix[2].y
			+ w * other.matrix[3].y
			;
			result.matrix[i].z
			= x * other.matrix[0].z
			+ y * other.matrix[1].z
			+ z * other.matrix[2].z
			+ w * other.matrix[3].z
			;
			result.matrix[i].w
			= x * other.matrix[0].w
			+ y * other.matrix[1].w
			+ z * other.matrix[2].w
			+ w * other.matrix[3].w
			;
		}
		return result;
	}

	__host__ __device__ double4 operator*(const double4& rhs) const {
		return {
			dot(matrix[0], rhs),
			dot(matrix[1], rhs),
			dot(matrix[2], rhs),
			dot(matrix[3], rhs)
		};
	}
};