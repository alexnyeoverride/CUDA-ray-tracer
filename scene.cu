#include "scene.cuh"

#include <limits>

__host__ __device__ bool Sphere::intersects(const Ray &ray, Ray &out_ray) const {
	// TODO: determine point, exit direction, and color for out_ray.
	const auto center = transform * double4{};
	const auto oc = center - ray.origin;
	const auto a = length(ray.direction);
	const auto b = dot(ray.direction, oc) * -2.0;
	const auto c = dot(oc, oc) - radius * radius;
	const auto discriminant = b * b - 4 * a * c;
	return discriminant >= 0.0;
}

__host__ __device__ bool Box::intersects(const Ray &ray, Ray &out_ray) const {
	// Not implemented.
	return true;
}

__global__ void destroyRecursively(const Scene& scene) {
	free(scene.spheres);
	free(scene.boxes);
	for (auto i = 0; i < scene.num_scenes; i++) {
		destroyRecursively<<<1, 1>>>(scene.scenes[i]);
	}
	free(scene.scenes);
}