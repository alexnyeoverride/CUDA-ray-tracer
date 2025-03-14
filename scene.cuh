#pragma once

#include <cuda_runtime.h>

#include "ray.cuh"


const auto recursion_limit = 10;


struct Scene;
__global__ void destroyRecursively(const Scene& scene);

struct Object {
	Matrix transform;
};

struct Sphere : Object {
	double radius = 0.0;
	__host__ __device__ bool intersects(const Ray& ray, Ray& out_ray) const;
};

struct Box : Object {
	double length = 0.0;
	double width = 0.0;
	double height = 0.0;
	__host__ __device__ bool intersects(const Ray& ray, Ray& out_ray) const;
};

struct Scene : Sphere {
	static constexpr int8_t CAPACITY = 32;

	Box* boxes = nullptr;
	int8_t num_boxes = 0;
	Sphere* spheres = nullptr;
	int8_t num_spheres = 0;
	Scene* scenes = nullptr;
	int8_t num_scenes = 0;

	__host__ __device__ bool cast(Ray& ray) const;

	static __device__ Scene device_construct() {
		auto scene = Scene{};
		scene.boxes = static_cast<Box*>(malloc(sizeof(Box) * CAPACITY));
		scene.spheres = static_cast<Sphere*>(malloc(sizeof(Sphere) * CAPACITY));
		scene.scenes = static_cast<Scene*>(malloc(sizeof(Scene) * CAPACITY));
		return scene;
	}

	static __host__ Scene construct() {
		auto scene = Scene{};
		cudaMallocManaged(&scene.spheres, sizeof(Sphere) * CAPACITY);
		cudaMallocManaged(&scene.boxes, sizeof(Box) * CAPACITY);
		cudaMallocManaged(&scene.scenes, sizeof(Scene) * CAPACITY);
		return scene;
	}

	__host__ void destroy() {
		cudaFree(spheres);
		cudaFree(boxes);
		for (int i = 0; i < num_scenes; i++) {
			destroyRecursively<<<1, 1>>>(scenes[i]);
		}
		cudaFree(scenes);
	}

	__host__ __device__ void insert(const Sphere &sphere) {
		const auto sphere_center = sphere.transform * double4{};
		const auto node = find_node_to_insert_into(sphere_center);

		if (node->num_spheres < CAPACITY) {
			node->spheres[node->num_spheres] = sphere;
			node->num_spheres++;
		} else {
			// TODO: find or create child scene to insert into.  Use the closest one with capacity.  Recurse (mind recursion_limit).  Then balance-swap.
			// TODO: reuse `intersects` on child nodes with a length-0 Ray.
		}
		// TODO: grow radius if needed.
	}

	// Other insertion method overloads not implemented.

private:

	// TODO: method to find overlapping child nodes and perform trades between them to minimize the radius of each.
	//		Else, have to spawn a new subtree (mind recursion_limit).  device_construct on device or construct on host.
	// TODO: for completeness, it should also balance the tree depths.

	// TODO: a method to find the bottom-most descendent node overlapping with a point.
	//		If it's already full, spawn a subtree (mind recursion_limit).  device_construct on device or construct on host.
	__host__ __device__ Scene* find_node_to_insert_into(const double4& center_of_object_to_insert) {
		return this;
		// TODO
	}
};