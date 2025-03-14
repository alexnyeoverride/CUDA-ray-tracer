#include "scene.cuh"

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
	// TODO: determine point, exit direction, and color
	return true;
}


__host__ __device__ bool Scene::cast(Ray &ray) const {
	auto sqr_distance_to_closest_hit = INFINITY;
	auto out_ray = ray;
	auto ray_alive = true;

	const auto recursion_limit = 10; // TODO: prevent creating BVHs deeper than this elsewhere.
	auto stack_depth = 0;
	const Scene** node_stack;
#ifdef __CUDA_ARCH__
	node_stack = static_cast<const Scene**>(malloc(sizeof(Scene*) * recursion_limit));
#else
	cudaMallocManaged(&node_stack, sizeof(Scene*) * recursion_limit);
#endif

	ray.direction = ray.direction * radius;

	auto node = this;
	for (auto bounce_number = 0; bounce_number < 10;) {
		ray_alive = false;
		auto bounced = false;

		for (auto i = 0; i < node->num_spheres; ++i) {
			if (!node->spheres[i].intersects(ray, out_ray)) continue;

			const auto hit_point = out_ray.origin;
			const auto sqr_distance = dot(hit_point - ray.origin, hit_point - ray.origin);
			if (sqr_distance < sqr_distance_to_closest_hit) {
				sqr_distance_to_closest_hit = sqr_distance;
				bounced = true;
				ray_alive = true;
			}
		}

		for (auto i = 0; i < node->num_boxes; ++i) {
			if (!node->boxes[i].intersects(ray, out_ray)) continue;

			const auto hit_point = out_ray.origin;
			const auto sqr_distance = dot(hit_point - ray.origin, hit_point - ray.origin);
			if (sqr_distance < sqr_distance_to_closest_hit) {
				sqr_distance_to_closest_hit = sqr_distance;
				bounced = true;
				ray_alive = true;
			}
		}

		// Need an actual stack because what if a child scene is closer than a box, passes through and hits the box.
		for (auto i = 0; i < node->num_scenes; ++i) {
			if (!node->scenes[i].intersects(ray, out_ray)) continue;

			const auto hit_point = out_ray.origin;
			const auto sqr_distance = dot(hit_point - ray.origin, hit_point - ray.origin);
			if (sqr_distance < sqr_distance_to_closest_hit) {
				sqr_distance_to_closest_hit = sqr_distance;
				bounced = false;
				ray_alive = true;
				node = &node->scenes[i];
				node_stack[stack_depth] = node;
				stack_depth++;
			}
		}

		if (bounced) {
			bounce_number++;
		}

		if (!ray_alive) {
			if (stack_depth == 0) {
				return false;
			} else {
				stack_depth--;
				node = node_stack[stack_depth];
			}
		}
	}

	ray = out_ray;
	return ray_alive;
}


__global__ void destroyRecursively(const Scene& scene) {
	free(scene.spheres);
	free(scene.boxes);
	for (auto i = 0; i < scene.num_scenes; i++) {
		destroyRecursively<<<1, 1>>>(scene.scenes[i]);
	}
	free(scene.scenes);
}