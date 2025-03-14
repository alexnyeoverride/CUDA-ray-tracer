#include "camera.cuh"

#include <iostream>
#include <limits>
#include <ostream>

#include "ray.cuh"
#include "scene.cuh"

__global__ void cast(
	const Scene& scene,
	const Camera& camera,
	Image& image,
	const double4 upper_left_pixel,
	const double4 pixel_delta_u,
	const double4 pixel_delta_v
) {
	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= image.width || y >= image.height)  return;

	const auto pixel_center = upper_left_pixel + pixel_delta_u * x + pixel_delta_v * y;
	const auto direction = pixel_center - camera.center;
	auto ray = Ray {.origin=camera.center, .direction=direction};

	auto sqr_distance_to_closest_hit = std::numeric_limits<double>::infinity();
	auto out_ray = ray;
	auto ray_alive = true;

	auto stack_depth = 0;
	const auto node_stack = static_cast<const Scene**>(malloc(sizeof(Scene*) * recursion_limit));

	ray.direction = ray.direction * scene.radius;

	auto node = &scene;
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
			// The ray hit the sky.
			if (stack_depth == 0) {
				free(node_stack);
				return;
			// The ray left a child BVH.
			} else {
				stack_depth--;
				node = node_stack[stack_depth];
			}
		}
	}

	free(node_stack);

	image.at(x, y) = {
		static_cast<unsigned char>(ray.color.x * 255.999),
		static_cast<unsigned char>(ray.color.y * 255.999),
		static_cast<unsigned char>(ray.color.z * 255.999),
		static_cast<unsigned char>(ray.color.w * 255.999),
	};
}

__host__ Image Camera::capture(const Scene& scene) const {
	constexpr auto aspect_ratio = 16.0 / 9.0;
	constexpr auto image_width = 400;
	constexpr auto image_height = static_cast<int16_t>(image_width / aspect_ratio);
	auto image = Image::create(image_width, image_height);

	constexpr auto viewport_height = 2.0;
	constexpr auto viewport_width = viewport_height * aspect_ratio;
	constexpr auto viewport_u = double4(viewport_width, 0.0, 0.0, 0.0);
	constexpr auto viewport_v = double4(0.0, -viewport_height, 0.0, 0.0);

	const auto pixel_delta_u = viewport_u / image_width;
	const auto pixel_delta_v = viewport_v / image_height;
	const auto upper_left_corner =
		center - double4{0.0, 0.0, focal_length, 0.0} - viewport_u/2 - viewport_v/2;
	const auto upper_left_pixel = upper_left_corner + (pixel_delta_u + pixel_delta_v) * 0.5;

	const auto gridDim = dim3(image_width / 32 + 1, image_height / 32 + 1);
	cast<<<gridDim, {32, 32}>>>(scene, *this, image, upper_left_pixel, pixel_delta_u, pixel_delta_v);
	cudaDeviceSynchronize();

	const auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
	}

	return image;
}