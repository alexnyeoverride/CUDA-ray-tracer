#include "camera.cuh"
#include "ray.cuh"

// TODO: device capture
__host__ Image Camera::capture() const {
	constexpr auto aspect_ratio = 16.0 / 9.0;
	constexpr auto image_width = 400;
	constexpr auto image_height = static_cast<int16_t>(image_width / aspect_ratio);
	const auto image = Image::create(image_width, image_height);

	constexpr auto viewport_height = 2.0;
	constexpr auto viewport_width = viewport_height * aspect_ratio;
	constexpr auto viewport_u = double4(viewport_width, 0.0, 0.0, 0.0);
	constexpr auto viewport_v = double4(0.0, -viewport_height, 0.0, 0.0);

	const auto pixel_delta_u = viewport_u / image_width;
	const auto pixel_delta_v = viewport_v / image_height;
	const auto upper_left_corner =
		center - double4{0.0, 0.0, focal_length, 0.0} - viewport_u/2 - viewport_v/2;
	const auto upper_left_pixel = upper_left_corner + (pixel_delta_u + pixel_delta_v) * 0.5;

	// TODO: parallelize
	for (auto y = 0; y < image_height; y++) {
		for (auto x = 0; x < image_width; x++) {
			const auto pixel_center = upper_left_pixel + pixel_delta_u * x + pixel_delta_v * y;
			const auto direction = pixel_center - center;
			const auto ray = Ray {.origin=center, .direction=direction};
			image.at(x, y) = ray.color();
		}
	}

	return image;
}