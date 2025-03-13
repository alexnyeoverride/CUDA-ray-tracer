#include "image.cuh"
#include "camera.cuh"

int main() {
	constexpr auto camera = Camera {};
	const auto image = camera.capture();
    image.save();
    image.destroy();

	return 0;
}