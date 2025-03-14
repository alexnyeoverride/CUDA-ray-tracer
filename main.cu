#include "image.cuh"
#include "camera.cuh"
#include "scene.cuh"

int main() {
	constexpr auto camera = Camera {};
	auto scene = Scene::construct();
	const auto image = camera.capture(/* TODO &scene */);
    image.save();

	scene.destroy();
    image.destroy();

	return 0;
}