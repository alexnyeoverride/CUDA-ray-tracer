cmake_minimum_required(VERSION 3.30)
project(FXAA CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(FXAA main.cu image.cu camera.cu)

set_target_properties(FXAA PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
