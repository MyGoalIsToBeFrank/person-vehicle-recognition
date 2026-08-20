cmake_minimum_required(VERSION 3.24)
project(pvr_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit 12.6 REQUIRED)
find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PNG REQUIRED)
find_library(WEBP_LIBRARY webp REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)

pybind11_add_module(pvr_native MODULE
    src/bindings.cpp
    src/engine.cpp
    src/pipeline.cpp
    src/tensorrt_model.cpp
    src/image_decoder.cpp
    src/kernels.cu
)
target_include_directories(pvr_native PRIVATE include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(pvr_native PRIVATE
    CUDA::cudart
    CUDA::nvjpeg
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    OpenSSL::Crypto
    PNG::PNG
    ${WEBP_LIBRARY}
)
target_compile_options(pvr_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror=return-type>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--threads=0>
)
set_target_properties(pvr_native PROPERTIES
    CUDA_ARCHITECTURES "86;80"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS pvr_native LIBRARY DESTINATION .)
