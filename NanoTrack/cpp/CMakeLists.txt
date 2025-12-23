cmake_minimum_required(VERSION 3.10)
project(nanotrack_cpp_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项：开启 CUDA Execution Provider（需使用带 CUDA 的 onnxruntime）
option(USE_CUDA_EP "Enable ONNXRuntime CUDA Execution Provider" OFF)

find_package(OpenCV REQUIRED)

# 优先读取 CMake 变量 ONNXRUNTIME_DIR，其次读取环境变量
if(NOT ONNXRUNTIME_DIR AND DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif()

if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "请通过 -DONNXRUNTIME_DIR=/path/to/onnxruntime 或环境变量 ONNXRUNTIME_DIR 指定 onnxruntime 目录")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "未找到 ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h，请检查 ONNXRUNTIME_DIR")
endif()

if(EXISTS "${ONNXRUNTIME_DIR}/lib")
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)
elseif(EXISTS "${ONNXRUNTIME_DIR}/lib64")
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib64)
else()
    message(FATAL_ERROR "未找到 ${ONNXRUNTIME_DIR}/lib 或 lib64，请检查 ONNXRUNTIME_DIR")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_LIB_DIR})

add_executable(nanotrack_demo nanotrack_demo.cpp nanotrack.cpp)
target_link_libraries(nanotrack_demo PRIVATE ${OpenCV_LIBS} onnxruntime)

if(USE_CUDA_EP)
    target_compile_definitions(nanotrack_demo PRIVATE USE_CUDA_EP)
endif()
