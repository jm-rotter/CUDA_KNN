cmake_minimum_required(VERSION 3.18)
project(cuda_knn LANGUAGES CXX CUDA)

include_directories(${CMAKE_SOURCE_DIR}/includes)

file(GLOB_RECURSE SRC_FILES
	"${CMAKE_SOURCE_DIR}/main.cu"
	"${CMAKE_SOURCE_DIR}/distance_kernels/*.cu"
	"${CMAKE_SOURCE_DIR}/distance_kernels/*.cu"
	"${CMAKE_SOURCE_DIR}/evaluate/*.cu"
)

add_executable(CUDA_KNN ${SRC_FILES})

target_compile_definitions(CUDA_KNN PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set_target_properties(CUDA_KNN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
