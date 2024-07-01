cmake_minimum_required(VERSION 3.10)
project(QRDecomposition)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include Eigen headers
include_directories("D:/projects/parallel-QR-Decomposition/eigen-3.4.0")

# Add executables
add_executable(generate_matrix src/generate_matrix.cpp)
add_executable(normal_householder src/normal_householder.cpp)
add_executable(parallel_householder src/parallel_householder.cpp)

# Link with PDH library
target_link_libraries(normal_householder pdh.lib)
target_link_libraries(parallel_householder pdh.lib)
# Find OpenMP and link it for parallel Householder
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel_householder OpenMP::OpenMP_CXX)
endif()
