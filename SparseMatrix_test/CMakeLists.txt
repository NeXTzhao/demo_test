cmake_minimum_required(VERSION 3.10)

# 项目名称
project(SparseVsDenseQP)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

add_executable(SparseVsDenseQP sparse_matrix.cpp)

target_link_libraries(SparseVsDenseQP Eigen3::Eigen)
