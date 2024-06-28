cmake_minimum_required(VERSION 3.10)

# 项目名称
project(SparseVsDenseQP)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(sparse_matrix sparse_matrix.cpp)
target_link_libraries(sparse_matrix
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen)

add_executable(solveDARE solveDARE.cpp)
target_link_libraries(solveDARE
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen)