cmake_minimum_required(VERSION 3.10)
project(test)
set(CMAKEEf_CXX_STANDARD 11)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(inverse_test)
add_executable(inverse_test inverse.cpp)
target_link_libraries(inverse_test
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen)
target_include_directories(inverse_test PUBLIC /usr/include/python3.8)

add_executable(correction_matrix correction_matrix.cpp)
target_link_libraries(correction_matrix
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen)
target_include_directories(correction_matrix PUBLIC /usr/include/python3.8)