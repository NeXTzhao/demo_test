cmake_minimum_required(VERSION 3.10)
project(matrix_inverse)
set(CMAKEEf_CXX_STANDARD 11)
find_package(Eigen3 REQUIRED)
find_package(Python3 3.10 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(inverse_test)
add_executable(${PROJECT_NAME} inverse.cpp)
target_link_libraries(${PROJECT_NAME}
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} PUBLIC /usr/include/python3.10)