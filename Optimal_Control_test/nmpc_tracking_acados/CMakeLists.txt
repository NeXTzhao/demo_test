cmake_minimum_required(VERSION 3.10)
project(BicycleModelNMPC)

find_package(acados REQUIRED)

add_executable(bicycle_model main.cpp)

# 链接 ACADOS 和其他依赖库
target_link_libraries(bicycle_model PRIVATE acados blasfeo hpipm)


