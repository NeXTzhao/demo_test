cmake_minimum_required(VERSION 3.10)

# 项目名称
project(SparseVsDenseQP)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED PLANNING_DEBUG)
    set(PLANNING_DEBUG 1)
endif ()
add_definitions(-DPLANNING_DEBUG=${PLANNING_DEBUG})

#include_directories(third_party)
list(APPEND CMAKE_PREFIX_PATH
        "third_party/proxsuite"
        "third_party/eigen"
        "third_party/fmt"
)


find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(proxsuite REQUIRED)

add_executable(convex_mpc convex_mpc.cpp)
target_link_libraries(convex_mpc
        proxsuite::proxsuite
        Python3::Python
        Python3::NumPy
        Eigen3::Eigen
        fmt::fmt
        spdlog::spdlog)
