cmake_minimum_required(VERSION 3.11)

project(ConvexMPC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED PLANNING_DEBUG)
    set(PLANNING_DEBUG 1)
endif ()
add_definitions(-DPLANNING_DEBUG=${PLANNING_DEBUG})

list(APPEND CMAKE_PREFIX_PATH
        "third_party/proxsuit_0_6_6"
)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(proxsuite REQUIRED)

add_executable(convex_mpc convex_mpc.cpp)
target_link_libraries(convex_mpc
        proxsuite::proxsuite
        ${CONAN_LIBS}
        Python3::Python
        Python3::NumPy
)