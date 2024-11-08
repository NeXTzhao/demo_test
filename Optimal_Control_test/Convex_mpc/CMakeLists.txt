cmake_minimum_required(VERSION 3.11)

project(ConvexMPC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED PLANNING_DEBUG)
    set(PLANNING_DEBUG 2)
endif ()
add_definitions(-DPLANNING_DEBUG=${PLANNING_DEBUG})

list(APPEND CMAKE_PREFIX_PATH
        "third_party/proxsuite"
)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include_directories(third_party/tinympc)
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

add_executable(convex_mpc_spares convex_mpc_spares.cpp)
target_link_libraries(convex_mpc_spares
        proxsuite::proxsuite
        ${CONAN_LIBS}
        Python3::Python
        Python3::NumPy
)

add_library(tinympcstatic STATIC
        third_party/tinympc/admm.cpp
        third_party/tinympc/tiny_api.cpp
        third_party/tinympc/codegen.cpp
)


add_executable(tinympc_spares tinympc_spares.cpp)
target_link_libraries(tinympc_spares
        proxsuite::proxsuite
        ${CONAN_LIBS}
        Python3::Python
        Python3::NumPy
        tinympcstatic
)