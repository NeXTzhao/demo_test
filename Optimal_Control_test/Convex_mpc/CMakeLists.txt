cmake_minimum_required(VERSION 3.11)

project(ConvexMPC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED PLANNING_DEBUG)
    set(PLANNING_DEBUG 2)
endif ()
add_definitions(-DPLANNING_DEBUG=${PLANNING_DEBUG})

list(APPEND CMAKE_PREFIX_PATH
        "third_party/nlohmann_3_11_2"
        "third_party/spdlog_1_12_0"
        "third_party/Eigen_3_4_0"
        "third_party/proxsuite"
)
find_package(Eigen3 3.4.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(Python3 3.8 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(nlohmann_json 3.11.2 REQUIRED)
find_package(spdlog 1.12.0 REQUIRED)
find_package(proxsuite REQUIRED)

add_executable(convex_mpc convex_mpc.cpp
        bp_path_tracking/longitudinal_model.hpp
        bp_path_tracking/lateral_model.hpp)
target_link_libraries(convex_mpc
        proxsuite::proxsuite
        Python3::Python
Python3::NumPy
        nlohmann_json::nlohmann_json
)

add_executable(convex_mpc_spares convex_mpc_spares.cpp)
target_link_libraries(convex_mpc_spares
        proxsuite::proxsuite
        Python3::Python
        Python3::NumPy
        nlohmann_json::nlohmann_json
)

file(GLOB TINYMPC_SOURCES "third_party/tinympc/*.cpp")
add_executable(tinympc_spares tinympc_spares.cpp ${TINYMPC_SOURCES})
target_link_libraries(tinympc_spares
        proxsuite::proxsuite
        Python3::Python
        Python3::NumPy
        nlohmann_json::nlohmann_json
)

add_executable(tinympc_bicycle  tinympc_bicycle_spares.cpp ${TINYMPC_SOURCES})
target_link_libraries(tinympc_bicycle
        Python3::Python
        Python3::NumPy
        nlohmann_json::nlohmann_json
)

add_executable(tinympc_spares_lat_lon  tinympc_spares_lat_lon.cpp ${TINYMPC_SOURCES})
target_link_libraries(tinympc_spares_lat_lon
        Python3::Python
        Python3::NumPy
        nlohmann_json::nlohmann_json
)