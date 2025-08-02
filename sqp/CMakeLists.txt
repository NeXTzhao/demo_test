cmake_minimum_required(VERSION 3.16)
project(mim_solvers VERSION 0.1.0 LANGUAGES CXX)

# 统一 C++ 标准（推荐用 compile_features）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(BUILD_DEMO "Build demo executable" ON)

# 源文件（把缺失的几个子目录也加上）
file(GLOB_RECURSE MIM_SOLVERS_SOURCES
        src/ocp/constraint/*.cc
        src/ocp/cost/*.cc
        src/ocp/csqp/*.cc
        src/ocp/dynamics/*.cc
        src/ocp/shooting/*.cc
        src/ocp/solver/*.cc
        src/ocp/utils/*.
        src/*.cc
)

add_library(mim_solvers SHARED ${MIM_SOLVERS_SOURCES})
add_library(mim_solvers::mim_solvers ALIAS mim_solvers)

target_compile_features(mim_solvers PUBLIC cxx_std_17)

target_include_directories(mim_solvers
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mim_solvers
        PUBLIC Eigen3::Eigen
)

# 可选：打开常见警告（按需）
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mim_solvers PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Demo
if (BUILD_DEMO)
    add_executable(mim_solvers_demo main.cc)
    target_link_libraries(mim_solvers_demo PRIVATE mim_solvers)
    target_compile_features(mim_solvers_demo PRIVATE cxx_std_17)
endif()

# 安装
install(TARGETS mim_solvers
        EXPORT mim_solversTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
