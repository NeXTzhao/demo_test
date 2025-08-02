cmake_minimum_required(VERSION 3.10)
project(casadi_codegen_nlp)

set(CMAKE_CXX_STANDARD 14)

# 自动包含所有 cpp 文件
file(GLOB SOURCES "*.cpp" "*.c")

add_executable(main ${SOURCES})

# 如果你使用 CasADi 头文件生成的 .hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
