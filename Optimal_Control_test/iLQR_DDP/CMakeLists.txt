cmake_minimum_required(VERSION 3.11)

project(ilqr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED PLANNING_DEBUG)
    set(PLANNING_DEBUG 2)
endif ()
add_definitions(-DPLANNING_DEBUG=${PLANNING_DEBUG})

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_executable(ilqr main.cpp)
target_link_libraries(ilqr
        ${CONAN_LIBS}
        Python3::Python
        Python3::NumPy
)