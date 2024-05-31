cmake_minimum_required(VERSION 3.27)
project(fallback)

set(CMAKE_CXX_STANDARD 17)
find_package(Python3 3.8 COMPONENTS Interpreter Development NumPy REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        Python3::Python
        Python3::NumPy
)

target_include_directories(${PROJECT_NAME} PUBLIC /usr/include/python3.8)
