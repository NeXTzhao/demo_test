cmake_minimum_required(VERSION 3.21)
project(linesearch VERSION 0.1.0)
set(CMAKE_C_STANDARD 11)

add_executable(line_search
        main.cpp
        linesearch.cpp
        cubicspline.c
)