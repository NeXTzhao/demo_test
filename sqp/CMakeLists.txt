cmake_minimum_required(VERSION 3.10)
project(mim_solvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE SOURCES
        src/ocp/constraint/*.cc
        src/ocp/cost/*.cc
        src/ocp/csqp/*.cc
        src/ocp/dynamics/*.cc
)

file(GLOB_RECURSE HEADERS
        include/ocp/constraint/*.h
        include/ocp/cost/*.h
        include/ocp/csqp/*.h
        include/ocp/dynamics/*.h
)

add_library(mim_solvers SHARED ${SOURCES} ${HEADERS})

target_include_directories(mim_solvers
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mim_solvers
        PUBLIC Eigen3::Eigen
)

add_executable(mim_solvers_demo main.cc)
target_link_libraries(mim_solvers_demo PRIVATE mim_solvers)

install(TARGETS mim_solvers DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
