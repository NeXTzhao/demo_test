cmake_minimum_required(VERSION 3.10)
project(MyProtoProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 找到 Protobuf 包
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# 指定生成的文件输出路径
set(PROTO_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

# 生成 protobuf 源文件
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS example.proto PROTOC_OUT_DIR ${PROTO_SRC_DIR})

# 包含生成的文件目录
include_directories(${PROTO_SRC_DIR})

# 添加可执行文件
add_executable(example example.cpp ${PROTO_SRCS} ${PROTO_HDRS})

# 链接 Protobuf 库
target_link_libraries(example ${Protobuf_LIBRARIES})

add_executable(main main.cpp)

