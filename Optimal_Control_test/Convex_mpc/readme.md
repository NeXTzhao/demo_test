# 在build文件中通过conan安装必要的依赖库
conan install . --build=missing
# 然后再cmake .. && make
