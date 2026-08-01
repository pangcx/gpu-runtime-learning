cmake_minimum_required(VERSION 3.18)
project(hello_gpu CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# RTX 4090 的架构是 Ada Lovelace = compute capability 8.9
# 设置 GPU 架构，让编译器生成最优代码
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(hello_gpu main.cu)
