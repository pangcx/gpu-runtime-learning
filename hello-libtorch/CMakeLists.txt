cmake_minimum_required(VERSION 3.18)
project(hello_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 从已安装的 PyTorch 包中找 LibTorch
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

add_executable(hello_libtorch main.cpp)
target_link_libraries(hello_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET hello_libtorch PROPERTY CXX_STANDARD 17)

message(STATUS "Torch found at: ${TORCH_CMAKE_PREFIX}")
message(STATUS "Torch version: ${Torch_VERSION}")
