cmake_minimum_required(VERSION 3.18)
project(concurrency)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(concurrency_demo main.cpp)

# macOS 需要链接 pthread
find_package(Threads REQUIRED)
target_link_libraries(concurrency_demo Threads::Threads)
