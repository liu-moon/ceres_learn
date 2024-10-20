# 指定 CMake 的最低版本要求为 3.20.0
cmake_minimum_required(VERSION 3.20.0)

project(ceres CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

add_subdirectory(src bin)