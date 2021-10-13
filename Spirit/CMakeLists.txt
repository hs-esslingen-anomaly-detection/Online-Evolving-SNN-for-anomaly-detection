cmake_minimum_required(VERSION 3.16)
project(spirit)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(${PROJECT_SOURCE_DIR}/include/pybind11)
include_directories(${PROJECT_SOURCE_DIR}/include)

# build spirit library
add_library(
        spirit SHARED
        spirit.cpp
        spirit.h)

# build executable (main)
add_executable(main main.cpp)
target_link_libraries(main spirit)

# export python module
pybind11_add_module(spirit_module python_module/python_module.cpp)
target_link_libraries(spirit_module PRIVATE spirit)
