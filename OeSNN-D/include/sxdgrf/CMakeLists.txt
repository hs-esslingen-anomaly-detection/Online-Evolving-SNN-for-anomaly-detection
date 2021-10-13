cmake_minimum_required(VERSION 3.16)
project(sxdgrf_lib_cpp)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(${PROJECT_SOURCE_DIR}/include/pybind11)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(sxdgrf_lib_cpp SHARED kmeans/kmeans.cpp kmeans/kmeans.h grf.cpp grf.h)

add_executable(tests unit_tests.cpp)
target_link_libraries(tests sxdgrf_lib_cpp)

pybind11_add_module(grf_module python_module/pymodule.cpp)
target_link_libraries(grf_module PRIVATE sxdgrf_lib_cpp)

#copy build module to /python_module directory
add_custom_command(
        TARGET grf_module POST_BUILD COMMAND
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/grf_module.cpython-37m-x86_64-linux-gnu.so
        ${PROJECT_SOURCE_DIR}/python_module/grf_module.cpython-37m-x86_64-linux-gnu.so)
