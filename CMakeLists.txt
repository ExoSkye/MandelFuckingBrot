cmake_minimum_required(VERSION 3.22)
project(MandelFuckingBrot CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MAXITER_HIGH 6000 CACHE STRING "Max iterations (HQ)")
set(MAXITER_LOW 1000 CACHE STRING "Max iterations")

add_custom_command(
        OUTPUT gradient.hpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generatePalette.py ${MAXITER_HIGH} ${MAXITER_LOW}
        COMMENT Generating palette
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generatePalette.py ${CMAKE_CURRENT_SOURCE_DIR}/palette.py
)

add_executable(MandelFuckingBrot main.cu gradient.hpp)

target_compile_definitions(MandelFuckingBrot PUBLIC MAX_ITER_HIGH=${MAXITER_HIGH} MAX_ITER_LOW=${MAXITER_LOW})

target_include_directories(MandelFuckingBrot PRIVATE ${CMAKE_BINARY_DIR})

set_target_properties(MandelFuckingBrot PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
