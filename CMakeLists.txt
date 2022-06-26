cmake_minimum_required(VERSION 3.22)

if (CUDA)
    project(MandelFuckingBrot CUDA CXX C)
    set(CMAKE_CUDA_STANDARD 20)
else()
    project(MandelFuckingBrot HIP CXX C)
    set(CMAKE_HIP_STANDARD 20)
    find_package(hip REQUIRED)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MAXITER_HIGH 6000 CACHE STRING "Max iterations (HQ)")
set(MAXITER_LOW 1000 CACHE STRING "Max iterations")

add_custom_command(
        OUTPUT gradient.hpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generatePalette.py ${MAXITER_HIGH} ${MAXITER_LOW}
        COMMENT Generating palette
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generatePalette.py ${CMAKE_CURRENT_SOURCE_DIR}/palette.py
)

add_executable(MandelFuckingBrot gradient.hpp)

if(CUDA)
    target_sources(MandelFuckingBrot PRIVATE main.cu)
    target_include_directories(MandelFuckingBrot PRIVATE ${ROCM_PATH}/include/hip/nvidia_detail/)
    target_compile_definitions(MandelFuckingBrot PRIVATE CUDA_BUILD)
else()
    target_sources(MandelFuckingBrot PRIVATE main.cpp)
    target_link_libraries(MandelFuckingBrot PRIVATE hip::host hip::device)
endif()

target_compile_definitions(MandelFuckingBrot PUBLIC MAX_ITER_HIGH=${MAXITER_HIGH} MAX_ITER_LOW=${MAXITER_LOW})

target_include_directories(MandelFuckingBrot PRIVATE ${CMAKE_BINARY_DIR})