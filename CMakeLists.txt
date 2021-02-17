cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ImageViewer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(
    ImageViewer
    Box.hpp
    Box.cpp
    compare.hpp
    compare.cpp
    Data.hpp
    Data.cpp
    GaussianPyramid.hpp
    GaussianPyramid.cu
    ImageReader.hpp
    ImageReader.cpp
    ImageWriter.hpp
    ImageWriter.cpp
    memory.h
    main.cu
)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)
find_package(Threads REQUIRED)

target_link_libraries(ImageViewer Threads::Threads sfml-graphics sfml-window sfml-system)
