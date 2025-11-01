cmake_minimum_required(VERSION 3.30)
project(logistic_map)
set(CMAKE_CXX_STANDARD 23)
SET(CMAKE_CXX_FLAGS "-O3 -march=native -std=c++23 -fopenmp")
add_executable(logistic_map main.cpp
        fpng.cpp)