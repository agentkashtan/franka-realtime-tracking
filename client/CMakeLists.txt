# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(my_project LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find cppzmq wrapper, installed by the make process of cppzmq
find_package(cppzmq REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add your program's executable
add_executable(client main.cpp)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

# Link libraries to your executable
target_link_libraries(client cppzmq ${OpenCV_LIBS})

