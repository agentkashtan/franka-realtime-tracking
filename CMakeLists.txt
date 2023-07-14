cmake_minimum_required(VERSION 3.0)
project(sixdobjtracking VERSION 0.0.1)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Franka REQUIRED)
find_package(realsense2 REQUIRED)


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++17 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

include_directories(
  SYSTEM
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(move_test src/move_test.cpp src/examples_common.cpp)
target_link_libraries(move_test ${OpenCV_LIBS} Franka::Franka realsense2)

