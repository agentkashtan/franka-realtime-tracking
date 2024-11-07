cmake_minimum_required(VERSION 3.10)
project(TestPinocchioEigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



find_package(pinocchio REQUIRED)
find_package(OpenCV REQUIRED aruco)
find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realsense2 REQUIRED)


add_executable(state_controller state_controller.cpp kalman/kalman.cpp) 
target_include_directories(state_controller PRIVATE ${CMAKE_SOURCE_DIR}/kalman)


target_link_libraries(state_controller PRIVATE ${OpenCV_LIBS} opencv_highgui pinocchio::pinocchio Eigen3::Eigen Franka::Franka realsense2::realsense2)
 
