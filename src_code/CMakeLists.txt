cmake_minimum_required(VERSION 3.10)
project(TestPinocchioEigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pinocchio REQUIRED)
find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(test_pinocchio_eigen pin_test.cpp)


target_link_libraries(test_pinocchio_eigen PRIVATE pinocchio::pinocchio Eigen3::Eigen Franka::Franka)
