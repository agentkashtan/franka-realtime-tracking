Real-Time Object Tracking and Following for Robotic Manipulation

This project showcases a robotic system that tracks and follows a moving object in real time, continuously adjusting its movements based on visual feedback. Using an Intel RealSense D405 camera, the system detects and tracks an object's 6D pose (position and orientation) while it moves on a conveyor belt. The robot continuously follows the object, adjusting its trajectory in real time. When conditions are optimal, it attempts to grasp the object.

The core of this system relies on the NVIDIA FoundationPose deep learning model running on a remote server equipped with an NVIDIA GPU. This model processes RGB-D images in real time and provides continuous 6D pose estimations, enabling accurate visual tracking and dynamic movement adjustments.

Features

Real-Time Visual Tracking

The Intel RealSense D405 captures RGB-D images of the moving object.

Images are sent to a remote server that has an NVIDIA GPU for real-time processing.

The server returns the 6D pose (position & orientation) of the object.

The robot continuously updates its trajectory based on the latest pose data.

Trajectory Planning & Adaptive Control

C++ implementation using CasADi for trajectory optimization.

Ensures the object remains in the camera’s field of view while tracking.

Uses real-time feedback for precise alignment.

Torque control is managed via libfranka.

Multi-Stage Operation

Initial Tracking: The robot moves to observe the object.

Dynamic Following: Continuously updates its motion based on real-time pose updates.

Grasp Execution: Once aligned, the robot attempts to grasp the object.

Repository Structure

kalman/                 # Kalman filter implementation
trajectory_generator/   # Trajectory optimization methods
CMakeLists.txt          # CMake build file for the robot controller
pose_estimator_server.py # Python server for running pose estimation
state_controller.cpp    # Main controller code

Dependencies

Ensure the following dependencies are installed:

Robot Controller (C++)

Pinocchio – Kinematics and dynamics library for robotics.

libfranka – Interface for controlling the Franka Emika Panda robot arm.

librealsense – SDK for Intel RealSense depth cameras.

Eigen – Linear algebra library.

Boost (v1.81.0) – C++ library for utility functions.

Poco – C++ libraries for networking and system utilities.

cppzmq – ZeroMQ C++ bindings for communication.

OpenCV – Computer vision and image processing library.

IPOPT – Nonlinear optimization library.

CasADi – Symbolic framework for optimization.

Install via Python: pip3 install casadi

CasADi C++ build guide

System Dependencies:

libopenexr-dev

build-essential

coinor-libipopt-dev

📌 All dependencies can be checked in CMakeLists.txt.

Pose Estimation Server (Python)

pyzmq – For ZeroMQ communication.

opencv-python – Image processing.

NVIDIA FoundationPose – 6D object pose estimation model.

Install via NVIDIA’s guide: FoundationPose Repository

Running the System

1. Starting the Pose Estimation Server

To run the server, you need to follow NVIDIA’s guide to set up FoundationPose. Once installed:

Place pose_estimator_server.py into the root folder of the FoundationPose repository.

Start the server:

python pose_estimator_server.py

This will load the model and listen for incoming pose estimation requests from the robot.

2. Running the Robot Controller

Build the controller:

mkdir build && cd build
cmake .. && make

Run the controller, specifying the robot's hostname:

./state_controller <robot_host>

Update IP Addresses:

Ensure the robot is communicating with the correct server IP for pose estimation.

Contact

For questions or contributions, feel free to reach out:
📧 ozamozhs@uwaterloo.ca

For more details, visit the GitHub Project Page.
