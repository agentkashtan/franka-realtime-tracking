# Real-Time Tracking & Grasping of Moving Objects with a Franka Panda

## Overview
This project showcases a **real-time object tracking and grasping system** for a **Franka Emika Panda** robot, designed to handle objects moving on a conveyor belt. An **Intel RealSense D405** camera captures **RGB-D** images, which are streamed via **ZeroMQ** to a **remote serve**r with an **NVIDIA RTX 4090 GPU**, where a pre-trained **NVIDIA FoundationPose** model performs **6D pose** estimation in real time. The robot controller, implemented in **C++**, uses **CasADi** for trajectory optimization and applies both **joint-level and Cartesian impedance control** via **libfranka**, with visual feedback continuously guiding its motion in response to the object's movement.

## Demo

https://github.com/user-attachments/assets/0e561416-2eff-4122-96ff-2f49c597eb31
https://github.com/user-attachments/assets/84b7e4d4-6665-42a5-86f0-c73ba519fadc







## Features
- **Real-Time Visual Tracking**  
  - The **Intel RealSense D405** captures **RGB-D** images of the moving object.  
  - Images are sent to a **remote server that has an NVIDIA GPU** for real-time processing.  
  - The server returns the **6D pose** (position & orientation) of the object.  
  - The robot continuously updates its trajectory based on the latest pose data.

- **Trajectory Planning & Adaptive Control**  
  - **C++ implementation** using **CasADi** for trajectory optimization.  
  - Ensures the object remains in the camera’s **field of view** while tracking.  
  - Uses **real-time feedback** for precise alignment.  
  - **Torque control** is managed via **libfranka**.

- **Multi-Stage Operation**
   1. **Initial Tracking**: The robot moves to a region with a clear view of the conveyor belt to begin monitoring for objects.
   2. **Trajectory Planning & Approach**: Once an object is detected, the robot plans and executes a trajectory to approach it, ensuring the object remains in the camera’s view throughout the motion.
   3. **Dynamic Following & Grasp Execution**: As the robot nears the object, it switches to visual servoing, continuously adjusting its motion based on real-time pose updates to follow and grasp the moving object accurately.

## Repository Structure
- **state_controller.cpp**  
  Main controller code
- **pose_estimator_server.py**  
  Python server for running pose estimation  
- **kalman/**  
  Kalman filter implementation  
- **trajectory_generator/**  
  Trajectory optimization methods  
- **CMakeLists.txt**  
  CMake build file for the robot controller  

## Dependencies

### Robot Controller (C++)
- **Pinocchio** – Kinematics and dynamics library for robotics.  
- **libfranka** – Interface for controlling the Franka Emika Panda robot arm.  
- **librealsense** – SDK for Intel RealSense depth cameras.  
- **Eigen** – Linear algebra library.  
- **Boost (v1.81.0)** – C++ library for utility functions.  
- **Poco** – C++ libraries for networking and system utilities.  
- **cppzmq** – ZeroMQ C++ bindings for communication.  
- **OpenCV** – Computer vision and image processing library.  
- **IPOPT** – Nonlinear optimization library.  
- **CasADi** – Symbolic framework for optimization.  
  - Install via Python: `pip3 install casadi`  
  - [CasADi C++ build guide](https://github.com/zehuilu/Tutorial-on-CasADi-with-CPP)  
- **System Dependencies**:  
  - `libopenexr-dev`  
  - `build-essential`  
  - `coinor-libipopt-dev`  

> **Note:** All dependencies can be checked in `CMakeLists.txt`.

### Pose Estimation Server (Python)
- `pyzmq` – For ZeroMQ communication.  
- `opencv-python` – Image processing.  
- **NVIDIA FoundationPose** – 6D object pose estimation model.  
  - Install via NVIDIA’s guide: [FoundationPose Repository](https://github.com/NVlabs/FoundationPose)

---

## Running the System

### 1. Starting the Pose Estimation Server
Follow NVIDIA’s guide to set up **FoundationPose**. Once installed:
1. Place `pose_estimator_server.py` into the **root folder** of the FoundationPose repository.  
2. Start the server:
   ```bash
   FIVEG_ENABLED=false python pose_estimator_server.py --debug 0

### 2. Running the Robot Controlle
1. Update IP Addresses:
  Ensure the robot is communicating with the correct server IP for pose estimation.
2. Build the controller:
   ```bash
   mkdir build && cd build
   cmake ..
   make
3. Run the controller, specifying the robot's hostname(exp_hash, exp_num, exp_dir are just for logging purposes)
   ```bash
   mode=5G exp_hash="#1" exp_num=0 exp_dir="default" ./state_controller franka2 true

   
