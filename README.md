# Real-Time Tracking & Grasping of Moving Objects with a Franka Panda

## Overview
This project showcases a **visual-servoing** pick-and-place system designed to handle objects moving on a conveyor belt using a **Franka-Emika Panda robot arm**. The main objective is to accurately locate and grasp these moving objects by continuously determining their **6D pose** (position and orientation). To accomplish this, an **Intel RealSense D405** camera captures **RGB-D** images, which are transmitted via **ZeroMQ** to a remote server equipped with an **NVIDIA RTX 4090 GPU**. On this server, a pre-trained **NVIDIA Foundation Pose** model processes the incoming data in real time and sends back the object’s 6D pose estimates. 

## Demo

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
  - **Initial Approach**: Move from a safe, elevated position to a location where the object is within clear view.
  -  **Refined Positioning**: Update the end-effector pose continuously using new 6D estimates, compensating for the object’s movement on the belt.
  -  **Grasp and Place**: Secure the object and move it to a designated drop-off location, completing the pick-and-place cycle. 

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
   python pose_estimator_server.py

### 2. Running the Robot Controlle
1. Update IP Addresses:
  Ensure the robot is communicating with the correct server IP for pose estimation.
2. Build the controller:
   ```bash
   mkdir build && cd build
   cmake ..
   make
3. Run the controller, specifying the robot's hostname
   ```bash
   ./state_controller <robot_host>
