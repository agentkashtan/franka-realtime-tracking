// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <cmath>
#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "examples_common.h"

#include <librealsense2/rs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

/**
 * @example cartesian_impedance_control.cpp
 * An example showing a simple cartesian impedance controller without inertia shaping
 * that renders a spring damper system where the equilibrium is the initial configuration.
 * After starting the controller try to push the robot around and try different stiffness levels.
 *
 * @warning collision thresholds are set to high values. Make sure you have the user stop at hand!
 */

mutex pose_mutex;
bool pose_initialized = false;
float markerLength = 0.06;
cv::Mat cameraMatrix(3,3, CV_32FC1), distCoeffs;
Eigen::Quaterniond orientation_d;
Eigen::Vector3d position_d;
Eigen::Vector3d position_p;
Eigen::Vector3d position_i;
int frame_number = 0;
double translational_stiffness{80.0};
double rotational_stiffness{20.0};
Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
 
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
//cv::aruco::Dictionary dictionary = cv::aruco::extendDictionary(1, 5);
Eigen::Isometry3d pose_offset;
Eigen::Vector3d t_offset(0, 0, 0.3);
Eigen::Isometry3d g_T_t;

void realsense_callback(const rs2::frame& frame) {
    Eigen::Isometry3d g_T_c;
    g_T_c.setIdentity();
    g_T_c.linear() = Eigen::AngleAxisd(-1.57, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    g_T_c.translation() << -0.03, 0., -0.06;
	
    rs2::frameset fs = frame.as<rs2::frameset>();
    if(!fs)return;
    rs2::video_frame cur_frame = fs.get_color_frame();
    if(!cur_frame)return;

    // Convert the RealSense frame to a cv::Mat
    rs2::video_frame video_frame = cur_frame.as<rs2::video_frame>();
    int width = video_frame.get_width();
    int height = video_frame.get_height();
    int channels = video_frame.get_bytes_per_pixel();
    int stride = video_frame.get_stride_in_bytes();

    // The image data is stored in a shared pointer
    const void* frame_data = video_frame.get_data();
    try {
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams(new cv::aruco::DetectorParameters);
    
    cv::Mat image(height, width, CV_8UC(channels), const_cast<void*>(frame_data), stride);
    vector<int> ids;
    vector<vector<cv::Point2f>> corners, rejected;     
    cv::aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
    
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    //cv::Mat outputImage;
    //image.copyTo(outputImage);    
    //cout << "[CB] id " << ids.size() << endl;
    if (ids.size() == 1 && ids.front() == 0) {
    	frame_number = 0;
	cv::Vec3d rvec, tvec;
	vector<int> dist;
	cv::solvePnP(objPoints, corners.at(0), cameraMatrix, dist, rvec, tvec);

	cv::drawFrameAxes(image, cameraMatrix, dist, rvec, tvec, 0.1);	
	
	// construct 4x4 transformation matrix

	// rotation
	cv::Mat rot_mat(3, 3, CV_32FC1); ;
	cv::Rodrigues(rvec, rot_mat);
	Eigen::Matrix3d eigen_matrix;
	cv::cv2eigen(rot_mat, eigen_matrix);
	Eigen::Isometry3d camera_T_tag;
	camera_T_tag.linear() = eigen_matrix;
	Eigen::Vector3d eigen_translation;
	cv::cv2eigen(tvec,eigen_translation);
	camera_T_tag.translation() = eigen_translation;

	// Should work as follows
        if (eigen_translation[0] > -0.3 && eigen_translation[0] < 0.3 && eigen_translation[1] < 0.3 && eigen_translation[1] > -0.3 && eigen_translation[2] > 0.1 && eigen_translation[2] < 0.6) {	
		Eigen::Vector3d offset;
                offset << 0.,0.,0.3;
		g_T_t = g_T_c * camera_T_tag;
	        g_T_t.translation() = g_T_t.translation() - offset; 	
		position_p = g_T_t.translation();
	}
	//cout << "pos in t: " << tvec[0] << " " << tvec[1] << " " << tvec[2] << endl << flush;
	//cout << translational_stiffness << endl << endl;

	//// TODO: run aruco tag detector in opencv
	//// TODO: compute tag pose (using instrinsics)
	//// assign pose to global variable
	//lock_guard<std::mutex> lock(pose_mutex);
	//pose_initialized = true;
    } else {
    	frame_number ++;
    }

    if (frame_number > 10) {    
    	translational_stiffness = 80;
    } else {
    	translational_stiffness = 200;
    }

    cv::imshow("tasde", image);
    cv::waitKey(5);
    } catch (std::exception const & ex ) {
    	std::cout << ex.what() << std::endl;
    }
};	


int main(int argc, char** argv) {
    // connect to robot
    franka::Robot robot(argv[1]);
    //while(true) {
    //	std::cout << "Press enter to read from robot";
    //    Eigen::Isometry3d transform(Eigen::Matrix4d::Map(state.O_T_EE.data()));
    //    std::cin.ignore();
    //    franka::RobotState state = robot.readOnce();
    //    std::cout << "O_T_EE\n" << transform.matrix() << std::endl;
    //}
  // Check whether the required arguments were passed
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " <robot-hostname>" << endl;
    return -1;
  }
  cv::Mat markerImage;
  cv::aruco::drawMarker(dictionary, 0, 200, markerImage, 1);
  cv::imwrite("marker.png", markerImage);

  // Define Transformation from camera to gripper frame
  // g_T_c.linear() = Eigen::AngleAxisd(-1.57, Eigen::Vector3d::UnitY()).toRotationMatrix();
  // g_T_c.translation() << 0., 0., 0.06;
  // g_T_c.translation() << 0.06, 0., 0.;
  
  // create librealsense pipelinei
  
  rs2::pipeline pipe;
  rs2::config cfg;
  const int fps = 6;
  cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, fps);
  cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, fps);
  cfg.enable_device("819612070440");
  pipe.start(cfg,realsense_callback);
  auto intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

   //TODO: convert instrinsics to OpenCV
   float camera_data[3][3] = {{intrinsics.fx, 0, intrinsics.width / 2.f}, {0, intrinsics.fy, intrinsics.height / 2.f}, {0, 0, 1}};
   cameraMatrix = cv::Mat(3, 3, CV_32FC1, &camera_data);
   
  // save to global variable

  // Compliance parameters
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                     Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                         Eigen::MatrixXd::Identity(3, 3);
  Eigen::Matrix<double,6,1> integral;
  integral.setZero();
  double integral_gain = 0.1;
  try {

    setDefaultBehavior(robot);
    // load the kinematics and dynamics model
    franka::Model model = robot.loadModel();

    franka::RobotState initial_state = robot.readOnce();
    // equilibrium point is the initial position
    Eigen::Isometry3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    position_d = Eigen::Vector3d::Zero();

    position_i = Eigen::Vector3d(initial_transform.translation());
    orientation_d = Eigen::Quaterniond(initial_transform.rotation());
    
    // set collision behavior
    robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // define callback for the torque control loop
    function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback = [&](const franka::RobotState& robot_state,
                                         franka::Duration time) -> franka::Torques {
      // get state variables
      array<double, 7> coriolis_array = model.coriolis(robot_state);
      array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

      // convert to Eigen
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
      Eigen::Isometry3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.rotation());

      Eigen::Isometry3d O_T_EE(transform);

      Eigen::Isometry3d O_T_tag = O_T_EE * g_T_t;
      Eigen::Vector3d den(O_T_tag.translation());
      //cout << "tag in  base: "<< den[0] << " " << den[1] << " " << den[2] << endl << endl << flush; 
      position_d = O_T_EE.linear() * g_T_t.translation();
      std::cout << "positioni_d " << position_d.transpose() << std::endl;

      // compute error to desired equilibrium pose
      // position error
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) << position - (position_i + position_d);
      integral.head(3) += error.head(3);
      integral = integral.cwiseMax(-100.).cwiseMin(100.);
      // orientation error
      // "difference" quaternion
      if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
      }
      // "difference" quaternion
      Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
      error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      // Transform to base frame
      error.tail(3) << -transform.rotation() * error.tail(3);

      // compute control
      Eigen::VectorXd tau_task(7), tau_d(7);

      // Spring damper system with damping ratio=1
      tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq) - integral_gain * integral);
      tau_d << tau_task + coriolis;

      array<double, 7> tau_d_array{};
      Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

      //cout << error.tail(3).norm() << endl;
      return tau_d_array;  
    };
    
    // start real-time control loop
    cout << "WARNING: Collision thresholds are set to high values. "
              << "Make sure you have the user stop at hand!" << endl
              << "After starting try to push the robot and see how it reacts." << endl
              << "Press Enter to continue..." << endl;
    cin.ignore();
    robot.control(impedance_control_callback);

  } catch (const franka::Exception& ex) {
    // print exception
    cout << ex.what() << endl;
  }

  return 0;
}
