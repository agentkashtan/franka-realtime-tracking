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
#include <opencv2/imgproc.hpp>
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
		g_T_t = camera_T_tag;


	    } else {
	    	frame_number ++;
	    }

	    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	    cv::imshow("tasde", image);
	    cv::waitKey(5);
    } catch (std::exception const & ex ) {
    	std::cout << ex.what() << std::endl;
    }
};	


int main(int argc, char** argv) {


  cv::Mat markerImage;
  cv::aruco::drawMarker(dictionary, 0, 200, markerImage, 1);
  cv::imwrite("marker.png", markerImage);


  rs2::pipeline pipe;
  rs2::config cfg;
  const int fps = 6;
  cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, fps);
  cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, fps);
  cfg.enable_device("825312072012");
  pipe.start(cfg,realsense_callback);
  auto intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

   //TODO: convert instrinsics to OpenCV
   float camera_data[3][3] = {{intrinsics.fx, 0, intrinsics.width / 2.f}, {0, intrinsics.fy, intrinsics.height / 2.f}, {0, 0, 1}};
   cameraMatrix = cv::Mat(3, 3, CV_32FC1, &camera_data);
   while (true){
   	cout << "main " << endl << g_T_t.translation()<< endl;
   }

  return 0;
}
