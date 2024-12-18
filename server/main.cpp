#include <zmq.hpp>
#include <iostream>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>


using namespace std;


cv::Mat cameraMatrix(3,3, CV_32FC1);
     

cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
float markerLength = 0.07;

void flush_queue(zmq::socket_t& socket) {
    //auto start = std::chrono::high_resolution_clock::now();

    while (true) {
    	zmq::message_t msg;
	if (!socket.recv(msg, zmq::recv_flags::dontwait)) break;
	std::string received_string(static_cast<char*>(msg.data()), msg.size());
	//auto end = std::chrono::high_resolution_clock::now();
        //chrono::duration<double> duration = end - start;
        //cout <<received_string << " "  <<  duration.count() << endl;
	cout << "discarding" << endl;	
    }
}


pair<bool, vector<double>> detect_pose(cv::Mat& image) {
    cv::aruco::DetectorParameters d_params = cv::aruco::DetectorParameters();
    vector<int> ids;
    vector<vector<cv::Point2f>> corners, rejected;

    cv::aruco::ArucoDetector detector(dictionary, d_params);
    detector.detectMarkers(image, corners, ids, rejected);

    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    Eigen::Isometry3d camera_T_tag;
    bool detected = false;
    vector<double> buffer(6);

    if (ids.size() == 1 && ids.front() == 0) {
        cv::Vec3d rvec, tvec;
        vector<int> dist;
        cv::solvePnP(objPoints, corners.at(0), cameraMatrix, dist, rvec, tvec);

	memcpy(buffer.data(), &rvec, sizeof(rvec));
        memcpy(buffer.data() + 3, &tvec, sizeof(tvec));
	detected = true;
        /*
        cv::drawFrameAxes(image, cameraMatrix, dist, rvec, tvec, 0.1);

        cv::Mat rot_mat(3, 3, CV_32FC1); ;
        cv::Rodrigues(rvec, rot_mat);
        Eigen::Matrix3d eigen_matrix;
        cv::cv2eigen(rot_mat, eigen_matrix);
        camera_T_tag.linear() = eigen_matrix;
        Eigen::Vector3d eigen_translation;
        cv::cv2eigen(tvec,eigen_translation);
        camera_T_tag.translation() = eigen_translation;
	cout << eigen_translation.transpose() << endl;
        */
    }
    //cv::imshow("gg", image);
    //cv::waitKey(5);
    return { detected, buffer };

}




int main()
{
    cameraMatrix.at<float>(0, 0) = 1000;  
    cameraMatrix.at<float>(0, 1) = 0;     
    cameraMatrix.at<float>(0, 2) = 640;   

    cameraMatrix.at<float>(1, 0) = 0;     
    cameraMatrix.at<float>(1, 1) = 1000;  
    cameraMatrix.at<float>(1, 2) = 360;   

    cameraMatrix.at<float>(2, 0) = 0;
    cameraMatrix.at<float>(2, 1) = 0;
    cameraMatrix.at<float>(2, 2) = 1;

	
    zmq::context_t ctx(2);
    zmq::socket_t socket_in(ctx, zmq::socket_type::pull);
    socket_in.bind("tcp://*:5555");

    zmq::socket_t socket_out(ctx, zmq::socket_type::push);
    socket_out.bind("tcp://*:5554");
    //auto start = std::chrono::high_resolution_clock::now();

    while(true){
    	zmq::message_t msg;
	zmq::message_t msg_timestamp;

	socket_in.recv(msg, zmq::recv_flags::none);    
	socket_in.recv(msg_timestamp, zmq::recv_flags::dontwait);

        vector<uchar> compressed_frame(
    		static_cast<uchar*>(msg.data()), 
    		static_cast<uchar*>(msg.data()) + msg.size()
	);	
	cv::Mat frame = cv::imdecode(compressed_frame, cv::IMREAD_COLOR);
	auto [detected, pose] = detect_pose(frame);
	if (detected) {
	    zmq::message_t msg_out(6 * sizeof(double));
	    memcpy(msg_out.data(), pose.data(), 6 * sizeof(double));
	    socket_out.send(msg_out, zmq::send_flags::sndmore);
	    socket_out.send(msg_timestamp, zmq::send_flags::none);
	}

	flush_queue(socket_in);
    
    }

    return 0;
}
