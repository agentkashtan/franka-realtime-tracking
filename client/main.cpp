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

#include <thread> 
#include <chrono> 


using namespace std;


void camera_data_receiver(zmq::context_t& ctx) {
    zmq::socket_t socket_in(ctx, zmq::socket_type::pull);
    socket_in.connect("tcp://localhost:5554");
    while (true) {
	zmq::message_t msg;
	zmq::message_t msg_timestamp;
        
	socket_in.recv(msg, zmq::recv_flags::none);
	socket_in.recv(msg_timestamp, zmq::recv_flags::dontwait);    
	
	vector<double> pose(6); 
	memcpy(pose.data(), msg.data(), 6 * sizeof(double));
	Eigen::Isometry3d camera_T_tag;
	Eigen::Vector3d rotation_vector(pose[0], pose[1], pose[2]);

	camera_T_tag.linear() = Eigen::AngleAxis(
			rotation_vector.norm(), rotation_vector.normalized()
			).toRotationMatrix();
    	camera_T_tag.translation() = Eigen::Vector3d(pose[3], pose[4], pose[5]);

	uint64_t timestamp_ns;
        memcpy(&timestamp_ns, msg_timestamp.data(), sizeof(timestamp_ns));
	auto now = chrono::high_resolution_clock::now();
        auto start = chrono::time_point<chrono::high_resolution_clock>(chrono::nanoseconds(timestamp_ns));
	chrono::duration<double> duration = now - start;

	cout << "full cycle: " << duration.count() <<  " " << camera_T_tag.translation().transpose() << endl;
	
    }
}



int main()
{   
    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceId = 0;
    int apiId = cv::CAP_ANY;
    cap.open(deviceId, apiId);
  
    zmq::context_t ctx(1);
    zmq::socket_t socket(ctx, zmq::socket_type::push);
    socket.connect("tcp://localhost:5555");
  	
    thread camera_data_receiver_worker(camera_data_receiver, ref(ctx));

    if (!cap.isOpened()) { 
	    std::cout << "unable to open camera "; 
    	return 0;
    }	
    chrono::duration<double> duration;

    while (true) {
    	cap.read(frame);
	if (frame.empty()) {
            std::cout << "blank frame";
	    break;
	}
        auto now = chrono::high_resolution_clock::now();
        auto since_epoch = chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch()).count();
        uint64_t timestamp_ns = static_cast<uint64_t>(since_epoch);
        zmq::message_t msg_timestamp(sizeof(timestamp_ns));
        std::memcpy(msg_timestamp.data(), &timestamp_ns, sizeof(timestamp_ns));

	vector<uchar> compressed_frame;
	cv::imencode(".jpg", frame, compressed_frame);
	
	//auto now1= chrono::high_resolution_clock::now();

	zmq::message_t msg(compressed_frame.size());
	memcpy(msg.data(), compressed_frame.data(), compressed_frame.size());

	socket.send(msg, zmq::send_flags::sndmore);
	socket.send(msg_timestamp, zmq::send_flags::none);
	//cout << (chrono::duration<double>(chrono::high_resolution_clock::now() - now1)).count() << endl;


    }

    return 0;
}
