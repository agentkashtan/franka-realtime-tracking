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

#include <librealsense2/rs.hpp>

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
    zmq::context_t ctx(1);
    zmq::socket_t socket(ctx, zmq::socket_type::push);
    socket.connect("tcp://localhost:5555");


    rs2::pipeline pipe;
    rs2::config cfg;
    const int fps = 30;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, fps);
    cfg.enable_device("123622270300");
    
    auto realsense_callback = [&socket](const rs2::frame& frame) {
            rs2::frameset fs = frame.as<rs2::frameset>();
            if(!fs)return;
            rs2::video_frame cur_frame = fs.get_color_frame();
            rs2::depth_frame dframe = fs.get_depth_frame();
            if(!cur_frame || !dframe)return;

            rs2::video_frame video_frame = cur_frame.as<rs2::video_frame>();
            int width = video_frame.get_width();
            int height = video_frame.get_height();
            int channels = video_frame.get_bytes_per_pixel();
            int stride = video_frame.get_stride_in_bytes();

            const void* frame_data = video_frame.get_data();
            cv::Mat color_data(height, width, CV_8UC(channels), const_cast<void*>(frame_data), stride);
            cv::cvtColor(color_data, color_data, cv::COLOR_RGB2BGR);

            rs2::video_frame depth_frame = dframe.as<rs2::video_frame>();
            cv::Mat depth_data(
                            height, width, CV_16UC1,
                            const_cast<void*>(depth_frame.get_data()),
                            depth_frame.get_stride_in_bytes()
            );


            auto now = chrono::high_resolution_clock::now();
            auto since_epoch = chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch()).count();
            uint64_t timestamp_ns = static_cast<uint64_t>(since_epoch);
            zmq::message_t msg_timestamp(sizeof(timestamp_ns));
            std::memcpy(msg_timestamp.data(), &timestamp_ns, sizeof(timestamp_ns));

            vector<uchar> compressed_cframe;
            cv::imencode(".jpg", color_data, compressed_cframe);

            vector<uchar> compressed_dframe;
	    std::vector<int> compressionParams = { cv::IMWRITE_PNG_COMPRESSION, 3 };
            cv::imencode(".png", depth_data, compressed_dframe, compressionParams);            
	    
	    zmq::message_t msg_color(compressed_cframe.size());
            memcpy(msg_color.data(), compressed_cframe.data(), compressed_cframe.size());

	    zmq::message_t msg_depth(compressed_dframe.size());
            memcpy(msg_depth.data(), compressed_dframe.data(), compressed_dframe.size());

            socket.send(msg_color, zmq::send_flags::sndmore);
	    socket.send(msg_depth, zmq::send_flags::sndmore);
            socket.send(msg_timestamp, zmq::send_flags::none);















           // cv::imshow("realsense1", depthMat);
           // cv::imshow("realsense", image);
           // cv::waitKey(5);
    };
    rs2::pipeline_profile profile = pipe.start(cfg, realsense_callback);
    auto intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    cout << "fx: " << intrinsics.fx << endl << "ppx: " << intrinsics.ppx << endl << "fy: " << intrinsics.fy << endl << "ppy: " << intrinsics.ppy << endl;
    
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    depth_sensor.set_option(RS2_OPTION_DEPTH_UNITS, 0.001f);
    float current_depth_unit = depth_sensor.get_option(RS2_OPTION_DEPTH_UNITS);
    cout << "depth unit: " << current_depth_unit << endl;
    cout << "initialized camera" << endl;
    thread camera_data_receiver_worker(camera_data_receiver, ref(ctx));
    while(true) {}

    /*
    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceId = 0;
    int apiId = cv::CAP_ANY;
    cap.open(deviceId, apiId);
  	   
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
	
	zmq::message_t msg(compressed_frame.size());
	memcpy(msg.data(), compressed_frame.data(), compressed_frame.size());

	socket.send(msg, zmq::send_flags::sndmore);
	socket.send(msg_timestamp, zmq::send_flags::none);
    }
    */
    return 0;
}
