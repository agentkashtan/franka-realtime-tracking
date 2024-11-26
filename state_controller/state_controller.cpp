#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include <Eigen/Dense>
#include "franka_ik_He.hpp"
#include <cmath>  // for M_PI
#include "pinocchio/fwd.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "pinocchio/algorithm/frames.hpp"

#include <iostream>
#include <array>
#include <vector>
#include <variant>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

#include "kalman.h"

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <librealsense2/rs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


float markerLength = 0.07;
cv::Mat cameraMatrix(3,3, CV_32FC1), distCoeffs;

int frame_number = 0;
cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);



//cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

/*
bool keep_running;

std::mutex ik_mutex;
std::condition_variable ik_cv;
int ik_state = 0; // 0 ... ready, 1 ... computin, 2... result ready
// TODO: ik query
// TODO: ik result

void ik_thread() {
	while(keep_running) {
		{
		std::lock_guard<std::mutex> lock(ik_mutex);
		ik_cv.wait(lock);
		// now you have the lock, get the query copy it to a thread local variable
		}
		// compute
		{
		// lock again, write to result variable
		}
	}
}
*/
struct CameraData {
	bool detected;
	Eigen::Isometry3d pose;
        bool global_frame;
	bool new_data;
};

std::mutex cameraDataMutex;
CameraData cameraData = { false, Eigen::Isometry3d::Identity(), false, false };

struct APResult {
	std::atomic<bool> finished{false};
	Eigen::VectorXd q_config;
	double exe_time;
	std::mutex apResultMutex;
};

APResult apResult;



enum class State {
    Idle,
    Observing,
    ComputingApproachPoint,
    Approaching,
    VisualServoing,
    ErrorRecovery,

};

struct IdleParams {
    double start_time;
};

struct ObservationParams {
    double start_time;
    MovementEstimator* estimator;
};

struct ApproachPointComputingParams {
    double start_time;
    Eigen::Vector3d p_start;
    Eigen::Vector3d velocity;
    Eigen::Matrix3d orientation;	 
};

struct ApproachParams {
    double start_time;
    Eigen::VectorXd approach_config;
    double exe_time;
};

struct ObjPosition {
	double time_stamp;
	Eigen::Vector3d data;
};

struct VisualServoingParams {
    double start_time;
    double prev_time_stamp;
    bool init;
    Eigen::Vector3d measured_position;
    Eigen::Vector3d desired_position;
    Eigen::Matrix3d desired_orientation;
    MovementEstimator* estimator;
    Eigen::Vector3d lpf_state;
};

struct PreVisualServoingParams {
    bool updated;
    Eigen::VectorXd state;
    Eigen::MatrixXd P;
};


int sgn(double a) {
 return a > 0 ? 1 : -1;
};



array<double, 7> ZERO_TORQUES = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


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
                   cv::aruco::DetectorParameters d_params = cv::aruco::DetectorParameters();

                    cv::Mat image(height, width, CV_8UC(channels), const_cast<void*>(frame_data), stride);
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

                    if (ids.size() == 1 && ids.front() == 0) {
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
                        camera_T_tag.linear() = eigen_matrix;
                        Eigen::Vector3d eigen_translation;
                        cv::cv2eigen(tvec,eigen_translation);
                        camera_T_tag.translation() = eigen_translation;

                        std::lock_guard<std::mutex> lock(cameraDataMutex);
                        frame_number = 0;
                        cameraData = { true, g_T_c * camera_T_tag, false, true };
			//cout << camera_T_tag.translation().transpose() << endl;

                    } else {
                        std::lock_guard<std::mutex> lock(cameraDataMutex);
                        frame_number ++;
                        cameraData = { false, g_T_c * camera_T_tag, false, true };
                    }

                    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    cv::imshow("tasde", image);
                    cv::waitKey(5);
            } catch (std::exception const & ex ) {
                std::cout << ex.what() << std::endl;
            }
        };





class StateController {
public:
    StateController(franka::Model& model, int maxMissingFrames, int observationWindow)
        : maxMissingFrames(maxMissingFrames),
          observationWindow(observationWindow),
          model(model),
          state(State::Idle),
          missingFrames(0)
          {
            //q_base <<  1.4784838,   0.58908088, -1.51156758, -2.32406426,  0.75959274,  2.20523463, -2.89;
            log_ik_ = std::ofstream("/dev/shm/ik.log");
	    log_ = std::ofstream("/dev/shm/controller.log");
            
	    log_rp_ = std::ofstream("/dev/shm/real_pose.log");
            log_pp_ = std::ofstream("/dev/shm/predicted_pose.log");
            log_vel_ = std::ofstream("/dev/shm/velocity.log");
            log_ad_ = std::ofstream("/dev/shm/a_data.log");
	    log_vs_ = std::ofstream("/dev/shm/vs_data.log");
	    log_vs_rp_ = std::ofstream("/dev/shm/vs_data_actual_pos.log");
	    
	    
	    start = std::chrono::high_resolution_clock::now();
	    Kp.diagonal() << 200, 200, 200, 200, 200, 200,200;
            Kd.diagonal() << 20, 20, 20, 20, 20, 20, 8;
          }

    string time(){
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        return std::to_string(duration.count());
    }

    double d_time(){
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        return duration.count();
    }



    array<double, 7> update(double time1, franka::RobotState robotState) {
        time_stamp = time1;
        robot_state = robotState;
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        switch (state) {
            case State::Idle:
		return processIdle();
                break;
            case State::Observing:
		log_ << time() << " observing " << endl;
		return processObserving();
                break;
            case State::ErrorRecovery:
                return processErrorRecovery();
                break;
	    case State::ComputingApproachPoint:
		log_ << time() << " update: waiting" << endl;
		return processApproachPointComputationPhase();
		break;
            case State::Approaching:
                return processApproachPhase();
                break;
            case State::VisualServoing:
                return processVisualServoing();
                break;
        }
    }
    Eigen::Matrix<double, 7, 1> q_base;

private:
    int maxMissingFrames;
    int observationWindow;
    State state;

    std::ofstream log_;
    std::ofstream log_ik_;
    std::ofstream log_rp_;
    std::ofstream log_pp_;
    std::ofstream log_vel_;
    std::ofstream log_ad_;
    std::ofstream log_vs_;
    std::ofstream log_vs_rp_;

    std::chrono::high_resolution_clock::time_point start;
    double OFFSET_LENGTH = 0.25;
    int missingFrames;
    vector<Eigen::Vector3d> positions;
    double time_stamp;
    franka::RobotState robot_state;
    franka::Model& model;
    Eigen::DiagonalMatrix<double, 7> Kp;
    Eigen::DiagonalMatrix<double, 7> Kd;
    IdleParams idleParams;
    ObservationParams observationParams;
    ApproachParams approachParams;
    VisualServoingParams visualServoingParams;
    ApproachPointComputingParams approachPointComputingParams;
    PreVisualServoingParams preVisualServoingParams;

    bool getPose(Eigen::Vector3d& pose) {
        pose = Eigen::Vector3d::Zero();
        return true;
    } 

    void convert_to_global(CameraData& data) {
    	if (!data.global_frame) {
	   data.global_frame = true;
	   Eigen::Isometry3d T_ee_o(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
	   data.pose = T_ee_o * data.pose;
	}    
    }
    
    array<double, 7> maintain_base_pos() {

        array<double, 7> coriolis_array = model.coriolis(robot_state);

        Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

	Eigen::VectorXd tau_cmd =  Kp * ( q_base - q) + Kd * (- dq) + coriolis;
        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
        return tau_d_array;
    }

    array<double, 7> processIdle() {
	std::lock_guard<std::mutex> lock(cameraDataMutex);
	convert_to_global(cameraData);
        if (cameraData.detected) {
            startObserving(cameraData.pose.translation());
        }
	return maintain_base_pos();
    }

    void handleMissingFrame() {
        missingFrames++;
        if (missingFrames >maxMissingFrames) {
            cout << "Too many missing frames, transitioning to Error Recovery.\n";
            state = State::ErrorRecovery;
        }
    }

    //Handle observing state

    void startObserving(const Eigen::Vector3d& initialPosition) {
        state = State::Observing;
        observationParams.start_time = time_stamp;
	Eigen::MatrixXd temp_p = Eigen::MatrixXd::Identity(6, 6) * 2;
	temp_p.topLeftCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 2;
	Eigen::VectorXd init_state = Eigen::VectorXd::Zero(6);
	init_state.head(3) = initialPosition;
	observationParams.estimator = new MovementEstimator(init_state, time_stamp, temp_p);
        cameraData.new_data = false;	
        positions.clear();
        positions.push_back(initialPosition);
        missingFrames = 0;
     }

    array<double, 7> processObserving() {
        Eigen::Vector3d pose;
	std::lock_guard<std::mutex> lock(cameraDataMutex);
	convert_to_global(cameraData);
        if (cameraData.detected) {
           // cout << cameraData.pose.translation().transpose() << endl;
            missingFrames = 0;
	    observationParams.estimator->predict(time_stamp);
	    if (cameraData.new_data) {
	    	observationParams.estimator->correct(cameraData.pose.translation());
	 	cameraData.new_data = false;
	        positions.push_back(cameraData.pose.translation());
		//cout << positions.size() << endl;
		if (positions.size() > observationWindow) {
			auto res = observationParams.estimator->get_state();
			approachPointComputingParams = { time_stamp, res.first.head(3), res.first.tail(3), cameraData.pose.linear() };
                        preVisualServoingParams.updated = true;
		        preVisualServoingParams.state = res.first; 
		        preVisualServoingParams.P = res.second;
			delete observationParams.estimator;
			observationParams.estimator = nullptr;
			cout << "velocity: " << res.first.tail(3).transpose() << endl;
			startApproachPointComputationPhase();
		}
	    }
            if (observationParams.estimator!= nullptr){
            auto res = observationParams.estimator->get_state();
	    log_rp_ << cameraData.pose.translation().transpose() << endl;
            log_pp_ << res.first.head(3).transpose() << " " << res.second.norm() << endl;
            log_vel_ << res.first.tail(3).transpose() << endl;
            }
            /*
	    if (cameraData.newData) {
	    	observationParams.estimator->predict(time_stamp);
	   	auto res = observationParams.estimator->get_state();
	   	log_rp_ << cameraData.pose.translation().transpose() << endl;
	   	log_pp_ << res.first.transpose() << endl;
	    	log_vel_ << res.second.transpose() << endl;

	    	if (positions.size() > observationWindow) {    
                	log_ << time() << " started fit " << time_stamp<< endl; 
			pair<Eigen::Vector3d, Eigen::Vector3d> result = fitLine(positions, time_stamp - observationParams.start_time);
                	log_ << time() << " started obs: " << observationParams.start_time << " ended " << time_stamp<< endl;  
			approachPointComputingParams = { time_stamp,  result.first, result.second, cameraData.pose.rotation() };
			startApproachPointComputationPhase();
            	}
	    }
	  */
        } else {
            handleMissingFrame();
        }
	log_ << time() << " finished observing " << endl;
        return maintain_base_pos();
    }

    // Everything for first phase
    struct Vec {
        double x,y;
    };

    struct Workspace {
        Vec bl; // bottom-left
        Vec tr; // top-right
    };

    bool in_bounds(const Vec& cur_pos, const Workspace& rect) {
        return cur_pos.x >= rect.bl.x && cur_pos.x <= rect.tr.x && cur_pos.y >= rect.bl.y && cur_pos.y <= rect.tr.y;
    }

    double time_to_reach_workspace(const Vec& start, const Vec& velocity, const Workspace& rect) {
        if (in_bounds(start, rect)) return 0;
        double t_entry = numeric_limits<double>::infinity();

        if (velocity.x != 0) {
            double t_left  = (rect.bl.x - start.x) / velocity.x;
            double t_right = (rect.tr.x - start.x) / velocity.x;

            if (t_left > 0 && start.y + t_left * velocity.y >= rect.bl.y && start.y + t_left * velocity.y <= rect.tr.y) {
                t_entry = min(t_entry, t_left);
            }
            if (t_right > 0 && start.y + t_right * velocity.y >= rect.bl.y && start.y + t_right * velocity.y <= rect.tr.y) {
                t_entry = min(t_entry, t_right);
            }
        }

        if (velocity.y != 0) {
            double t_bottom = (rect.bl.y - start.y) / velocity.y;
            double t_top    = (rect.tr.y - start.y) / velocity.y;

            if (t_bottom > 0 && start.x + t_bottom * velocity.x >= rect.bl.x && start.x + t_bottom * velocity.x <= rect.tr.x) {
                t_entry = min(t_entry, t_bottom);
            }
            if (t_top > 0 && start.x + t_top * velocity.x >= rect.bl.x && start.x + t_top * velocity.x <= rect.tr.x) {
                t_entry = min(t_entry, t_top);
            }
        }
        return (t_entry == numeric_limits<double>::infinity()) ? -1 : t_entry;
    }


    Eigen::VectorXd compute_ik(Eigen::VectorXd q_init, Eigen::Vector3d position, Eigen::Matrix3d orientation, pinocchio::Model& model, pinocchio::Data& data) {
        // TODO: handle different oreinetions
        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        Eigen::Vector4d offset;
	offset << orientation.col(2) * OFFSET_LENGTH, 0;
	transformation(0, 3) = position.x();
        transformation(1, 3) = position.y();
        transformation(2, 3) = position.z();
        transformation.col(3) += offset;  	
        transformation.col(0) << orientation.col(0), 0;
	transformation.col(1) << -orientation.col(1), 0;
        transformation.col(2) <<- orientation.col(2), 0;
        
        double min_dist = 1000000000;
        Eigen::VectorXd best_solution(7);

        array<double, 16> flat_array;
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>(flat_array.data()) = transformation;
        array<double, 7> q_init_arr;
        for (int i = 0; i < 7; ++i) {
            q_init_arr[i] = q_init(i);
        }

        for (double q7 = -2.89; q7 <= 2.89; q7 += 0.05) {
            auto solutions = franka_IK_EE(flat_array, q7, q_init_arr);
            for (array<double, 7> q_arr: solutions) {
                bool valid_sol = true;
                for (auto i: q_arr) if (isnan(i)) valid_sol = false;
                if (!valid_sol) continue;
                Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(q_arr.data(), q_arr.size());
                Eigen::VectorXd q_temp(9);
                q_temp.head(7) = q;

                Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, model.nv);
                pinocchio::computeFrameJacobian(model, data, q_temp, 20, pinocchio::WORLD, jacobian);

                Eigen::EigenSolver<Eigen::MatrixXd> solver(jacobian.topLeftCorner(3, 7) * jacobian.topLeftCorner(3,7).transpose());
                Eigen::VectorXd eigenvalues = solver.eigenvalues().real();

                double dist = (q - q_init).norm();
                if (dist  < min_dist) {
                    best_solution = q;
                    min_dist= dist;
                }
            }
        }
        //std::cout << "max man " << max_manupuability << std::endl;
        if (min_dist < 1000000000) return best_solution;
        throw runtime_error("Couldnt #2ik reach the object.");
    }


    tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> get_q(double t, Eigen::VectorXd& q_init, Eigen::VectorXd& q_fin) {
        Eigen::VectorXd MAX_VELOCITIES(7);
        MAX_VELOCITIES << 2, 2, 2, 2, 2, 2, 2.6;
        MAX_VELOCITIES *= 0.15;
        Eigen::VectorXd MAX_ACCELERATIONS(7);
        MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
        MAX_ACCELERATIONS *= 0.15;
        Eigen::VectorXd q(7);
        Eigen::VectorXd q_dot(7);
        Eigen::VectorXd q_ddot(7);

        for (int i = 0; i < q_init.size(); i ++) {
            double q_a = pow(MAX_VELOCITIES[i], 2 ) / (2 * MAX_ACCELERATIONS[i]);
            double t_a = MAX_VELOCITIES[i] / MAX_ACCELERATIONS[i];
            double total_distance = abs(q_fin[i] - q_init[i]);

            int k = 1;
            if (q_fin[i] < q_init[i]) k = -1;

            if (total_distance > 2 * q_a) {
                double t_cs = (total_distance - 2 * q_a) / MAX_VELOCITIES[i];
                if (t > 2 * t_a + t_cs) {
                    q[i] = q_fin[i];
                    q_dot[i] = 0;
                    q_ddot[i] = 0;
                    continue;
                }
                if (t < t_a) {
                    q[i] = q_init[i] + k * pow(t, 2) * MAX_ACCELERATIONS[i] / 2;
                    q_dot[i] = k * (MAX_ACCELERATIONS[i] * t);
                    q_ddot[i] = k * MAX_ACCELERATIONS[i];

                } else if (t_a <= t && t <= t_a + t_cs) {
                    q[i] = q_init[i] + k * (pow(t_a, 2) * MAX_ACCELERATIONS[i] / 2 + (t - t_a) * MAX_VELOCITIES[i]);
                    q_dot[i] = k * (MAX_VELOCITIES[i]);
                    q_ddot[i] = 0;
                } else {
                    q[i] = q_init[i] + k * (pow(t_a, 2) * MAX_ACCELERATIONS[i] / 2 + t_cs * MAX_VELOCITIES[i] + (t - t_a - t_cs) * MAX_VELOCITIES[i] - std::pow(t - t_a - t_cs, 2) * MAX_ACCELERATIONS[i] / 2);
                    q_dot[i] = k * (MAX_VELOCITIES[i] - (t - t_a - t_cs) * MAX_ACCELERATIONS[i]);
                    q_ddot[i] = - k * MAX_ACCELERATIONS[i];
                }
            } else {
                double t_h = sqrt(total_distance / MAX_ACCELERATIONS[i]);
                if (t > 2 * t_h) {
                    q[i] = q_fin[i];
                    q_dot[i] = 0;
                    q_ddot[i] = 0;
                    continue;
                }
                if (t < t_h) {
                    q[i] = q_init[i] + k * (pow(t, 2) * MAX_ACCELERATIONS[i] / 2);
                    q_dot[i] = k * (MAX_ACCELERATIONS[i] * t);
                    q_ddot[i] = k * MAX_ACCELERATIONS[i];
                } else {
                    q[i] = q_init[i] + k * (pow(t_h, 2) * MAX_ACCELERATIONS[i] / 2 + (t - t_h) * t_h * MAX_ACCELERATIONS[i] - std::pow(t - t_h, 2) * MAX_ACCELERATIONS[i] / 2);
                    q_dot[i] = k * (MAX_ACCELERATIONS[i] * t_h - MAX_ACCELERATIONS[i] * (t - t_h));
                    q_ddot[i] = - k * MAX_ACCELERATIONS[i];
                }
            }
            if (total_distance < 0.005) q_ddot[i] = 0;
        }
        tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> res(q, q_dot, q_ddot);
        return res;
    }


    double completion_time(Eigen::VectorXd q_init, Eigen::VectorXd q_fin) {
        Eigen::VectorXd MAX_VELOCITIES(7);
        MAX_VELOCITIES << 2, 2, 2, 2, 2, 2, 2.6;
        MAX_VELOCITIES *= 0.15;
        Eigen::VectorXd MAX_ACCELERATIONS(7);
        MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
        MAX_ACCELERATIONS *= 0.15;

        double completion_time = 0;
        for (int i = 0; i < q_init.size(); i ++) {
            double q_a = pow(MAX_VELOCITIES[i], 2)/ (2 * MAX_ACCELERATIONS[i]);
            double t_a = MAX_VELOCITIES[i] / MAX_ACCELERATIONS[i];
            double q_d = q_a;
            double total_distance = abs(q_fin[i] - q_init[i]);

            if (total_distance > 2 * q_a) {
                double t_cs = (total_distance - 2 * q_a) / MAX_VELOCITIES[i];
                completion_time = max(completion_time, t_cs + 2 * t_a);
            } else {
                double t_h = sqrt(total_distance / MAX_ACCELERATIONS[i]);
                completion_time = max(completion_time, 2 * t_h);
            }
        }
        return completion_time;
    }


    pair<Eigen::VectorXd, double> compute_approach_point() {
        auto start = std::chrono::high_resolution_clock::now();
		    
	Eigen::Vector3d obj_position_init= approachPointComputingParams.p_start;
        Eigen::Vector3d obj_velocity = approachPointComputingParams.velocity;
	Eigen::VectorXd q_init = q_base;
	Eigen::Matrix3d desired_orientation = approachPointComputingParams.orientation;
	    
	//const Workspace ws = {{0.2, -0.4}, {0.6, 0.4}};
        const Workspace ws = {{-0.5,-0.6 }, {0.5, -0.2}};
	double init_time = time_to_reach_workspace({obj_position_init.x(), obj_position_init.y()}, {obj_velocity.x(), obj_velocity.y()}, ws);
        double cur_time = init_time;
	if (cur_time == -1) throw runtime_error("The object is out of the workspace.");
	Eigen::Vector3d obj_position = obj_position_init +  obj_velocity * init_time;
        double sample_duration = 0.01 / obj_velocity.norm();
        int iteration_num = 0;

        const string urdf_filename = "../urdf/panda.urdf";
        pinocchio::Model model;
        pinocchio::urdf::buildModel(urdf_filename, model);
        pinocchio::Data data(model);
        
        while (in_bounds({obj_position.x(), obj_position.y()}, ws)) {

            Eigen::VectorXd q_fin = compute_ik(q_init, obj_position, desired_orientation, model, data);
	    double execution_time = completion_time(q_init, q_fin);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
	    log_ik_ << time() << " current time: " << cur_time << "; obj pos:  " << obj_position.transpose() << "; execution time: " << execution_time << "; duration: " << duration.count() << "; total time: "<< execution_time + duration.count() << endl;
	    if (execution_time + sample_duration + duration.count() < cur_time) {	
		return {q_fin, execution_time};
            }
	    iteration_num ++;
            cur_time = init_time + sample_duration * iteration_num; 
            obj_position = obj_position_init +  obj_velocity * cur_time;
	}
        throw runtime_error("Could not reach the object.");
    }

    Eigen::VectorXd first_phase_controller(double t, Eigen::VectorXd q_init, Eigen::VectorXd q_fin, Eigen::VectorXd q_cur, Eigen::VectorXd q_dot_cur, Eigen::Matrix<double, 7,7> M) {
        auto [q_des, q_dot_des, q_ddot_des] = get_q(t, q_init, q_fin);
        return Kp * (q_des - q_cur) + Kd * (q_dot_des  - q_dot_cur) + M * q_ddot_des;
    }


    // Handle approach point calculation
    void compute_approach_point_mth()  {
	    log_ik_ << time() <<  " compute_approach_point_mth entry" << std::endl;
	    try { 
		    apResult.finished.store(false);
		    cout << "strated thread" << endl;
		    pair<Eigen::VectorXd, double> res = compute_approach_point();
		    cout << " computed approach point" << endl;
		    {
			    std::lock_guard<std::mutex> lock(apResult.apResultMutex);
			    apResult.q_config = res.first;
			    apResult.exe_time = res.second;
		    }
		    apResult.finished.store(true);
	    } catch (std::exception const & ex ) {
		    std::cerr << "compute_approach_point_mth catch ex: " << ex.what() <<std::endl;
		    log_ik_ << time() << "compute_approach_point_mth catch ex: " << ex.what() <<std::endl;

	    }
	    log_ik_ << time() <<  " compute_approach_point_mth exit" << std::endl;
    }

    std::thread new_thread_; 
    void startApproachPointComputationPhase() {
        log_ << endl << time() << " starting computing" << endl;
        apResult.finished.store(false);
	new_thread_ = std::thread(&StateController::compute_approach_point_mth, this);
        state = State::ComputingApproachPoint;
	log_ << endl << time() << " exited startApproachPointComputationPhase()" << endl;

    }

    array<double, 7> processApproachPointComputationPhase() {
    	log_  << time() << " waiting" << endl;
	if (apResult.finished.load()) {
	    log_ << endl << time() << " completed computing" << endl;
            std::lock_guard<std::mutex> lock(apResult.apResultMutex);		
            startApproachPhase(apResult.q_config, apResult.exe_time);
	} 
        return maintain_base_pos();
    }



    
    // Handle approaching state
    void startApproachPhase(Eigen::VectorXd approach_config, double exe_time) {
        log_ << endl << time() << " started approaching" << endl;
	state = State::Approaching;
        approachParams = {time_stamp, approach_config, exe_time};
    }

    array<double, 7> processApproachPhase() {
        
        if (approachParams.start_time <= time_stamp && time_stamp <= approachParams.start_time + approachParams.exe_time) {
            array<double, 7> coriolis_array = model.coriolis(robot_state);
            array<double, 49> mass_array = model.mass(robot_state);
            Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
            Eigen::Matrix4d T_ee_o(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

            Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
            tau_cmd  = first_phase_controller(time_stamp - approachParams.start_time, q_base, approachParams.approach_config, q, dq, mass);
	    tau_cmd += coriolis;
            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
            return tau_d_array;
        } else {
	    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            cout << "actual: " << q.transpose() << endl << "desired: "<< approachParams.approach_config.transpose() << endl;
	    cout << "diff: " << q.transpose() - approachParams.approach_config.transpose();
	    cout << "cur time: " <<  time_stamp << " " << approachParams.start_time + approachParams.exe_time << endl;   
	    Eigen::Matrix4d T_ee_o(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
	    cout <<"ee pose: " << T_ee_o(0,3) << " " << T_ee_o(1,3) << " " << T_ee_o(2,3) << endl;
	    //throw runtime_error("approached");
            return startVisualServoing();
        }
    }

    //Handle visual servoing state

    Eigen::VectorXd visual_servoing_controller(Eigen::VectorXd q, Eigen::VectorXd q_dot, Eigen::Isometry3d pose, Eigen::Vector3d obj_velocity, Eigen::Matrix<double, 6, 7>jacobian, Eigen::Matrix4d T_ee_0) {
        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(7, 7);
        Eigen::DiagonalMatrix<double, 6> Kp_ts(300, 300, 300, 30, 30, 30);
        Eigen::DiagonalMatrix<double, 7> Kd(20, 20, 20, 20, 20, 20, 8);
        
        Eigen::Vector3d offset;
        offset << pose.linear().col(2) * OFFSET_LENGTH;
	//disabled smart offset for now
        Eigen::Vector3d obj_position = pose.translation() + Eigen::Vector3d(0, 0, 0.2);// offset;
        
	Eigen::Vector3d error_pos = obj_position - T_ee_0.topRightCorner<3,1>();
        
        Eigen::Matrix3d r_desired = Eigen::Matrix3d::Identity();
        r_desired.col(0) << pose.linear().col(0);
        r_desired.col(1) << - pose.linear().col(1);
        r_desired.col(2) << - pose.linear().col(2);

	
	Eigen::Matrix3d r_diff = r_desired * T_ee_0.topLeftCorner<3,3>().transpose();
        Eigen::AngleAxisd angle_axis(r_diff);
	

        Eigen::Vector3d error_orient = angle_axis.axis() * angle_axis.angle();

        Eigen::VectorXd error_p(6);
        error_p.head(3) = error_pos;
        error_p.tail(3) = error_orient;
        log_ik_ << time() <<  " from VS " << error_p.transpose() << endl;
        Eigen::VectorXd q_dot_des_ts = Eigen::VectorXd::Zero(6);
        q_dot_des_ts.head(3) = obj_velocity;
        Eigen::VectorXd q_dot_des_js = w.inverse() * jacobian.transpose() * (jacobian * w.inverse() * jacobian.transpose()).inverse() * q_dot_des_ts;

        Eigen::VectorXd tau(7);
        tau = jacobian.transpose() * Kp_ts * error_p + Kd * (q_dot_des_js - q_dot);

	return tau;
    }




    array<double, 7> startVisualServoing() {
        state = State::VisualServoing;
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        q_base = q;
        
        if (!preVisualServoingParams.updated) throw runtime_error("init params for VS werent set");

	visualServoingParams = { time_stamp, time_stamp,  false, Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), Eigen::MatrixXd::Identity(3, 3), nullptr, Eigen::VectorXd::Zero(3) };
       	
	std::lock_guard<std::mutex> lock(cameraDataMutex);
        cameraData.new_data = false;

	
	return maintain_base_pos(); 
    } 

    array<double, 7> processVisualServoing() {
        Eigen::Vector3d pose;
        std::lock_guard<std::mutex> lock(cameraDataMutex);
        convert_to_global(cameraData);
        
	array<double, 7> coriolis_array = model.coriolis(robot_state);
        array<double, 49> mass_array = model.mass(robot_state);
        array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

        Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
        Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
        Eigen::Matrix4d T_ee_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

        q_base = q;


        double MAX_RATE = 0.0001;
        double alpha = 0.95;
        if (cameraData.new_data) {
		if (cameraData.detected) {
			if (!visualServoingParams.init) {
				visualServoingParams.init = true;
				visualServoingParams.desired_position = cameraData.pose.translation();
				visualServoingParams.lpf_state = cameraData.pose.translation();
				visualServoingParams.prev_time_stamp = time_stamp;

                                
				Eigen::VectorXd new_kf_state = preVisualServoingParams.state;
				new_kf_state.head(3) = cameraData.pose.translation();

				MovementEstimator* estimator = new MovementEstimator(new_kf_state, time_stamp, preVisualServoingParams.P);
                                visualServoingParams.estimator = estimator;

			}	
			
			visualServoingParams.measured_position = cameraData.pose.translation();
			visualServoingParams.desired_orientation = cameraData.pose.linear();
		        log_ik_ << time() <<  " --->new frame<---" << endl;
			visualServoingParams.estimator->correct(visualServoingParams.measured_position);

		}
	}

        log_ik_ << time() << " desired posistion " << visualServoingParams.desired_position.transpose()  << endl;
	log_ik_ << time() << "m measured posistion " << visualServoingParams.measured_position.transpose() << endl << endl;
         
        if (!visualServoingParams.init) return maintain_base_pos();
        

        visualServoingParams.estimator->predict(time_stamp);
	auto data = visualServoingParams.estimator->get_state();
	Eigen::Vector3d measured_position = data.first.head(3); 
        Eigen::Vector3d desired_change = measured_position - visualServoingParams.desired_position;

	desired_change = desired_change.cwiseMin(MAX_RATE);
	desired_change = desired_change.cwiseMax(- MAX_RATE);

	Eigen::Vector3d velocity = Eigen::Vector3d::Zero(3);

        if (time_stamp - visualServoingParams.prev_time_stamp > 0.00001) {
		visualServoingParams.desired_position += desired_change;
		Eigen::Vector3d prev_state = visualServoingParams.lpf_state;

		visualServoingParams.lpf_state = alpha * visualServoingParams.lpf_state + (1 - alpha) * visualServoingParams.desired_position;
                velocity = (visualServoingParams.lpf_state - prev_state) / (time_stamp - visualServoingParams.prev_time_stamp);
                //data.first.tail(3);  (visualServoingParams.lpf_state - prev_state) / (time_stamp - visualServoingParams.prev_time_stamp);
                visualServoingParams.prev_time_stamp = time_stamp;

        } else cout <<"min time exceeded" << endl; 

        log_vs_ << time() << " "  <<  visualServoingParams.measured_position.transpose() << " " << visualServoingParams.desired_position.transpose() << " " << measured_position.transpose() << " " << visualServoingParams.lpf_state.transpose() << " " << velocity.transpose() << endl;
	log_vs_rp_ << time() << " " << T_ee_0.topRightCorner<3,1>().transpose() << endl;

	Eigen::Isometry3d vs_pose_data;
        vs_pose_data.translation() = visualServoingParams.lpf_state; //visualServoingParams.desired_position;
        vs_pose_data.linear() = visualServoingParams.desired_orientation;
log_ik_ << "typost' " << visualServoingParams.desired_position << " " << vs_pose_data.translation().transpose() << endl;
	


        Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
        tau_cmd = visual_servoing_controller(q, dq, vs_pose_data, velocity, jacobian, T_ee_0); // get velocity somewhere ?
       
        //throw runtime_error("vot tak");

        tau_cmd += coriolis;

        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
        return tau_d_array;



/*


	if (cameraData.detected) {
 
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
            Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
            Eigen::Matrix4d T_ee_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            //
	    q_base = q;
	    //
            Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
            //

            Eigen::Vector3d velocity = approachPointComputingParams.velocity;//Eigen::Vector3d::Zero();
            //                        approachPointComputingParams = { time_stamp, res.first.head(3), res.first.tail(3), cameraData.pose.rotation() };
            Eigen::Isometry3d obj_pose = cameraData.pose;
	   
            tau_cmd = visual_servoing_controller(q, dq, cameraData.pose, velocity, jacobian, T_ee_0); // get velocity somewhere ?
	    tau_cmd += coriolis;

            std::array<double, 7> tau_d_array{};
	    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
                     //throw runtime_error("stopped vs");

	    return tau_d_array;

        } else {
            //handleMissingFrame();
	    cout << "frame not found" << endl;
            return maintain_base_pos();//visualServoingParams.tau_prev;
        }*/

    }


    array<double, 7> processErrorRecovery() {
        cout << "recovery";
        throw runtime_error("Entered recovery.");
        return ZERO_TORQUES;

    }

    

    Eigen::Vector3d calculateCentroid(const std::vector<Eigen::Vector3d>& points) {
        Eigen::Vector3d centroid(0, 0, 0);
        for (const auto& point : points) {
            centroid += point;
        }
        centroid /= points.size();
        return centroid;
    }

    Eigen::Matrix3d computeCovarianceMatrix(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& centroid) {
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto& point : points) {
            Eigen::Vector3d centered = point - centroid;
            covariance += centered * centered.transpose();
        }
        covariance /= points.size();
        return covariance;
    }

    pair<Eigen::Vector3d, Eigen::Vector3d> fitLine(const std::vector<Eigen::Vector3d>& points, double time) {
        Eigen::Vector3d centroid = calculateCentroid(points);

        Eigen::Matrix3d covarianceMatrix = computeCovarianceMatrix(points, centroid);


        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(covarianceMatrix);
        Eigen::Vector3d direction = eigenSolver.eigenvectors().col(2);

        double t_start = (points[0] - centroid).dot(direction) / direction.dot(direction);
        double t_end = (points.back() - centroid).dot(direction) / direction.dot(direction);
        Eigen::Vector3d p_start = centroid + t_start * direction;
        Eigen::Vector3d p_end = centroid + t_end * direction;

        std::cout << "Centroid of the fitted line: " << centroid.transpose() << std::endl;
        std::cout << "Direction of the fitted line: " << direction.transpose() << std::endl;
        std::cout << "Line equation: p(t) = [" << centroid.transpose() << "] + t * [" << direction.transpose() << "]" << std::endl;
        std::cout << "Starting point: " << p_start.transpose() << std::endl;
        return {p_end, (p_end - p_start) / time};
    }
};

int main(int argc, char ** argv) {
    using namespace pinocchio;
    



    // Robot set up
    
    if(!getenv("URDF"))throw std::runtime_error("no URDF env");

    const string urdf_filename = getenv("URDF");
    Eigen::VectorXd q_test_init(7);
    q_test_init << 1.4784838,   0.58908088, -1.51156758, -2.32406426,  0.75959274,  2.20523463, -2.89;

    franka::Robot robot(argv[1]);
    robot.automaticErrorRecovery();
    franka::Model model = robot.loadModel();
    double time = 0.0;
    franka::RobotState initial_state = robot.readOnce();
    string mode = argv[2];

    StateController controller(model, 600, 90);
    
    rs2::pipeline pipe;
    rs2::config cfg;
    const int fps = 30;
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, fps);
    cfg.enable_device("944122072327");
    pipe.start(cfg, realsense_callback);
    auto intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

   //TODO: convert instrinsics to OpenCV
    float camera_data[3][3] = {{intrinsics.fx, 0, intrinsics.width / 2.f}, {0, intrinsics.fy, intrinsics.height / 2.f}, {0, 0, 1}};
    cameraMatrix = cv::Mat(3, 3, CV_32FC1, &camera_data);
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q_init(initial_state.q.data());
    
    controller.q_base = q_init;
    cout << q_init.transpose() << endl;
    auto control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
        time += period.toSec();
        std::array<double, 7> tau_d_array{};
	/*
        if (mode == "true")  {
                array<double, 7> coriolis_array = model.coriolis(robot_state);
                array<double, 49> mass_array = model.mass(robot_state);
                Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
                Eigen::Matrix4d T_ee_o(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
                //tau_cmd = first_phase_controller(time, q_init, q_test_init, q, dq, mass);
                tau_cmd += coriolis;
                Eigen::VectorXd::Map(&t
        }
        else {
            tau_d_array = controller.update(time, robot_state);
        }*/
        //return tau_d_array;
	    tau_d_array = controller.update(time, robot_state);
           
  
    	    return tau_d_array;

    };
    robot.control(control_callback);


    return 0;
}
