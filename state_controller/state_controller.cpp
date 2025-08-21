#include <netinet/in.h>
#include <sys/socket.h>

#include <Eigen/Dense>
#include <cmath>  // for M_PI

#include <iostream>
#include <array>
#include <vector>
#include <variant>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cmath>

#include "kalman.h"
#include "trajectory_generator.h"

#include <franka/gripper.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <librealsense2/rs.hpp>

#include <zmq.hpp>
#include <msgpack.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

using namespace std;


string BASE_IP;

float markerLength = 0.07;
cv::Mat cameraMatrix(3,3, CV_32FC1), distCoeffs;

int frame_number = 0;
cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

struct CameraData {
    bool detected;
    Eigen::Isometry3d pose;
    bool global_frame;
    bool new_data;
    bool is_logged;
    int tracking_time;
    int frameNumber;
    std::chrono::time_point<std::chrono::high_resolution_clock> processTime;
    Eigen::Isometry3d o_T_g;
};


mutex iterationMutex;
int iterationIdx = 0;

std::mutex cameraDataMutex;
CameraData cameraData = { false, Eigen::Isometry3d::Identity(), false, true  };

struct StateEntry {
    double timestamp;
    Eigen::Matrix<double, 6, 1> state;
    Eigen::Matrix<double, 6, 6> P;
};

class RingBuffer {
    public:
        explicit RingBuffer(size_t capacity) : capacity_(capacity), data_(capacity), start_(0), end_(0), full_(false) {
        }

        void push_back(StateEntry entry) {
            data_[end_] = entry;
            if (full_) start_ = (start_ + 1) % capacity_;
            end_ = (end_ + 1) % capacity_;
            full_ = (end_ == start_);
            //cout << "pushed " << entry.state.topLeftCorner<3, 1>().transpose() << endl;
        }

        StateEntry get_state(double ts) {
            int i = end_ - 1;
            if (!full_) {
                while (i >= 0 && static_cast<size_t>(i) >= start_) {
                    if (data_[i].timestamp <= ts) {
                       return data_[i];
                    }
                    i --;
                }
                return data_[start_];
            }
            while (i >= 0) {
                if (data_[i].timestamp <= ts) { 
                    return data_[i];
                }
                i --;
            }
            i = capacity_ - 1;
            while (i >= 0 && static_cast<size_t>(i) >= start_) {
                if (data_[i].timestamp <= ts) {
                    return data_[i];
                }
                i --;
            }
            return data_[start_];
        }
    private:
        size_t capacity_;
        std::vector<StateEntry> data_;
        size_t start_;
        size_t end_;
        bool   full_;
};

struct APResult {
    std::atomic<bool> finished{false};
    double reachTime;
    vector<vector<Eigen::VectorXd>> spline;
    double intervalDuration;
    Eigen::Isometry3d predictedObjectPose;
    std::mutex apResultMutex;
};

APResult apResult;


std::atomic<bool> workerFinished{false};
std::mutex workerMutex;

struct SharedEndEffectorPose {
    mutex mtx;
    Eigen::Matrix4d pose;
};

SharedEndEffectorPose sharedEEPose;

enum class State {
    MoveToStartingPoint,
    RestartServer,
    Idle,
    Registration,
    ComputingApproachPoint,
    Approaching,
    VisualServoing,
    Lifting,
};

struct MoveToStartingPointParams {
	double startTime;
	bool init;
    Eigen::VectorXd qInit;
	Eigen::VectorXd qFinal;
};

struct RestartServerParams {
	bool init = false;
};

enum class VisualServoingSubState {
    Following,
    Approaching,
    Grasping,
};

enum class LiftingSubState {
    Lift,
    ComputeTransport,
    Transport,
    PlacementDown,
    DropOff,
    PlacementUp,
};


struct IdleParams {
    double start_time;
};

struct RegParams {
    double start_time;
    Eigen::Vector3d init_position;
    Eigen::Matrix3d init_orientation;
    vector<Eigen::VectorXd> coeffs;
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
    double startTime;
    double executionTime;
    vector<vector<Eigen::VectorXd>> spline;
    int intervalNumber;
    double intervalDuration;
    Eigen::Isometry3d predictedObjectPose;
};

struct ObjPosition {
	double time_stamp;
	Eigen::Vector3d data;
};

struct VisualServoingParams {
    double startTime;
    double previousTimeStamp;
    Eigen::Vector3d filteredObjectPosition; 
    Eigen::Matrix3d filteredObjectOrientation;
    Eigen::Matrix3d estimatedObjectOrientation;
    MovementEstimator* estimator;
    RingBuffer* ringBuffer;
    VisualServoingSubState subState;
    double newData;
    Eigen::Vector3d previousAcceleration;
    double newDataTimestamp;
    vector<Eigen::VectorXd> coeffs;
    Eigen::Vector3d offset;
};

struct LiftingParams {
    double startTime;
    LiftingSubState subState;
    Eigen::Isometry3d currentPose;
    Eigen::Isometry3d liftingGoalPose;
    Eigen::VectorXd transportConfig;
    double transportTime;
    vector<Eigen::VectorXd> transportCoeffs;
    Eigen::Isometry3d actualGrasp;
};


int sgn(double a) {
 return a > 0 ? 1 : -1;
};

array<double, 7> ZERO_TORQUES = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

void adjustComputeThreadPriority(std::thread& compute_thread, bool f=true) {
    struct sched_param param;
    int policy;

    // Get current thread scheduling parameters (compute thread)
    errno = pthread_getschedparam(compute_thread.native_handle(), &policy, &param);
    if (errno) {
        perror("ERROR: pthread_getschedparam");
        throw std::runtime_error("errno != 0");
    }

    // Decrease the priority of the compute thread, ensuring it doesn't go below 0
    if (f) param.sched_priority = 1; else {
        param.sched_priority = 1;//std::max(static_cast<int>(param.sched_priority) - 1, 1);
        cout << "new priority " << param.sched_priority << endl;
    }
    // Debug message for starting the compute thread with the new priority
    //std::cout << "Starting ModelPinocchio compute_thread with priority: "
    //          << param.sched_priority << std::endl;
   
        // Set the new scheduling parameters for the compute thread
    errno = pthread_setschedparam(compute_thread.native_handle(), policy, &param);
    if (errno) {
        perror("ERROR: pthread_setschedparam");
        throw std::runtime_error("errno != 0");
    }
    if (!f) {
        errno = pthread_getschedparam(compute_thread.native_handle(), &policy, &param);
        cout << "real new priority " << static_cast<int>(param.sched_priority) <<endl;
    }

}

class StateController {
public:
    StateController(franka::Model& model, franka::Gripper& gripper, zmq::socket_t& socket_control, Eigen::VectorXd qInit, bool grasp, string exp_hash, int exp_num, string exp_dir)
        : model(model),
          gripper_(gripper),
	  socket_control_(socket_control),
          q_base(qInit),
          graspObject_(grasp),
          exp_hash_(exp_hash),
          run_number_(exp_num),
          exp_dir_(exp_dir),
          state(State::MoveToStartingPoint) {
            log_ik_ = std::ofstream("/dev/shm/ik.log");
	        log_ = std::ofstream("/dev/shm/controller.log");
            
	        log_rp_ = std::ofstream("/dev/shm/real_pose.log");
            log_pp_ = std::ofstream("/dev/shm/predicted_pose.log");
            log_vel_ = std::ofstream("/dev/shm/velocity.log");
            log_ad_ = std::ofstream("/dev/shm/a_data.log");
	        log_vs_ = std::ofstream("/dev/shm/vs_data.log");
	        log_vs_rp_ = std::ofstream("/dev/shm/vs_data_actual_pos.log");
	        
            log_tq_ = std::ofstream("/dev/shm/tq_data.log");
            log_tq_ff_ = std::ofstream("/dev/shm/tq_ff_data.log");


	        start = std::chrono::high_resolution_clock::now();
	        Kp.diagonal() << 200, 200, 200, 200, 200, 200,200;
            Kd.diagonal() << 20, 20, 20, 20, 20, 20, 8;
            
            //CONVEYOR_BELT_SPEED << -0.259, 0.0, 0.0; //high speed
            //double MAX_CARTESIAN_VELOCITY = 0.7;
	    CONVEYOR_BELT_SPEED << -0.5, 0.0, 0.0;
            OFFSET << 0.0, 0, 0.23;   
            GRASPING_TRANSFORMATION << 0, 1, 0,
                                      -1, 0, 0,
                                      0, 0, 1;
            
            Eigen::AngleAxisd rotZPI(M_PI, Eigen::Vector3d::UnitZ());
            ALTERNATIVE_GRASPING_TRANSFORMATION = rotZPI.toRotationMatrix();
            if (graspObject_) GRASP_OFFSET << 0, 0, 0.1;
            else GRASP_OFFSET = OFFSET;
            LIFT_HEIGHT = 0.5;
            DROP_OFF_POSITION << 0.55, -0.1, 0; 
            OBJECT_BOTTOM_TO_CENTRE = 0.075;
	    STARTING_CONFIGURATION = Eigen::VectorXd(7);
            STARTING_CONFIGURATION << -0.420854,  0.181446, -0.535689,  -1.50956,  0.301637 ,  1.80662, -0.191084;
            moveToStartingPointParams = {0, true, qInit, STARTING_CONFIGURATION };
            configure_logs_path();            
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

    void set_q_base(Eigen::Matrix<double, 7, 1> q) {
        q_base = q;
    }

    string get_unix_time_ms() {
        auto now = std::chrono::system_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        return to_string(epoch_ms) +  " ";
    }

    array<double, 7> update(double time, franka::RobotState robotState) {
        time_stamp = time;
        robot_state = robotState;
     
        const auto& jacobianArray = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
        const auto& massArray = model.mass(robot_state);
        const auto& coriolisArray = model.coriolis(robot_state);

        jacobian_ = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(jacobianArray.data());
        M_ = Eigen::Map<const Eigen::Matrix<double, 7, 7>>(massArray.data());
        coriolis_ = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(coriolisArray.data());
        q_ = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state.q.data());
        dq_ = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state.dq.data());
    
        Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        if(state==State::VisualServoing || state==State::Approaching) {
            if( T_EE_0(2,3) <= 0.04 || T_EE_0(1,3) >= -0.33) {
                throw runtime_error("outside of workspace: translation");
            }
            if (T_EE_0.block<3,1>(0,2).dot(-Eigen::Vector3d::UnitZ()) < 0.7) {
                std::cout << T_EE_0.block<3,1>(0,2).transpose() << std::endl;
                throw runtime_error("Outside of workspace: orientation");
            }
        }
        {
            lock_guard<mutex> lock(sharedEEPose.mtx);
            sharedEEPose.pose = T_EE_0;
        }
        
        log_robot_state_ << get_unix_time_ms();
        for (int i = 0; i < 4; ++ i) {
            for (int j = 0; j < 4; ++j) {
                log_robot_state_ << T_EE_0(i, j) << " ";        
            }
        }
        for (int i = 0; i < 7; ++ i) log_robot_state_ << q_(i) << " ";
        log_robot_state_ << static_cast<int>(state) << endl;
        
        {
            std::lock_guard<std::mutex> lock(cameraDataMutex);
            if (!cameraData.is_logged) {
                cameraData.is_logged = true;
                convert_to_global(cameraData);

                auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cameraData.processTime.time_since_epoch()).count();

                log_measurements_ << get_unix_time_ms() + to_string(epoch_ms) + " " + to_string(cameraData.tracking_time) + " " + to_string(cameraData.frameNumber) + " " ;
                for (int i = 0; i < 4; ++ i) {
                    for (int j = 0; j < 4; ++j) {
                        log_measurements_  << cameraData.pose.matrix()(i, j) << " ";        
                    }
                }
                log_measurements_ << endl;
            }                
            if ((cameraData.pose.translation()(2) < -0.06) && state!=State::VisualServoing) {
                //std::cout << "object low: " << cameraData.pose.translation()(2) << std::endl;
                //throw runtime_error("Object detected too low: stop");
            }
        }

        switch (state) {
            case State::MoveToStartingPoint:
                return processMoveToStartingPoint();            
	        case State::RestartServer:
		        return processRestartServer();
	        case State::Idle:
		        return processIdle();
            case State::Registration:
                return processRegistration(); 
            case State::ComputingApproachPoint:
                return processApproachPointComputationPhase();  
            case State::Approaching:
                return processApproachPhase();
            case State::VisualServoing:
                return processVisualServoing();
            case State::Lifting:
                return processLifting();
            default:
                throw runtime_error("Received unexpected controller state");    
        }
    }
    
private:
    franka::Model& model;
    franka::Gripper& gripper_;
    zmq::socket_t& socket_control_;
    Eigen::Matrix<double, 7, 1> q_base;
    bool graspObject_;
    string exp_hash_;
    State state;
    int run_number_;
    string exp_dir_;
    Eigen::Matrix<double, 6, 7> jacobian_;
    Eigen::Matrix<double, 7, 7> M_;
    Eigen::Matrix<double, 7, 1> q_;
    Eigen::Matrix<double, 7, 1> dq_;
    Eigen::Matrix<double, 7, 1> coriolis_;
    
    std::ofstream log_;
    std::ofstream log_ik_;
    std::ofstream log_rp_;
    std::ofstream log_pp_;
    std::ofstream log_vel_;
    std::ofstream log_ad_;
    std::ofstream log_vs_;
    std::ofstream log_vs_rp_;
    ofstream log_tq_;
    ofstream log_tq_ff_;

    ofstream log_robot_state_;
    ofstream log_measurements_;
    ofstream log_kalmanfilter_;
    ofstream log_controller_input_;
    ofstream log_filtered_position_;


    std::chrono::high_resolution_clock::time_point start;
    Eigen::Vector3d OFFSET;
    Eigen::Vector3d CONVEYOR_BELT_SPEED;
    Eigen::Matrix3d GRASPING_ORIENTATION;
    Eigen::Matrix3d GRASPING_TRANSFORMATION;
    Eigen::Matrix3d ALTERNATIVE_GRASPING_TRANSFORMATION;
    Eigen::Vector3d GRASP_OFFSET;
    Eigen::Vector3d DROP_OFF_POSITION;
    Eigen::VectorXd STARTING_CONFIGURATION;
    double LIFT_HEIGHT;
    double OBJECT_BOTTOM_TO_CENTRE;

    double REGISTRATION_DURATION_ESTIMATE = 0.15;
    double MAX_CARTESIAN_VELOCITY = 0.7; //high speed
    //double MAX_CARTESIAN_VELOCITY = 0.40;    
    
    int N = 20;

    int missingFrames;
    vector<Eigen::Vector3d> positions;
    double time_stamp;
    franka::RobotState robot_state;
    Eigen::DiagonalMatrix<double, 7> Kp;
    Eigen::DiagonalMatrix<double, 7> Kd;
    IdleParams idleParams;
    RegParams regParams;

    ObservationParams observationParams;
    ApproachParams approachParams;
    VisualServoingParams visualServoingParams;
    ApproachPointComputingParams approachPointComputingParams;
    LiftingParams liftingParams;
    MoveToStartingPointParams moveToStartingPointParams;
    RestartServerParams restartServerParams;
    
    bool getPose(Eigen::Vector3d& pose) {
        pose = Eigen::Vector3d::Zero();
        return true;
    } 

    void configure_logs_path() {
        log_robot_state_ = std::ofstream("/dev/shm/experiment/" + exp_dir_ + "/"  + exp_hash_ + "_robotstate_" + to_string(run_number_) + ".log");
        log_measurements_ = std::ofstream("/dev/shm/experiment/"  + exp_dir_ + "/" + exp_hash_ + "_measurements_" + to_string(run_number_) + ".log");
        
        log_kalmanfilter_ = std::ofstream("/dev/shm/experiment/"  + exp_dir_ + "/" + exp_hash_ + "_kalmanfilter_" + to_string(run_number_) + ".log");
        log_controller_input_ = std::ofstream("/dev/shm/experiment/"  + exp_dir_ + "/" + exp_hash_ + "_controllerinput_" + to_string(run_number_) + ".log");
        
        log_filtered_position_ = std::ofstream("/dev/shm/experiment/"  + exp_dir_ + "/" + exp_hash_ + "_filteredposition_" + to_string(run_number_) + ".log");
        
        ++ run_number_;
    }

    //make sure to lock camera mutex before calling this function
    void convert_to_global(CameraData& data) {
        if (!data.global_frame) {
	        data.global_frame = true;
	        data.pose = data.o_T_g * data.pose;
	    }    
    }

    //make sure to lock camera mutex before calling this function
    void convertToEEFrame(CameraData& data) {
        if (data.global_frame) {
            data.global_frame = false;
            data.pose = data.o_T_g.inverse() * data.pose;
        }
    }

    array<double, 7> maintain_base_pos() {
	    Eigen::VectorXd tauCmd =  Kp * (q_base - q_) + Kd * (- dq_) + coriolis_;
        std::array<double, 7> tauDArray{};
        Eigen::VectorXd::Map(&tauDArray[0], 7) = tauCmd;
        return tauDArray;
    }

    tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> get_q(double t, Eigen::VectorXd& q_init, Eigen::VectorXd& q_fin) {
	Eigen::VectorXd MAX_VELOCITIES(7);
	MAX_VELOCITIES << 2, 2, 2, 2, 2, 2, 2.6;
	MAX_VELOCITIES *= 0.3;
	Eigen::VectorXd MAX_ACCELERATIONS(7);
	MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
	MAX_ACCELERATIONS *= 0.3;
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
    Eigen::VectorXd first_phase_controller(double t, Eigen::VectorXd q_init, Eigen::VectorXd q_fin) {
    	Eigen::DiagonalMatrix<double, 7> Kp(200, 200, 200, 200, 200, 200,200);
    	Eigen::DiagonalMatrix<double, 7> Kd(20, 20, 20, 20, 20, 20, 8);
    	auto [q_des, q_dot_des, q_ddot_des] = get_q(t, q_init, q_fin);
    	return Kp * (q_des - q_) + Kd * (q_dot_des  - dq_) + M_ * q_ddot_des;
     }

    Eigen::VectorXd cartesianController(Eigen::Vector3d positionD, Eigen::Matrix3d orientationD, Eigen::Vector3d objectVelocity, bool logTorques = false, Eigen::Vector3d accelerationD = Eigen::Vector3d::Zero()) {
        Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

        const double translational_stiffness{500.0};
        const double rotational_stiffness{50.0};
        Eigen::MatrixXd stiffness(6, 6), damping(6, 6), integral(6, 6);
        stiffness.setZero();
        stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
        stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
        damping.setZero();
        Eigen::MatrixXd M_task_space = (jacobian_ * M_.inverse() * jacobian_.transpose()).inverse();
        const double xi = 0.5;
        for(int idx=0;idx<3;idx++) {
            damping(idx,idx) = 2.0 * xi * sqrt(translational_stiffness *
                                               M_task_space(idx,idx));
            damping(3+idx,3+idx) = 2.0 * xi * sqrt(rotational_stiffness *
                                               M_task_space(3+idx,3+idx));
        }

        Eigen::Vector3d error_pos = positionD - T_EE_0.topRightCorner<3,1>();
        Eigen::Matrix3d r_diff = orientationD * T_EE_0.topLeftCorner<3,3>().transpose();
        Eigen::AngleAxisd angle_axis(r_diff);
        Eigen::Vector3d error_orient = angle_axis.axis() * angle_axis.angle();

        Eigen::VectorXd error_p(6);
        error_p.head(3) = error_pos;
        error_p.tail(3) = error_orient;
        Eigen::VectorXd dqDesiredTaskSpace = Eigen::VectorXd::Zero(6);
        dqDesiredTaskSpace.head(3) = objectVelocity;

        Eigen::VectorXd tau(7);
        Eigen::VectorXd ff(6);
        
        Eigen::VectorXd accelerationDesired6d(6);
        accelerationDesired6d.setZero();
        accelerationDesired6d.head(3) = accelerationD;
        ff = M_task_space * accelerationDesired6d;
        //cout  << " ; ff: " << ff.norm() << " ;pos tq: " << (jacobian_.transpose() * ( stiffness * error_p)).norm() << " ;vel tq: " << (jacobian_.transpose() * damping * (dqDesiredTaskSpace - jacobian_ * dq_)).norm() << endl;
        tau = jacobian_.transpose() * ( stiffness * error_p + damping * (dqDesiredTaskSpace - jacobian_ * dq_) + ff);
        
        if (logTorques) {
            cout << "CC error. pos: " << error_pos.norm() << "; orient: " << error_orient.norm() <<  " " << error_orient.transpose() << endl;
            cout << "Vel error " << (dqDesiredTaskSpace - jacobian_ * dq_).norm() << " " << (dqDesiredTaskSpace - jacobian_ * dq_).transpose() << endl;
            cout << "FF error" << ff.norm() << " " << ff.transpose() << endl << endl; 
        }
        return tau;
    }
    
    array<double, 7> processMoveToStartingPoint() {
        if (!moveToStartingPointParams.init) { 
            moveToStartingPointParams = {time_stamp, true, q_base, STARTING_CONFIGURATION };
        }
        if ((q_ - moveToStartingPointParams.qFinal).norm() < 0.03) {
		    q_base = moveToStartingPointParams.qFinal;
	        state = State::RestartServer;
            restartServerParams.init = false;
		    return maintain_base_pos();
	    }
        Eigen::VectorXd tauCmd  = first_phase_controller(time_stamp - moveToStartingPointParams.startTime, moveToStartingPointParams.qInit, moveToStartingPointParams.qFinal);        
	    tauCmd += coriolis_;
	    std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
        return tau_d_array;
    }

    void restartServer() {
    	string msg = "restart";
    	zmq::message_t request(msg.begin(), msg.end());
    	socket_control_.send(request, zmq::send_flags::none);
    	zmq::message_t reply;
    	socket_control_.recv(reply, zmq::recv_flags::none);
        string reply_msg(static_cast<char*>(reply.data()), reply.size());
        cout << "Response from server: " << reply_msg << endl;
    	workerFinished.store(true, std::memory_order_release);
    }

    array<double, 7> processRestartServer() {  
        if (!restartServerParams.init) {
            cout << "Sent restart command" << endl;
            workerFinished.store(false, std::memory_order_release);
                workerThread_ = thread(&StateController::restartServer, this);		
                adjustComputeThreadPriority(workerThread_);
                restartServerParams.init = true;
        } else {
            if (workerFinished.load(std::memory_order_acquire)) {
                workerThread_.join();
                {
                    lock_guard<mutex> lock(iterationMutex);
                    iterationIdx ++;
                }
                configure_logs_path();
                state = State::Idle;
                std::lock_guard<std::mutex> lock(cameraDataMutex);
                {
                   cameraData.new_data = false;
                }
               GRASPING_TRANSFORMATION << 0, 1, 0,
                                         -1, 0, 0,
                                          0, 0, 1;

                cout << "Server restarted" << endl;
            }
        }
    	return maintain_base_pos();
    }

    array<double, 7> processIdle() {
        std::lock_guard<std::mutex> lock(cameraDataMutex);
        if (cameraData.new_data && cameraData.detected) {
            cameraData.new_data = false;
            Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
 
            vector<Eigen::VectorXd> coeffs(3);

            Eigen::Vector3d predictedPosition = T_EE_0.topRightCorner<3,1>() + CONVEYOR_BELT_SPEED * REGISTRATION_DURATION_ESTIMATE;
            for (int i = 0; i < 3; i ++) {
                Eigen::Matrix<double, 6, 6> A;
                double predictionTimeDelta = REGISTRATION_DURATION_ESTIMATE;
                A <<  pow(0, 5),     pow(0, 4),     pow(0, 3),     pow(0, 2),  0, 1.0,
                    pow(predictionTimeDelta, 5),     pow(predictionTimeDelta, 4),     pow(predictionTimeDelta, 3),     pow(predictionTimeDelta, 2),  predictionTimeDelta, 1.0,
                    5.0*pow(0, 4), 4.0*pow(0, 3), 3.0*pow(0, 2), 2.0*0,      1.0, 0.0,
                    5.0*pow(predictionTimeDelta, 4), 4.0*pow(predictionTimeDelta, 3), 3.0*pow(predictionTimeDelta, 2), 2.0*predictionTimeDelta,      1.0, 0.0,
                    20.0*pow(0, 3),12.0*pow(0, 2), 6.0*0,         2.0,         0.0, 0.0,
                    20.0*pow(predictionTimeDelta, 3),12.0*pow(predictionTimeDelta, 2),6.0*predictionTimeDelta,         2.0,         0,0;
                Eigen::Matrix<double, 6, 1> B;
                B << T_EE_0.topRightCorner<3,1>()(i), predictedPosition(i), 0, CONVEYOR_BELT_SPEED(i), 0, 0;
                coeffs[i] = A.fullPivLu().solve(B);
            }
           
            regParams = { time_stamp, T_EE_0.topRightCorner<3,1>(), T_EE_0.topLeftCorner<3,3>(), coeffs };
            state = State::Registration;
        }
        return maintain_base_pos();
    }
    
    array<double, 7> processRegistration() {
        std::lock_guard<std::mutex> lock(cameraDataMutex);
        if (!cameraData.new_data) {          
            Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            log_rp_ << time() << " " <<  T_EE_0.topRightCorner<3,1>().transpose()<< " " << T_EE_0.topLeftCorner<3,3>().eulerAngles(0, 1, 2).transpose() << endl;
            log_vs_rp_ << time() << " " << (regParams.init_position + (time_stamp - regParams.start_time) * CONVEYOR_BELT_SPEED).transpose() << " " << regParams.init_orientation.eulerAngles(0, 1, 2).transpose()  << endl;
            q_base = q_;
            Eigen::VectorXd tauCmd;
            if (time_stamp <= regParams.start_time + REGISTRATION_DURATION_ESTIMATE) {

                Eigen::Vector3d controllerInputPosition;
                Eigen::Vector3d controllerInputVelocity;
                Eigen::Vector3d controllerInputAcceleration;
                vector<Eigen::VectorXd> c = regParams.coeffs;
                double t = time_stamp - regParams.start_time;
                for (int i =0; i < 3; i ++) {
                    controllerInputPosition(i) = c[i](0) * pow(t, 5) + c[i](1) * pow(t, 4) + c[i](2) * pow(t, 3) + c[i](3) * pow(t, 2) + c[i](4) * t + c[i](5);
                    controllerInputVelocity(i) = 5 * c[i](0) * pow(t, 4) + 4 * c[i](1) * pow(t, 3) + 3 * c[i](2) * pow(t, 2) + 2 * c[i](3) * t  + c[i](4);
                    controllerInputAcceleration(i) = 20 * c[i](0) * pow(t, 3) + 12 * c[i](1) * pow(t, 2) + 6 * c[i](2) * t + 2 * c[i](3);
                }
                tauCmd = cartesianController(controllerInputPosition, regParams.init_orientation, controllerInputVelocity, false, controllerInputAcceleration);
            } else { 
                tauCmd = cartesianController(regParams.init_position + (time_stamp - regParams.start_time) * CONVEYOR_BELT_SPEED, regParams.init_orientation, CONVEYOR_BELT_SPEED);
            }
            tauCmd += coriolis_;
            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
            return tau_d_array;
        } else {
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            q_base = q_;
            convert_to_global(cameraData);
            startApproachPointComputationPhase(cameraData.pose);
            cameraData.new_data = false;
            return maintain_base_pos();
        }
    }
   
    thread computeApproachConfig_thread_; 
    void computeApproachTrajectory(int n, Eigen::Isometry3d objectPose, Eigen::Matrix4d T_EE_0, Eigen::VectorXd q)  {
        try { 
            apResult.finished.store(false);
            log_ << time() << " started computiotions" << endl;
            auto startComp = chrono::high_resolution_clock::now();
            double intersectTime = feasibleMinTime(objectPose.translation(), CONVEYOR_BELT_SPEED, T_EE_0.topRightCorner<3,1>(), MAX_CARTESIAN_VELOCITY);
            if (intersectTime == -1) throw runtime_error("Failed to intersept the object");
            cout << (objectPose.translation() + intersectTime * CONVEYOR_BELT_SPEED).transpose() << endl;
            
            pair<vector<Eigen::VectorXd>, bool> result = generate_joint_waypoint(
                    N, 
                    intersectTime,
                    objectPose,
                    CONVEYOR_BELT_SPEED,
                    q,
                    OFFSET,
                    GRASPING_TRANSFORMATION
                    );
            vector<Eigen::VectorXd>& jointWaypoints = result.first;
            cout <<"fin pose " << jointWaypoints.back().transpose() << endl;
            if (result.second) GRASPING_TRANSFORMATION *= ALTERNATIVE_GRASPING_TRANSFORMATION; 
            auto endComp = chrono::high_resolution_clock::now();
            chrono::duration<double> computationTime = endComp - startComp;
            cout << "comp time " << computationTime.count() << endl;
	        double reachTime = intersectTime - computationTime.count();    
            double intervalNumber = n - 1;
            double deltaT = reachTime / intervalNumber;
            
            vector<Eigen::VectorXd> jointVelocities(N);
            jointVelocities[0] = Eigen::VectorXd::Zero(7);
            jointVelocities.back() = Eigen::VectorXd::Zero(7);
            for (int i = 1; i < N - 1; i ++) jointVelocities[i] = (jointWaypoints[i] - jointWaypoints[i - 1]) / deltaT;
            
            vector<Eigen::VectorXd> jointAccelerations(N);
            jointAccelerations[0] = Eigen::VectorXd::Zero(7);
            jointAccelerations.back() = Eigen::VectorXd::Zero(7);
            for (int i = 1; i < N - 1; i ++) jointAccelerations[i] = (jointVelocities[i] - jointVelocities[i - 1]) / deltaT;
            
            vector<vector<Eigen::VectorXd>> spline(intervalNumber);
            for (int i = 0; i < intervalNumber; i ++) {
                double timeP = i * deltaT;
                double timeN = (i + 1) * deltaT;
                vector<Eigen::VectorXd> localSpline(7);
                for  (int joint = 0; joint < 7; joint ++) { 
                    double prevQ = jointWaypoints[i](joint); 
                    double nextQ = jointWaypoints[i + 1](joint);
                    double prevV = jointVelocities[i](joint);
                    double nextV = jointVelocities[i + 1](joint);
                    double prevA = jointAccelerations[i](joint);
                    double nextA = jointAccelerations[i + 1](joint);
                    Eigen::Matrix<double, 6, 6> M;
                    M <<  pow(timeP, 5),     pow(timeP, 4),     pow(timeP, 3),     pow(timeP, 2),  timeP, 1.0,
                          pow(timeN, 5),     pow(timeN, 4),     pow(timeN, 3),     pow(timeN, 2),  timeN, 1.0,
                          5.0*pow(timeP, 4), 4.0*pow(timeP, 3), 3.0*pow(timeP, 2), 2.0*timeP,      1.0, 0.0,
                          5.0*pow(timeN, 4), 4.0*pow(timeN, 3), 3.0*pow(timeN, 2), 2.0*timeN,      1.0, 0.0,
                          20.0*pow(timeP, 3),12.0*pow(timeP, 2), 6.0*timeP,         2.0,         0.0, 0.0,
                          20.0*pow(timeN, 3),12.0*pow(timeN, 2),6.0*timeN,         2.0,         0.0, 0.0;
                    Eigen::Matrix<double, 6, 1> b;
                    b << prevQ, nextQ, prevV, nextV, prevA, nextA;
                    Eigen::VectorXd coeffs = M.fullPivLu().solve(b);
                    localSpline[joint] = coeffs;
                }
                spline[i] = localSpline;
            }
            
            {
                std::lock_guard<std::mutex> lock(apResult.apResultMutex);
                apResult.reachTime = reachTime;
                apResult.spline = spline;
                apResult.intervalDuration = deltaT;
                apResult.predictedObjectPose.translation() = objectPose.translation() + CONVEYOR_BELT_SPEED * intersectTime;
                apResult.predictedObjectPose.linear() = objectPose.linear();  
            }
            
            apResult.finished.store(true);
        } catch (exception const & ex ) {
            cerr << "compute_approach_point_mth catch ex: " << ex.what() << endl;
        }
    }

    void startApproachPointComputationPhase(Eigen::Isometry3d objectPose) {
        apResult.finished.store(false);
        Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        GRASPING_ORIENTATION = objectPose.linear() * GRASPING_TRANSFORMATION;
        computeApproachConfig_thread_ = std::thread(&StateController::computeApproachTrajectory, this, N, objectPose, T_EE_0, q_);
        adjustComputeThreadPriority(computeApproachConfig_thread_, false);
        state = State::ComputingApproachPoint;
    }
    
    array<double, 7> processApproachPointComputationPhase() {        
        if (apResult.finished.load()) {
            computeApproachConfig_thread_.join();
            lock_guard<std::mutex> lock(apResult.apResultMutex);
            approachParams = {time_stamp, apResult.reachTime, apResult.spline, 0, apResult.intervalDuration, apResult.predictedObjectPose};
            state = State::Approaching;
        }
        return maintain_base_pos();
    }     
    
    vector<Eigen::VectorXd> computeQ(double t, vector<Eigen::VectorXd> coeffs) {
        Eigen::VectorXd q(7);
        Eigen::VectorXd dq(7);
        Eigen::VectorXd ddq(7);
        for (int joint = 0; joint < 7; joint ++) {
            q(joint) = coeffs[joint](0) * pow(t, 5) + coeffs[joint](1) * pow(t, 4) + coeffs[joint](2) * pow(t, 3) + coeffs[joint](3) * pow(t, 2) + coeffs[joint](4) * pow(t, 1) + coeffs[joint](5);
            dq(joint) = 5 * coeffs[joint](0) * pow(t, 4) + 4 * coeffs[joint](1) * pow(t, 3) + 3 * coeffs[joint](2) * pow(t, 2) + 2 * coeffs[joint](3) * pow(t, 1) + coeffs[joint](4);
            ddq(joint) = 20 * coeffs[joint](0) * pow(t, 3) + 12 * coeffs[joint](1) * pow(t, 2) + 6 * coeffs[joint](2) * pow(t, 1) + 2 * coeffs[joint](3); 
        }
        vector<Eigen::VectorXd> result = {q, dq, ddq};
        return result;
    }

    array<double, 7> processApproachPhase() {
         double timeSinceStart = time_stamp - approachParams.startTime;
 
         if (timeSinceStart > (approachParams.intervalNumber + 1) * approachParams.intervalDuration) approachParams.intervalNumber ++;
         
         if (approachParams.intervalNumber >= N - 1) {
            q_base = q_;
            state = State::VisualServoing;
            visualServoingParams = { 
                time_stamp, 
                time_stamp, 
                approachParams.predictedObjectPose.translation(), 
                approachParams.predictedObjectPose.linear(),
                Eigen::Matrix3d::Identity(), 
                nullptr,
                new RingBuffer(500),
                VisualServoingSubState::Following,
                //VisualServoingSubState::Approaching,
                false,
                Eigen::Vector3d::Zero(),
                0,
                vector<Eigen::VectorXd>(),
                OFFSET
            };
            std::lock_guard<std::mutex> lock(cameraDataMutex);
            cameraData.new_data = false;
            return maintain_base_pos();
         }
         vector<Eigen::VectorXd> result = computeQ(timeSinceStart, approachParams.spline[approachParams.intervalNumber]);
       
         Eigen::VectorXd desiredQ = result[0];
         Eigen::VectorXd desiredDQ = result[1];
         Eigen::VectorXd desiredDDQ = result[2];
         
         Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
          
         tau_cmd = Kp * (desiredQ - q_) + Kd * (desiredDQ - dq_) + M_ * desiredDDQ + coriolis_;
         std::array<double, 7> tau_d_array{};
         Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
         return tau_d_array;
    }
    
    thread workerThread_;
    void graspObject() {
        workerFinished.store(false, std::memory_order_release);
        this->gripper_.grasp(0.005, 0.3, 7, 0.08, 0.08);
        workerFinished.store(true, std::memory_order_release);
    }

    array<double, 7> processVisualServoing() {
        Eigen::Vector3d pose;
        std::lock_guard<std::mutex> lock(cameraDataMutex);
        convert_to_global(cameraData);
        Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        if (cameraData.new_data) {
            cameraData.new_data = false;
            visualServoingParams.newData = true;
            visualServoingParams.newDataTimestamp = time_stamp;
            if (visualServoingParams.estimator == nullptr) {
                visualServoingParams.previousTimeStamp = time_stamp; 
                visualServoingParams.estimatedObjectOrientation = cameraData.pose.linear();

                Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6, 6) * 0.02;
                P.topLeftCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 0.2;
                Eigen::VectorXd kalmanFilterInitialState = Eigen::VectorXd::Zero(6);
                //kalmanFilterInitialState.head(3) = visualServoingParams.filteredObjectPosition;
                kalmanFilterInitialState.head(3) = cameraData.pose.translation();
                kalmanFilterInitialState.tail(3) = CONVEYOR_BELT_SPEED;
                visualServoingParams.estimator = new MovementEstimator(kalmanFilterInitialState, time_stamp, P);
                visualServoingParams.ringBuffer->push_back({time_stamp, kalmanFilterInitialState, P});
                visualServoingParams.previousAcceleration.setZero();
                return maintain_base_pos();
            }


            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = currentTime - cameraData.processTime;
            
            const StateEntry& pastState = visualServoingParams.ringBuffer->get_state(time_stamp - duration.count());
            visualServoingParams.estimator->set_state(pastState.state, pastState.P, time_stamp - duration.count()); 
            visualServoingParams.estimator->correct(cameraData.pose.translation());

            Eigen::Quaterniond currentObjOrientaionQ(visualServoingParams.estimatedObjectOrientation);
            Eigen::Quaterniond measuredObjOrientaionQ(cameraData.pose.linear());
            if (currentObjOrientaionQ.dot(measuredObjOrientaionQ) < 0.0) {
                measuredObjOrientaionQ.coeffs() *= -1;
            }
            visualServoingParams.estimatedObjectOrientation = currentObjOrientaionQ.slerp(0.2, measuredObjOrientaionQ).toRotationMatrix();    

            std::chrono::duration<double> test = cameraData.processTime -  start;
            Eigen::Vector3d real;
            Eigen::Vector3d eulerAngels = (cameraData.pose.linear() * GRASPING_TRANSFORMATION).eulerAngles(0, 1, 2);
            real =  cameraData.pose.translation() ;
            string real_time = to_string(test.count());
            log_vs_ << real_time << " " << real.transpose() << " " << eulerAngels.transpose() << endl;
            log_ad_ << time() << " " << real.transpose() << " " << eulerAngels.transpose()<< endl;
        }
       
        if (visualServoingParams.estimator == nullptr) return maintain_base_pos();
          
        // Kalman filter
        visualServoingParams.estimator->predict(time_stamp);
        auto [estimatedObjectData, P ] = visualServoingParams.estimator->get_state();
        visualServoingParams.ringBuffer->push_back({time_stamp, estimatedObjectData, P});
        log_kalmanfilter_ << get_unix_time_ms() << estimatedObjectData.transpose() << " "; 
        for (int i = 0;i < 3; ++i) for (int j = 0; j < 3; ++j) log_kalmanfilter_ << visualServoingParams.estimatedObjectOrientation(i,j) << " ";
        log_kalmanfilter_ << endl;
        
        
        // Rate limiter
        double MAX_RATE = 0.005; // 0.5 m / s
        Eigen::Vector3d desiredChange = estimatedObjectData.head(3) - visualServoingParams.filteredObjectPosition;
        Eigen::Vector3d saturedChange = desiredChange;
        saturedChange = saturedChange.cwiseMin(MAX_RATE);
        saturedChange = saturedChange.cwiseMax(- MAX_RATE);
        Eigen::Vector3d saturedObjectPosition = visualServoingParams.filteredObjectPosition + saturedChange; 
        
        
        //filteredObjectOrientation
        double MAX_ANGLE_RATE = 0.0015; // 1 rad / s
        Eigen::Matrix3d desiredChangeOrient = visualServoingParams.estimatedObjectOrientation * visualServoingParams.filteredObjectOrientation.transpose(); 
        Eigen::AngleAxisd aa(desiredChangeOrient);
        double angle = aa.angle();
        Eigen::Vector3d axis = aa.axis();  
        
        if (angle > MAX_ANGLE_RATE) {
            angle = MAX_ANGLE_RATE;
        }

        Eigen::AngleAxisd aa_limited(angle, axis);
        Eigen::Matrix3d R_step = aa_limited.toRotationMatrix();

        visualServoingParams.filteredObjectOrientation = R_step * visualServoingParams.filteredObjectOrientation;
        // log_vs_rp_ << time() << (visualServoingParams.filteredObjectPosition).transpose() << endl;

        // Low pass filter
        double alpha = 0.3;
        visualServoingParams.filteredObjectPosition += alpha * ( saturedObjectPosition - visualServoingParams.filteredObjectPosition);
        log_filtered_position_ << get_unix_time_ms() << visualServoingParams.filteredObjectPosition.transpose() << endl;
        Eigen::Vector3d objectVelocity = estimatedObjectData.tail(3); 
        visualServoingParams.previousTimeStamp = time_stamp;
   
        Eigen::VectorXd tauCmd;
        log_rp_ << time() << " " <<  T_EE_0.topRightCorner<3,1>().transpose()<< " " << T_EE_0.topLeftCorner<3,3>().eulerAngles(0, 1, 2).transpose() << endl;

        if (visualServoingParams.subState == VisualServoingSubState::Following){
             if (time_stamp >= visualServoingParams.startTime + 0.0) visualServoingParams.subState = VisualServoingSubState::Approaching;
        } else {
            Eigen::Vector3d desiredOffsetChange = GRASP_OFFSET - visualServoingParams.offset;
            
            Eigen::Vector3d saturedOffsetChange = desiredOffsetChange;
            saturedOffsetChange = saturedOffsetChange.cwiseMin(MAX_RATE / 20);
            saturedOffsetChange = saturedOffsetChange.cwiseMax(- MAX_RATE / 20); 
            visualServoingParams.offset += saturedOffsetChange; 
            
            if (visualServoingParams.subState == VisualServoingSubState::Approaching) {
                if (graspObject_ && (visualServoingParams.filteredObjectPosition + GRASP_OFFSET - T_EE_0.topRightCorner<3, 1>()).norm() <= sqrt(3 * pow(0.010, 2)) ) { //high speed: 0.010
                    workerFinished.store(false, std::memory_order_release);
                    workerThread_ = thread(&StateController::graspObject, this);
                    adjustComputeThreadPriority(workerThread_);
                    visualServoingParams.subState = VisualServoingSubState::Grasping;
                }
            } else {
                if (workerFinished.load(std::memory_order_acquire)) {
                    workerThread_.join();
                    state = State::Lifting;
                    Eigen::Isometry3d currentPose;
                    Eigen::Isometry3d liftingGoalPose;
                    currentPose.linear() = T_EE_0.topLeftCorner<3, 3>(); 
                    currentPose.translation() =  T_EE_0.topRightCorner<3, 1>();
                    liftingGoalPose = currentPose;
                    liftingGoalPose.translation()(2) = LIFT_HEIGHT;
                    liftingParams = {time_stamp, LiftingSubState::Lift, currentPose, liftingGoalPose};
                    return maintain_base_pos();
                }        
            }
            
        }  
       
        double predictionTimeDelta = 0.3; //high speed 0.3
        if (visualServoingParams.newData) {
            visualServoingParams.newData = false;
            visualServoingParams.coeffs = vector<Eigen::VectorXd>(3);
            Eigen::Vector3d predictedObjectPosition = visualServoingParams.filteredObjectPosition + predictionTimeDelta * objectVelocity + visualServoingParams.offset; 
            for (int i = 0; i < 3; i ++) {
                Eigen::Matrix<double, 6, 6> A;
                A <<  pow(0, 5),     pow(0, 4),     pow(0, 3),     pow(0, 2),  0, 1.0,
                    pow(predictionTimeDelta, 5),     pow(predictionTimeDelta, 4),     pow(predictionTimeDelta, 3),     pow(predictionTimeDelta, 2),  predictionTimeDelta, 1.0,
                    5.0*pow(0, 4), 4.0*pow(0, 3), 3.0*pow(0, 2), 2.0*0,      1.0, 0.0,
                    5.0*pow(predictionTimeDelta, 4), 4.0*pow(predictionTimeDelta, 3), 3.0*pow(predictionTimeDelta, 2), 2.0*predictionTimeDelta,      1.0, 0.0,
                    20.0*pow(0, 3),12.0*pow(0, 2), 6.0*0,         2.0,         0.0, 0.0,
                    20.0*pow(predictionTimeDelta, 3),12.0*pow(predictionTimeDelta, 2),6.0*predictionTimeDelta,         2.0,         0,0;
                Eigen::Matrix<double, 6, 1> B;
                B << T_EE_0.topRightCorner<3,1>()(i), predictedObjectPosition(i), (jacobian_ * dq_)(i), objectVelocity(i), visualServoingParams.previousAcceleration(i), 0;
                visualServoingParams.coeffs[i] = A.fullPivLu().solve(B);
               }
            visualServoingParams.newDataTimestamp = time_stamp;
        }

        Eigen::Vector3d controllerInputPosition;
        Eigen::Vector3d controllerInputVelocity;
        Eigen::Vector3d controllerInputAcceleration;

        if (time_stamp - visualServoingParams.newDataTimestamp > predictionTimeDelta){
            controllerInputPosition = visualServoingParams.filteredObjectPosition + visualServoingParams.offset;
            controllerInputVelocity = objectVelocity;
            controllerInputAcceleration.setZero();
            cout << "run out off pred window" << endl;
        } else {
            double nextStep = time_stamp - visualServoingParams.newDataTimestamp;
            vector<Eigen::VectorXd> c = visualServoingParams.coeffs;
            for (int i =0; i < 3; i ++) {
                controllerInputPosition(i) = c[i](0) * pow(nextStep, 5) + c[i](1) * pow(nextStep, 4) + c[i](2) * pow(nextStep, 3) + c[i](3) * pow(nextStep, 2) + c[i](4) * nextStep + c[i](5);
                controllerInputVelocity(i) = 5 * c[i](0) * pow(nextStep, 4) + 4 * c[i](1) * pow(nextStep, 3) + 3 * c[i](2) * pow(nextStep, 2) + 2 * c[i](3) * nextStep  + c[i](4);
                controllerInputAcceleration(i) = 20 * c[i](0) * pow(nextStep, 3) + 12 * c[i](1) * pow(nextStep, 2) + 6 * c[i](2) * nextStep + 2 * c[i](3);
            }
        }
 
        log_pp_ << time() << " "  <<  estimatedObjectData.head(3).transpose()  << endl;
        log_vs_rp_ << time() << " "  << (controllerInputPosition).transpose() << " " << (visualServoingParams.filteredObjectOrientation * GRASPING_TRANSFORMATION).eulerAngles(0, 1, 2).transpose() << endl;

        if(visualServoingParams.filteredObjectPosition(2) < -0.08) {
            throw std::runtime_error("filteredObjectOrientation to low: stop");
        }
        
        Eigen::Matrix3d controllerInputOrientation = visualServoingParams.filteredObjectOrientation * GRASPING_TRANSFORMATION;
        log_controller_input_ << get_unix_time_ms(); 
        for (int i = 0; i< 3; ++ i) {
            for (int j = 0; j < 3; ++ j) log_controller_input_ << controllerInputOrientation(i, j) << " ";
            log_controller_input_ << controllerInputPosition(i) << " ";
        }
        log_controller_input_ << "0 0 0 1" << endl;
        //if (T_EE_0(2,3) <= 0.1 || T_EE_0(1,3) >= -0.33) throw runtime_error("stop");
        tauCmd = cartesianController(controllerInputPosition, controllerInputOrientation, controllerInputVelocity, false,  controllerInputAcceleration);
        visualServoingParams.previousAcceleration = controllerInputAcceleration;
        tauCmd += coriolis_;
        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
        
        return tau_d_array;
    }

    void computeTransportConfig(Eigen::Matrix<double, 7, 1> q, Eigen::Matrix<double, 4, 4> desiredPose) {
        workerFinished.store(false, std::memory_order_release);
        Eigen::VectorXd config = solveInverseKinematics(q, desiredPose);
        lock_guard<mutex> lock(workerMutex);
        liftingParams.transportConfig = config;
        workerFinished.store(true, std::memory_order_release);
    }

    void openGripper() {
        workerFinished.store(false, std::memory_order_release);
        this->gripper_.move(0.08, 0.15);
        workerFinished.store(true, std::memory_order_release);
    }

    array<double, 7> processLifting() { 
        Eigen::Matrix4d T_EE_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

	double MAX_CARTESIAN_DISPLACEMENT = 0.0003;
        //double MAX_CARTESIAN_DISPLACEMENT = 0.00035; //high speed
    
	Eigen::VectorXd tauCmd;
        switch (liftingParams.subState) {
            case LiftingSubState::Lift: {                           
                if ((liftingParams.currentPose.translation() - liftingParams.liftingGoalPose.translation()).norm() <= sqrt(3 * pow(0.005, 2))) {
                    liftingParams.subState = LiftingSubState::ComputeTransport;
                    q_base = q_;
                    workerFinished.store(false, std::memory_order_release);
                    Eigen::Vector3d desiredTransportPosition = DROP_OFF_POSITION;
                    desiredTransportPosition(2) = LIFT_HEIGHT; 
                    Eigen::Isometry3d transportPose;
                    transportPose.translation() = desiredTransportPosition;
                    transportPose.linear() = liftingParams.liftingGoalPose.linear();
                    workerThread_ = thread(&StateController::computeTransportConfig, this, q_, transportPose.matrix());
                    adjustComputeThreadPriority(workerThread_);
                    lock_guard<mutex> lock(cameraDataMutex);
                    convertToEEFrame(cameraData);
                    liftingParams.actualGrasp = cameraData.pose;
                    return maintain_base_pos();
                }
                Eigen::Vector3d saturatedChange  = liftingParams.liftingGoalPose.translation() - liftingParams.currentPose.translation();
                saturatedChange = saturatedChange.cwiseMin(MAX_CARTESIAN_DISPLACEMENT);
                saturatedChange = saturatedChange.cwiseMax(- MAX_CARTESIAN_DISPLACEMENT);
                liftingParams.currentPose.translation() += saturatedChange;
                tauCmd = cartesianController(liftingParams.currentPose.translation() , liftingParams.liftingGoalPose.linear(), Eigen::Vector3d::Zero());
                tauCmd += coriolis_;
                std::array<double, 7> tau_d_array{};
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
                return tau_d_array;
            }
            case LiftingSubState::ComputeTransport: {
                if (workerFinished.load(std::memory_order_acquire)) { 
                    workerThread_.join();
                    Eigen::Vector3d desiredTransportPosition = DROP_OFF_POSITION;
                    desiredTransportPosition(2) = LIFT_HEIGHT;
                    double transportExecutionTime = (T_EE_0.topRightCorner<3, 1>() - desiredTransportPosition).norm() / 0.25; // high speed 0.4
                    double finalTime = time_stamp + transportExecutionTime; 
                    lock_guard<mutex> lock(workerMutex);
                    vector<Eigen::VectorXd> coeffs(7);
                    for (int i = 0; i < 7; i ++) {
                        Eigen::Matrix<double, 6, 6> A;
                        A <<  pow(time_stamp, 5),     pow(time_stamp, 4),     pow(time_stamp, 3),     pow(time_stamp, 2),  time_stamp, 1.0,
                            pow(finalTime, 5),     pow(finalTime, 4),     pow(finalTime, 3),     pow(finalTime, 2),  finalTime, 1.0,
                            5.0*pow(time_stamp, 4), 4.0*pow(time_stamp, 3), 3.0*pow(time_stamp, 2), 2 * time_stamp,      1.0, 0.0,
                            5.0*pow(finalTime, 4), 4.0*pow(finalTime, 3), 3.0*pow(finalTime, 2), 2 * finalTime,      1.0, 0.0,
                            20.0*pow(time_stamp, 3), 12.0*pow(time_stamp, 2), 6 * time_stamp,         2.0,         0.0, 0.0,
                            20.0*pow(finalTime, 3),12.0*pow(finalTime, 2),6.0*finalTime,         2.0,         0,0;
                        Eigen::Matrix<double, 6, 1> B;
                        B << q_(i), liftingParams.transportConfig(i), 0, 0, 0, 0;
                        coeffs[i] = A.fullPivLu().solve(B);
                     }
                     liftingParams.transportCoeffs = coeffs;
                     liftingParams.transportTime = finalTime;
                     liftingParams.subState = LiftingSubState::Transport;
                }
                return maintain_base_pos(); 
            }
            case LiftingSubState::Transport: {
                if (time_stamp <= liftingParams.transportTime){
                    vector<Eigen::VectorXd> result = computeQ(time_stamp, liftingParams.transportCoeffs);
                    Eigen::VectorXd desiredQ = result[0];
                    Eigen::VectorXd desiredDQ = result[1];
                    Eigen::VectorXd desiredDDQ = result[2];
                    tauCmd = Kp * (desiredQ - q_) + Kd * (desiredDQ - dq_) + M_ * desiredDDQ;
                    tauCmd += coriolis_;
                    std::array<double, 7> tau_d_array{};
                    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
                    return tau_d_array;
                } else {
                    q_base = q_;
                    liftingParams.subState = LiftingSubState::PlacementDown;
                    Eigen::Vector3d desiredTransportPosition = DROP_OFF_POSITION;
                    //Depends on object placement config // TODO implement that based on grasping orientation
                    desiredTransportPosition(2) = LIFT_HEIGHT;// OBJECT_BOTTOM_TO_CENTRE + liftingParams.actualGrasp.translation()(2);
                    liftingParams.liftingGoalPose.translation() = desiredTransportPosition;
                    liftingParams.currentPose.linear() = T_EE_0.topLeftCorner<3, 3>();
                    liftingParams.currentPose.translation() = T_EE_0.topRightCorner<3, 1>();
                    return maintain_base_pos();
                }
             }
             case LiftingSubState::PlacementDown: {
                if ((liftingParams.currentPose.translation() - liftingParams.liftingGoalPose.translation()).norm() <= sqrt(3 * pow(0.005, 2))) {
                    liftingParams.subState = LiftingSubState::DropOff;
                    q_base = q_;
                    workerFinished.store(false, std::memory_order_release);
                    workerThread_ = thread(&StateController::openGripper, this);
                    adjustComputeThreadPriority(workerThread_);
                    return maintain_base_pos();
                }
                Eigen::Vector3d saturatedChange  = liftingParams.liftingGoalPose.translation() - liftingParams.currentPose.translation();
                saturatedChange = saturatedChange.cwiseMin(MAX_CARTESIAN_DISPLACEMENT);
                saturatedChange = saturatedChange.cwiseMax(- MAX_CARTESIAN_DISPLACEMENT);
                liftingParams.currentPose.translation() += saturatedChange;
                tauCmd = cartesianController(liftingParams.currentPose.translation() , liftingParams.liftingGoalPose.linear(), Eigen::Vector3d::Zero());
                tauCmd += coriolis_;
                std::array<double, 7> tau_d_array{};
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
                return tau_d_array;

            }
            case LiftingSubState::DropOff: {
                if (workerFinished.load(std::memory_order_acquire)) {
                    workerThread_.join();
                    liftingParams.subState = LiftingSubState::PlacementUp;
                    q_base = q_;
                    liftingParams.liftingGoalPose.translation() = DROP_OFF_POSITION;
                    liftingParams.liftingGoalPose.translation()(2) = LIFT_HEIGHT;
                    return maintain_base_pos();
                }
                return maintain_base_pos();
            }
            case LiftingSubState::PlacementUp: {
                if ((liftingParams.currentPose.translation() - liftingParams.liftingGoalPose.translation()).norm() <= sqrt(3 * pow(0.005, 2))) {
                    state = State::MoveToStartingPoint;
                    q_base = q_;
                    moveToStartingPointParams.init = false;
                    delete visualServoingParams.estimator;
                    delete visualServoingParams.ringBuffer;

                    return maintain_base_pos();
                }
                Eigen::Vector3d saturatedChange  = liftingParams.liftingGoalPose.translation() - liftingParams.currentPose.translation();
                saturatedChange = saturatedChange.cwiseMin(MAX_CARTESIAN_DISPLACEMENT);
                saturatedChange = saturatedChange.cwiseMax(- MAX_CARTESIAN_DISPLACEMENT);
                liftingParams.currentPose.translation() += saturatedChange;
                tauCmd = cartesianController(liftingParams.currentPose.translation() , liftingParams.liftingGoalPose.linear(), Eigen::Vector3d::Zero());
                tauCmd += coriolis_;
                std::array<double, 7> tau_d_array{};
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
                return tau_d_array;
            }
        }
        tauCmd += coriolis_;
        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tauCmd;
        return tau_d_array;
    }
};    

void camera_data_receiver(zmq::context_t& ctx) {
    zmq::socket_t socket_in;
    socket_in = zmq::socket_t(ctx, zmq::socket_type::pull);
    socket_in.connect("tcp://" + BASE_IP + ":5554");
    cout <<"Reciever is connected" << endl;
    while (true) {
        zmq::message_t msg;
        zmq::message_t msg_frameNumber;
        zmq::message_t msg_timestamp;
        zmq::message_t msg_ee_transform;
        zmq::message_t msg_iteration_idx;
        zmq::message_t msg_tracking_time;

        vector<double> pose(6);
        vector<double> ee_pose(16);
        uint64_t timestamp_ns;
        int frameNumber;
	    int iterationIdxServer;
        int trackingTime;
	
       try {
   
		socket_in.recv(msg, zmq::recv_flags::none);
		socket_in.recv(msg_frameNumber, zmq::recv_flags::dontwait);
		socket_in.recv(msg_timestamp, zmq::recv_flags::dontwait);
		socket_in.recv(msg_ee_transform, zmq::recv_flags::dontwait);
		socket_in.recv(msg_iteration_idx, zmq::recv_flags::dontwait);
	    socket_in.recv(msg_tracking_time, zmq::recv_flags::dontwait);
		
		memcpy(pose.data(), msg.data(), 6 * sizeof(double));
		memcpy(ee_pose.data(), msg_ee_transform.data(), 16 * sizeof(double));
		memcpy(&frameNumber, msg_frameNumber.data(), sizeof(frameNumber));
		memcpy(&iterationIdxServer, msg_iteration_idx.data(), sizeof(iterationIdxServer));
		memcpy(&timestamp_ns, msg_timestamp.data(), sizeof(timestamp_ns));
        memcpy(&trackingTime, msg_tracking_time.data(), sizeof(int));

	} catch (const zmq::error_t& e) {
            continue;
        }
	
	    {
	        lock_guard<mutex> lock(iterationMutex);
            if (iterationIdxServer != iterationIdx) {
		        cout << "WRONG ITERATION IDX" << endl;
		        continue;
	        }
	    }
        
        Eigen::Isometry3d camera_T_tag;
        Eigen::Vector3d rotation_vector(pose[0], pose[1], pose[2]);
        camera_T_tag.linear() = Eigen::AngleAxis(
            rotation_vector.norm(), rotation_vector.normalized()
        ).toRotationMatrix();
        camera_T_tag.translation() = Eigen::Vector3d(pose[3], pose[4], pose[5]);
        Eigen::Matrix4d o_T_g_matrix = Eigen::Matrix4d::Map(ee_pose.data());

        auto start = chrono::time_point<chrono::high_resolution_clock>(chrono::nanoseconds(timestamp_ns));
        auto now = chrono::high_resolution_clock::now();

        chrono::duration<double> duration = now - start;
        Eigen::Isometry3d g_T_cad;
        g_T_cad.setIdentity();
        g_T_cad.linear() << 0, 1, 0,
                            1, 0, 0,
                            0, 0, -1;
        g_T_cad.translation() << 0, 0, -0.1034;

        Eigen::Isometry3d cad_T_cam_c;
        cad_T_cam_c.setIdentity();
        double cam_c_to_cad_angle = - 5 * M_PI / 6;
        cad_T_cam_c.linear() << 1, 0, 0,
                                0, cos(cam_c_to_cad_angle), -sin(cam_c_to_cad_angle),
                                0, sin(cam_c_to_cad_angle), cos(cam_c_to_cad_angle);
        cad_T_cam_c.translation() << 0, -0.10069, -0.01048;

        Eigen::Isometry3d cam_c_T_c;
        cam_c_T_c.setIdentity();
        cam_c_T_c.translation() << -0.02, 0, 0;

        Eigen::Isometry3d g_T_c =  g_T_cad * cad_T_cam_c * cam_c_T_c;
        Eigen::Isometry3d o_T_g(o_T_g_matrix);
        std::lock_guard<std::mutex> lock(cameraDataMutex);
        
        cameraData = { true, g_T_c * camera_T_tag, false, true, false, trackingTime, frameNumber, start, o_T_g };
 
    }
}


int main(int argc, char ** argv) {
    pthread_t this_thread = pthread_self();
    struct sched_param param;
    int policy;
    errno = pthread_getschedparam(this_thread, &policy, &param);
    if (errno) {
        perror("ERROR: pthread_getschedparam");
        throw std::runtime_error("errno != 0");
    }
    param.sched_priority = 99;

   // errno = pthread_setschedparam(this_thread, SCHED_FIFO, &param);

    // Decrease the priority of the compute thread, ensuring it doesn't go below 0
    //if (f) param.sched_priority = 1; else param.sched_priority = std::max
    errno = pthread_getschedparam(this_thread, &policy, &param);

    cout << "priority " << static_cast<int>(param.sched_priority) << endl;

    
    //Sockets set up
    cout << "ENV MODE: " << getenv("mode") << endl;
    if (string(getenv("mode")) == "5g") {
        BASE_IP = "192.168.80.1";
        cout << "using 5g"; 
    } else {
        BASE_IP = "129.97.71.51";
        cout << "using wifi";
    }

    if (string(getenv("exp_hash")) == "" || string(getenv("exp_num")) == "" || string(getenv("exp_dir")) == "")  {
        throw runtime_error("experiment hash or run number is not provided");
    }


    cout << "Remote ip: " << BASE_IP << endl;
    
    zmq::context_t ctx(1);

    zmq::socket_t socket;
    zmq::socket_t socket_control;

    socket = zmq::socket_t(ctx, zmq::socket_type::push);
    int max_backlog = 8;
    socket.setsockopt(ZMQ_SNDHWM, &max_backlog, sizeof(max_backlog));

    socket.connect("tcp://" + BASE_IP + ":5555");
    socket_control = zmq::socket_t(ctx, zmq::socket_type::req);
    socket_control.connect("tcp://" + BASE_IP + ":5553");
    cout << "Connected" << endl;
    // Robot set up
    Eigen::VectorXd q_test_init(7);
    q_test_init << 1.4784838,   0.58908088, -1.51156758, -2.32406426,  0.75959274,  2.20523463, -2.89;
    franka::Robot robot(argv[1]);
    robot.automaticErrorRecovery();
    franka::Model model = robot.loadModel();
    double time = 0.0;
    franka::RobotState initial_state = robot.readOnce();
    {
        lock_guard<mutex> lock(sharedEEPose.mtx);
        Eigen::Matrix4d initilEEPose(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
        sharedEEPose.pose = initilEEPose;
    }

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q_init(initial_state.q.data());
    franka::Gripper gripper(argv[1]);
    bool grasp = false;
    if (argc == 3) {
        if (string(argv[2]) == "true") grasp = true;
    }
    StateController controller(model, gripper, socket_control, q_init, grasp, string(getenv("exp_hash")), stoi(getenv("exp_num")), string(getenv("exp_dir")));  
    if(getenv("GRIPPER_HOMING")) {
        gripper.homing();
    } else {
        gripper.move(0.08, 0.15);
    }
    cout << "robot is set up" << endl; 
    //Camera set up
    rs2::pipeline pipe;
    rs2::config cfg;
    const int fps = 30;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, fps);
    cfg.enable_device("123622270300");
    
    int frameNumberG = 0;
    auto start = chrono::high_resolution_clock::now();

    auto realsense_callback = [&socket, &frameNumberG, &start](const rs2::frame& frame) {   
            rs2::frameset fs = frame.as<rs2::frameset>();
            if(!fs) return;
            rs2::video_frame cur_frame = fs.get_color_frame();
            rs2::depth_frame dframe = fs.get_depth_frame();
            if(!cur_frame || !dframe) return;
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
                
            chrono::duration<double> duration = now - start;
            zmq::message_t msgFrameNumber(sizeof(frameNumberG));
            std::memcpy(msgFrameNumber.data(), &frameNumberG, sizeof(frameNumberG));
            
            zmq::message_t msgIterationIdx(sizeof(int));

            {
                lock_guard<mutex> lock(iterationMutex); 
                std::memcpy(msgIterationIdx.data(), &iterationIdx, sizeof(iterationIdx));
                //cout << "sender-> " << frameNumberG <<  " " << iterationIdx << endl;
            }
            vector<uchar> compressed_cframe;
            cv::imencode(".jpg", color_data, compressed_cframe);

            vector<uchar> compressed_dframe;
	        std::vector<int> compressionParams = { cv::IMWRITE_PNG_COMPRESSION, 3 };
            cv::imencode(".png", depth_data, compressed_dframe, compressionParams);            
           
            zmq::message_t msg_color(compressed_cframe.size());
            memcpy(msg_color.data(), compressed_cframe.data(), compressed_cframe.size());

            zmq::message_t msg_depth(compressed_dframe.size());
            memcpy(msg_depth.data(), compressed_dframe.data(), compressed_dframe.size());
             
            vector<double> flattenedData;
            {
                lock_guard<mutex> lock(sharedEEPose.mtx);
                flattenedData = vector<double>(sharedEEPose.pose.data(), sharedEEPose.pose.data() + sharedEEPose.pose.size());
            }
            zmq::message_t msg_ee_pose(sizeof(double) * flattenedData.size());
            memcpy(msg_ee_pose.data(), flattenedData.data(), flattenedData.size() * sizeof(double));

            auto now_zmq_start = chrono::high_resolution_clock::now();
            std::optional<long int> res;// | zmq::send_flags::dontwait
            res = socket.send(msg_color, zmq::send_flags::sndmore);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            res = socket.send(msg_depth, zmq::send_flags::sndmore);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            res = socket.send(msgFrameNumber, zmq::send_flags::sndmore);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            res = socket.send(msg_timestamp, zmq::send_flags::sndmore);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            res = socket.send(msg_ee_pose, zmq::send_flags::sndmore);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            res = socket.send(msgIterationIdx, zmq::send_flags::none);
            if(res.value()==-1 && errno == EAGAIN)std::cout << "hwm hit" << std::endl;
            auto delta_zmq = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - now_zmq_start).count();
            auto delta_zmq_ns = static_cast<uint64_t>(delta_zmq);
            if(delta_zmq_ns > 100000) {
                std::cout << "delta_zmq_ns: " << delta_zmq_ns << std::endl;
            }
            
            frameNumberG ++;
    };
   
    rs2::pipeline_profile profile = pipe.start(cfg, realsense_callback);
    cout << "camera is set up" << endl;
    auto intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    depth_sensor.set_option(RS2_OPTION_DEPTH_UNITS, 0.001f);
    float current_depth_unit = depth_sensor.get_option(RS2_OPTION_DEPTH_UNITS);
    //Start reciever thread
    thread camera_data_receiver_worker(camera_data_receiver, ref(ctx));
    //adjustComputeThreadPriority(camera_data_receiver_worker);    
    controller.set_q_base(q_init); 
    auto control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
                   std::array<double, 7> tau_d_array{};

        try {
            time += period.toSec();

            tau_d_array = controller.update(time, robot_state);   
       }
       catch (const std::exception& e) {
            std::cerr << "Exception in control loop: " << e.what() << std::endl;
        throw;  // rethrow so libfranka can handle shutdown
        }
        return tau_d_array;
    };
    robot.control(control_callback);


    return 0;
}
    
