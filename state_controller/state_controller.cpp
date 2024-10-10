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

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
// For vector math


using namespace std;

enum class State {
    Idle,
    Observing,
    Approching,
    VisualServoing,
    ErrorRecovery,

};

struct IdleParams {
    double start_time;
};

struct ObservationParams {
    double start_time;
};

struct ApproachParams {
    double start_time;
    Eigen::VectorXd approach_config;
    double exe_time;
};

struct VisualServoingParams {
    double start_time;
    array<double, 7> tau_prev;
};

array<double, 7> ZERO_TORQUES = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


class StateController {
public:
    StateController(franka::Model& model, int maxMissingFrames, int observationWindow)
        : maxMissingFrames(maxMissingFrames),
          observationWindow(observationWindow),
          model(model),
          state(State::Idle),
          missingFrames(0)
          {
            q_base <<  1.4784838,   0.58908088, -1.51156758, -2.32406426,  0.75959274,  2.20523463, -2.89;
            Kp.diagonal() << 200, 200, 200, 200, 200, 200,200;
            Kd.diagonal()  << 20, 20, 20, 20, 20, 20, 8;
          }


    array<double, 7> update(double time, franka::RobotState robotState) {
        time_stamp = time;
        robot_state = robotState;
        switch (state) {
            case State::Idle:
                return processIdle();
                break;
            case State::Observing:
                return processObserving();
                break;
            case State::ErrorRecovery:
                return processErrorRecovery();
                break;
            case State::Approching:
                return processApproachPhase();
                break;
            case State::VisualServoing:
                return processVisualServoing();
                break;
        }
    }


private:
    int maxMissingFrames;
    int observationWindow;
    State state;
    int missingFrames;
    Eigen::VectorXd q_base;
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

    bool getPose(Eigen::Vector3d& pose) {
        pose = Eigen::Vector3d::Zero();
        return true;
    }

    array<double, 7> processIdle() {
        Eigen::Vector3d pose;

        if (getPose(pose)) {
            cout << "Object detected, switching to Observing state.\n";
            startObserving(pose);
        } else {
            cout << "Idle state, waiting for object detection.\n";
        }
        return ZERO_TORQUES;
    }

    void handleMissingFrame() {
        missingFrames++;
        if (missingFrames > maxMissingFrames) {
            cout << "Too many missing frames, transitioning to Error Recovery.\n";
            state = State::ErrorRecovery;
        }
    }

    //Handle observing state

    void startObserving(const Eigen::Vector3d& initialPosition) {
        state = State::Observing;
        observationParams.start_time = time_stamp;
        positions.clear();
        positions.push_back(initialPosition);
        missingFrames = 0;
    }

    array<double, 7> processObserving() {
        Eigen::Vector3d pose;
        if (getPose(pose)) {
            missingFrames = 0;
            positions.push_back(pose);
            if (positions.size() > observationWindow) {
                pair<Eigen::Vector3d, Eigen::Vector3d> result = fitLine(positions, time_stamp - observationParams.start_time);
                startApproachPhase(result);
            }
        } else {
            handleMissingFrame();
        }
        return ZERO_TORQUES;
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


    Eigen::VectorXd compute_ik(Eigen::VectorXd q_init, Eigen::Vector3d position, pinocchio::Model& model, pinocchio::Data& data) {
        // TODO: handle different oreinetions
        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        transformation(0, 3) = position.x();
        transformation(1, 3) = position.y();
        transformation(2, 3) = position.z() + 0.1; // depends on orient
        transformation(0,0) = -1;
        transformation(2,2) = -1;

        double max_manupuability = 0;
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

                double manupuability = eigenvalues.minCoeff() / eigenvalues.maxCoeff();
                if (manupuability > max_manupuability) {
                    best_solution = q;
                    max_manupuability = manupuability;
                }
            }
        }
        //std::cout << "max man " << max_manupuability << std::endl;
        if (max_manupuability > 0) return best_solution;
        throw runtime_error("Couldnt #2ik reach the object.");
    }


    tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> get_q(double t, Eigen::VectorXd& q_init, Eigen::VectorXd& q_fin) {
        Eigen::VectorXd MAX_VELOCITIES(7);
        MAX_VELOCITIES << 2, 2, 2, 2, 2, 2, 2.6;
        MAX_VELOCITIES *= 0.1;
        Eigen::VectorXd MAX_ACCELERATIONS(7);
        MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
        MAX_ACCELERATIONS *= 0.1;
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
        MAX_VELOCITIES *= 0.1;
        Eigen::VectorXd MAX_ACCELERATIONS(7);
        MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
        MAX_ACCELERATIONS *= 0.1;

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


    pair<Eigen::VectorXd, double> compute_approach_point(Eigen::Vector3d obj_position, Eigen::Vector3d obj_velocity, const Eigen::VectorXd& q_init) {
        const Workspace ws = {{0.2, -0.4}, {0.6, 0.4}};
        double cur_time = time_to_reach_workspace({obj_position.x(), obj_position.y()}, {obj_velocity.x(), obj_velocity.y()}, ws);
        if (cur_time == -1) throw runtime_error("The object is out of the workspace.");
        obj_position += obj_velocity * cur_time;
        double sample_duration = 0.01;
        int iteration_num = 0;


        const string urdf_filename = "../urdf/panda.urdf";
        pinocchio::Model model;
        pinocchio::urdf::buildModel(urdf_filename, model);
        pinocchio::Data data(model);

        while (in_bounds({obj_position.x(), obj_position.y()}, ws)) {

            Eigen::VectorXd q_fin = compute_ik(q_init, obj_position, model, data);
            double execution_time = completion_time(q_init, q_fin);
            if (execution_time + sample_duration < cur_time) {
                return {q_fin, execution_time};
            }
            cur_time += sample_duration;
            obj_position += obj_velocity * sample_duration;
        }
        throw runtime_error("Could not reach the object.");
    }

    Eigen::VectorXd first_phase_controller(double t, Eigen::VectorXd q_init, Eigen::VectorXd q_fin, Eigen::VectorXd q_cur, Eigen::VectorXd q_dot_cur, Eigen::Matrix<double, 7,7> M) {
        auto [q_des, q_dot_des, q_ddot_des] = get_q(t, q_init, q_fin);
        return Kp * (q_des - q_cur) + Kd * (q_dot_des  - q_dot_cur) + M * q_ddot_des;
    }

    // Handle approaching state
    void startApproachPhase(pair<Eigen::Vector3d, Eigen::Vector3d> data) {
        Eigen::Vector3d p_start = data.first;
        Eigen::Vector3d velocity = data.second;
        pair<Eigen::VectorXd, double> res = compute_approach_point(p_start, velocity, q_base);
        Eigen::VectorXd approach_config = res.first;
        double exe_time = res.second;
        state = State::Approching;
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
            tau_cmd  = first_phase_controller(time_stamp, q_base, approachParams.approach_config, q, dq, mass);
            tau_cmd += coriolis;
            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
            return tau_d_array;
        } else {
            startVisualServoing();
        }
    }

    //Handle visual servoing state

    Eigen::VectorXd visual_servoing_controller(Eigen::VectorXd q, Eigen::VectorXd q_dot, Eigen::Vector3d obj_position, Eigen::Vector3d obj_velocity, Eigen::Matrix<double, 6, 7>jacobian, Eigen::Matrix4d T_ee_0) {
        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(7, 7);
        Eigen::DiagonalMatrix<double, 6> Kp_ts(300, 300, 300, 30, 30, 30);
        Eigen::DiagonalMatrix<double, 7> Kd(20, 20, 20, 20, 20, 20, 8);

        Eigen::VectorXd obj_pos_ext(4);
        obj_pos_ext.head(3) = obj_position;
        obj_pos_ext(3) = 1;

        obj_position = obj_pos_ext.head(3);
        Eigen::Vector3d error_pos = obj_position - T_ee_0.topRightCorner<3,1>();

        Eigen::Matrix3d r_desired = Eigen::Matrix3d::Identity();
        r_desired(0,0) = -1;
        r_desired(2,2) = -1;
        Eigen::Matrix3d r_diff = r_desired * T_ee_0.topLeftCorner<3,3>().transpose();

        Eigen::AngleAxisd angle_axis(r_diff);
        Eigen::Vector3d error_orient = angle_axis.axis() * angle_axis.angle();

        Eigen::VectorXd error_p(6);
        error_p.head(3) = error_pos;
        error_p.tail(3) = error_orient;

        Eigen::VectorXd q_dot_des_ts = Eigen::VectorXd::Zero(6);
        q_dot_des_ts.head(3) = obj_velocity;
        Eigen::VectorXd q_dot_des_js = w.inverse() * jacobian.transpose() * (jacobian * w.inverse() * jacobian.transpose()).inverse() * q_dot_des_ts;

        Eigen::VectorXd tau(7);
        tau = jacobian.transpose() * Kp_ts * error_p + Kd * (q_dot_des_js - q_dot);
        return tau;
    }




    void startVisualServoing() {
        state = State::VisualServoing;
        visualServoingParams = {time_stamp,  ZERO_TORQUES};
    }

    array<double, 7> processVisualServoing() {
        Eigen::Vector3d pose;
        if (getPose(pose)) {
            missingFrames = 0;
            array<double, 7> coriolis_array = model.coriolis(robot_state);
            array<double, 49> mass_array = model.mass(robot_state);
            array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

            Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
            Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
            Eigen::Matrix4d T_ee_0(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

            Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
            //

            Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
            //
            tau_cmd = visual_servoing_controller(q, dq, pose, velocity, jacobian, T_ee_0); // get velocity somewhere ?
            tau_cmd += coriolis;
            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
            visualServoingParams.tau_prev = tau_d_array;
            return tau_d_array;

        } else {
            handleMissingFrame();
            return visualServoingParams.tau_prev;
        }

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
        return {p_start, (p_end - p_start) / time};
    }
};

int main(int argc, char ** argv) {
    using namespace pinocchio;
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

    StateController controller(model, 20, 50);
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q_init(initial_state.q.data());


    auto control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
        time += period.toSec();
        std::array<double, 7> tau_d_array{};
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
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_cmd;
        }
        else {
            tau_d_array = controller.update(time, robot_state);
        }
        return tau_d_array;

    };
    robot.control(control_callback);


    return 0;
}
