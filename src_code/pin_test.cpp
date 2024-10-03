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


#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

using namespace std;


tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> get_q(double t, Eigen::VectorXd& q_init, Eigen::VectorXd& q_fin) {
    Eigen::VectorXd MAX_VELOCITIES(7);
    MAX_VELOCITIES << 2, 2, 2, 2, 2, 2, 2.6;
    Eigen::VectorXd MAX_ACCELERATIONS(7);
    MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;

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
    Eigen::VectorXd MAX_ACCELERATIONS(7);
    MAX_ACCELERATIONS << 15, 7.5, 10, 12.5, 15, 20, 20;
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

Eigen::VectorXd first_phase_controller(double t, Eigen::VectorXd q_init, Eigen::VectorXd q_fin, Eigen::VectorXd q_cur, Eigen::VectorXd q_dot_cur, Eigen::Matrix<double, 7,7> M) {
    Eigen::DiagonalMatrix<double, 7> Kp(200, 200, 200, 200, 200, 200,200);
    Eigen::DiagonalMatrix<double, 7> Kd(20, 20, 20, 20, 20, 20, 8);
    auto [q_des, q_dot_des, q_ddot_des] = get_q(t, q_init, q_fin);

    return Kp * (q_des - q_cur) + Kd * (q_dot_des  - q_dot_cur) + M * q_ddot_des;
}


Eigen::VectorXd visual_servoing_controller(Eigen::VectorXd q, Eigen::VectorXd q_dot, Eigen::Vector3d obj_position, Eigen::Vector3d obj_velocity, Eigen::Matrix<double, 6, 7>jacobian, Eigen::Matrix4d T_ee_0) {
    Eigen::MatrixXd w = Eigen::MatrixXd::Identity(7, 7);
    Eigen::DiagonalMatrix<double, 6> Kp_ts(300, 300, 300, 30, 30, 30);
    Eigen::DiagonalMatrix<double, 7> Kd(20, 20, 20, 20, 20, 20, 8);

    //Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 9);
    //pinocchio::computeFrameJacobian(model, data, q_temp, 26, pinocchio::WORLD, jacobian);

    //Eigen::Matrix4d T_ee_0 = data.oMf[26];
    Eigen::VectorXd obj_pos_ext(4);
    obj_pos_ext.head(3) = obj_position;
    obj_pos_ext(3) = 1;


    //obj_pos_ext = T_7_0 * T_ee_0.inverse() * obj_pos_ext;

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
    transformation(2, 3) = position.z() + 0.05; // depends on orient
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


pair<Eigen::VectorXd, double> compute_approach_point(Eigen::Vector3d obj_position, Eigen::Vector3d obj_velocity, const Eigen::VectorXd& q_init) {
    const Workspace ws = {{0.2, -0.4}, {0.6, 0.4}};
    double cur_time = time_to_reach_workspace({obj_position.x(), obj_position.y()}, {obj_velocity.x(), obj_velocity.y()}, ws);
    if (cur_time == -1) throw runtime_error("The object is out of the workspace.");
    obj_position += obj_velocity * cur_time;
    double sample_duration = 0.01;
    int iteration_num = 0;


    const string urdf_filename = "/home/bot/test_pin/urdf/panda.urdf";
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


int main(int argc, char ** argv)
{
    using namespace pinocchio;

    const string urdf_filename = "/home/bot/test_pin/urdf/panda.urdf";
    Model sim_model;
    pinocchio::urdf::buildModel(urdf_filename, sim_model);
    cout << "model name: " << sim_model.name << endl;

    Data sim_data(sim_model);
    Eigen::VectorXd q_sim = Eigen::VectorXd::Zero(sim_model.nv);
    Eigen::VectorXd q_dot_sim = Eigen::VectorXd::Zero(sim_model.nv);

    Eigen::VectorXd q_test_init(7);
    q_test_init << 1.4784838,   0.58908088, -1.51156758, -2.32406426,  0.75959274,  2.20523463, -2.89;
    Eigen::VectorXd q_test_fin(7);
    q_test_fin << -2.02504024, -1.56926117,  1.70563035, -2.63044967,  2.4754281, 1.65576549, 2.34613614;
    q_sim.head(7) = q_test_init;

    forwardKinematics(sim_model, sim_data, q_sim, q_dot_sim);
    updateFramePlacements(sim_model, sim_data);

    double dt = 0.001;


    Eigen::Vector3d obj_start_pos = Eigen::Vector3d(0.5, -0.4875, 0.1);
    Eigen::Vector3d obj_velocity = Eigen::Vector3d(0.0 , 0.135, 0.0);


    auto control_data = compute_approach_point(obj_start_pos, obj_velocity, q_sim.head(7));

    double ff_time = control_data.second;
    Eigen::VectorXd q_ff = control_data.first;


    franka::Robot robot("123.213.213.111");
    franka::Model model = robot.loadModel();
    double time = 0.0;

    auto control_callback = [&](const franka::RobotState& robot_state,
                                      franka::Duration period) -> franka::Torques {
        time += period.toSec();
        array<double, 7> coriolis_array = model.coriolis(robot_state);
        array<double, 49> mass_array = model.mass(robot_state);
        array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

        Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
        Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
        Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(7);
        if (time <= ff_time) {
            tau_cmd = first_phase_controller(dt * time, q_test_init, q_ff, q, dq, mass);
        } else if (time <= 4 ) {
            Eigen::Matrix4d T_ee_o(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            tau_cmd = visual_servoing_controller(q, dq, obj_start_pos + obj_velocity * time * dt, obj_velocity, jacobian, T_ee_o);
        }
        //Eigen::VectorXd visual_servoing_controller(Eigen::VectorXd q, Eigen::VectorXd q_dot, Eigen::Vector3d obj_position, Eigen::Vector3d obj_velocity, Eigen::Matrix<double, 6, 7>jacobian, Eigen::Matrix4d T_ee_0) {

        tau_cmd += coriolis;


        std::array<double, 7> tau_d_array{};
      return tau_d_array;
    };

    std::vector<Eigen::VectorXd> logs;
    std::vector<Eigen::VectorXd> logs_obj;

    Eigen::VectorXd q_sf_f;
    for (int t = 0; t <= 7000; t ++){
        forwardKinematics(sim_model, sim_data, q_sim, q_dot_sim);
        updateFramePlacements(sim_model, sim_data);
        computeAllTerms(sim_model, sim_data, q_sim, q_dot_sim);
        sim_data.M.triangularView<Eigen::StrictlyLower>() = sim_data.M.transpose().triangularView<Eigen::StrictlyLower>();
        Eigen::MatrixXd M = sim_data.M.topLeftCorner(7, 7);
        Eigen::MatrixXd C = sim_data.nle.head(7);


        Eigen::MatrixXd jacobian_sim = Eigen::MatrixXd::Zero(6, 9);
        pinocchio::computeFrameJacobian(sim_model, sim_data, q_sim, 26, pinocchio::WORLD, jacobian_sim);

        Eigen::Matrix4d T_ee_0_sim = sim_data.oMf[26];
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(7);
        if (t*dt <= ff_time) tau = first_phase_controller(dt * t, q_test_init, q_ff, q_sim.head(7), q_dot_sim.head(7), M);
        else if (t*dt <= 4){
            tau = visual_servoing_controller(q_sim.head(7), q_dot_sim.head(7), obj_start_pos + obj_velocity*t*dt, obj_velocity, jacobian_sim.topLeftCorner(6, 7), T_ee_0_sim);
            q_sf_f = q_sim;
        } else tau = first_phase_controller(dt * t, q_sf_f.head(7), q_test_init, q_sim.head(7), q_dot_sim.head(7), M);


        tau += C;
        Eigen::VectorXd q_ddot = M.lu().solve(- C + tau);
        Eigen::VectorXd q_ddot_full = Eigen::VectorXd::Zero(sim_model.nv);
        q_ddot_full.head(7) = q_ddot;
        q_dot_sim += q_ddot_full * dt;
        q_sim += q_dot_sim * dt;
        logs_obj.push_back(obj_start_pos + obj_velocity*t*dt);
        logs.push_back(q_sim);
    }

    freopen("output.txt", "w", stdout);
    for (auto& val: logs) {
        for (int i =0; i < val.size(); i++){
            std::cout << val[i];
            if (i < val.size() - 1) std::cout << ",";
        }
        std::cout << std::endl;
    }

    freopen("obj.txt", "w", stdout);
    for (auto& val: logs_obj) {
        for (int i =0; i < val.size(); i++){
            std::cout << val[i];
            if (i < val.size() - 1) std::cout << ",";
        }
        std::cout << std::endl;
    }

    return 0;
}
