#include "kalman.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

MovementEstimator::MovementEstimator(Eigen::VectorXd init_state, double init_time, Eigen::MatrixXd init_P) {
    H = Eigen::MatrixXd::Zero(3 ,6);
    H.leftCols(3) = Eigen::MatrixXd::Identity(3, 3);  
    Q = Eigen::MatrixXd::Identity(6 ,6) *0.001;
    Q.topLeftCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 0.000001;  
    
    R = Eigen::MatrixXd::Identity(3, 3) * 0.0004;  
    P = init_P;
    prev_t = init_time;
    state_estimate = init_state;
}

void MovementEstimator::correct(Eigen::Vector3d measurement) {
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::MatrixXd kalman_gain = P * H.transpose() * (H * P * H.transpose() + R).inverse(); 
    Eigen::VectorXd new_estimate = state_estimate  + kalman_gain * (measurement - H * state_estimate);
    Eigen::MatrixXd new_P = P - kalman_gain * H * P;
    
    state_estimate = new_estimate;
    P = new_P;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> MovementEstimator::get_state() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {state_estimate, P};
}

void MovementEstimator::predict(double t) {
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6, 6);
    A.topRightCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * (t - prev_t);
    prev_t = t;

    Eigen::VectorXd predicted_state = A * state_estimate;
    Eigen::MatrixXd predicted_P = A * P * A.transpose() + Q;

    state_estimate = predicted_state;
    P = predicted_P;
}
