#include "kalman.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

MovementEstimator::MovementEstimator(Eigen::VectorXd init_position, double init_time, Eigen::MatrixXd init_P) {
    H = Eigen::MatrixXd::Zero(3 ,6);
    H.leftCols(3) = Eigen::MatrixXd::Identity(3, 3);  
    Q = Eigen::MatrixXd::Identity(6 ,6) * 1;
    Q.topLeftCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;  
    
    R = Eigen::MatrixXd::Identity(3, 3) * 1;  
    prev_P = init_P;
    prev_t = init_time;
    prev_estimate = Eigen::VectorXd::Zero(6);
    prev_estimate.head(3) = init_position;
}

void MovementEstimator::correct(Eigen::Vector3d measurement) {
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::MatrixXd kalman_gain = P * H.transpose() * (H * P * H.transpose() + R).inverse(); 
    predicted_state += kalman_gain * (measurement - H * predicted_state);  
    P = P - kalman_gain * H * predicted_P;  
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> MovementEstimator::get_state() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {estimate.head(3), estimate.tail(3)};
}

void MovementEstimator::predict(double t) {
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6, 6);
    A.topRightCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * (t - prev_t);
    prev_t = t;
    
    predicted_state = A * predicted_state;
    
    P = A * prev_P * A.transpose() + Q;
}
