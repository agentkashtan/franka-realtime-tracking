#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Dense>
#include <utility>
#include <mutex>

class MovementEstimator {
public:
    // Constructor
    MovementEstimator(Eigen::VectorXd init_position, double init_time, Eigen::MatrixXd init_P);

    // Predict function declaration
    std::pair<Eigen::VectorXd, double> get_state();
    void predict(double t);
    void correct(Eigen::Vector3d measurement);

private:
    Eigen::MatrixXd H;  // Measurement model matrix
    Eigen::MatrixXd P;  // Previous covariance matrix
    Eigen::MatrixXd R;  // Measurement noise covariance matrix
    Eigen::MatrixXd Q;  // Process noise covariance matrix
    Eigen::VectorXd state_estimate;  // Previous state estimate
    double prev_t;  // Previous time step
    std::mutex mutex_;
};

#endif // KALMAN_H
