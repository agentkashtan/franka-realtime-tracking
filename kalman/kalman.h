#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Dense>
#include <utility>
#include <mutex>

class MovementEstimator {
public:
    // Constructor
    MovementEstimator(Eigen::VectorXd init_state, double init_time, Eigen::MatrixXd init_P);

    // Predict function declaration
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> get_state();
    void set_state(Eigen::VectorXd newState, Eigen::MatrixXd newP, double newTimestamp);
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
