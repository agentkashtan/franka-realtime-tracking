#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Dense>
#include <utility>

class MovementEstimator {
public:
    // Constructor
    MovementEstimator(Eigen::VectorXd init_position, double init_time, Eigen::MatrixXd init_P);

    // Predict function declaration
    std::pair<Eigen::Vector3d, Eigen::Vector3d> predict(Eigen::Vector3d measurement, double t);

private:
    Eigen::MatrixXd H;
    Eigen::MatrixXd prev_P;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Q;
    Eigen::VectorXd prev_estimate;
    double prev_t;
};

#endif // KALMAN_H
