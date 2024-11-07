#include "kalman.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

class MovementEstimator {

public:	
	MovementEstimator(Eigen::Vector3d init_position, double init_time, Eigen::MatrixXd init_P) {
		H = Eigen::MatrixXd::Zero(3 ,6);
		H.leftCols(3) = Eigen::MatrixXd::Identiry(3, 3);
		Q = Eigen::MatrixXd::Zero(6 ,6);
		R = Eigen::Matrix:Xd:Zero(3, 3);
		prev_P = P;
		prev_t = init_time;
		prev_estimate = init_posistion;
	
	}
	std::pair<Eigen::Vector3d, Eigen::Vector3d> predict(Eigen::Vector3d measurement, double t) {
		Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6, 6);
		A.topRightCorner(3, 3) = Eigen::MatrixXd::Identity(3, 3) * (t - prev_t);
		prev_t = t;
		
		Eigen::VectorXd predicted_state = A * prev_estimate;
		Eigen::MatrixXd predicted_P = A * prev_P * A.transpose() + Q;
		Eigen::MatrixXd kalman_gain = predcited_p * H.transpose() * (H * predicted_P * H.transpose() + R).inverse();
	
		Eigen::VectorXd estimate = predicted_estimate + kalman_gain * (measurement - H * predicted_state);
		prev_estimate = estimate;
		prev_P = predicted_P - kalman_gain * H * predicted_P;

		return {esimate.head(3), estimate.tail(3)};	
			
	}
private:
	Eigen::MatrixXd H;
	Eigen::MatrixXd prev_P;
	Eigen::MatrixXd R;
	Eigen::MatrixXd Q;
	Eigen::VectorXd prev_estimate;
	double prev_t;
};
