#ifndef TRAJECTORY_GENERATOR_H
#define TRAJECTORY_GENERATOR_H

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
 
using namespace casadi;

double feasibleMinTime(Eigen::Vector3d  objectPosition, Eigen::Vector3d objectVelocity, Eigen::Vector3d robotPosition, double maxCartesianVelocity);

/// DH link transform
SX link_transform(double a, double alpha, double d, const SX &theta);
 
//------------------------------------------------------------------------------
 
/// Orientation error via Angle-Axis
SX orientationErrorAngleAxis(const SX &R_current, const SX &R_des);
 
//------------------------------------------------------------------------------
 
/// Forward kinematics for the described manipulator
std::pair<SX,SX> forward_kinematics(const SX &q);

std::vector<Eigen::VectorXd> generate_joint_waypoint(
        int N,
        double total_time,
        Eigen::Isometry3d obj_init_pose,
        Eigen::Vector3d obj_vel,
        Eigen::VectorXd robot_joint_config,
        Eigen::Vector3d offset
        );

#endif //  TRAJECTORY_GENERATOR_H

