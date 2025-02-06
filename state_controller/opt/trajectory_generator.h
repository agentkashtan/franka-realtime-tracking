#ifndef TRAJECTORY_GENERATOR_H
#define TRAJECTORY_GENERATOR_H

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
 
using namespace casadi;
 
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
        Eigen::Vector3d obj_init_pos,
        Eigen::Vector3d obj_vel,
        Eigen::Vector3d robot_pos_init,
        Eigen::Vector3d offset
        );


#endif //  TRAJECTORY_GENERATOR_H

