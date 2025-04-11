#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
 
using namespace casadi;
 
//Compute time for EE to intersect with object, when moving in a straight line with exceeding maxCartesianVelocity. If -1, there is no valid time
double feasibleMinTime(Eigen::Vector3d  objectPosition, Eigen::Vector3d objectVelocity, Eigen::Vector3d robotPosition, double maxCartesianVelocity) {
    Eigen::Vector3d delta = objectPosition - robotPosition;
    double C = delta.dot(delta);
    double B = 2 * delta.dot(objectVelocity);
    double A = objectVelocity.dot(objectVelocity) - std::pow(maxCartesianVelocity, 2);
    double e = 1e-12;
    if (std::abs(A) < e) {
        if (std::abs(B) < e) {
            return -1;
        }
        if (B > 0) {
            double t = - C / B;
            if (t >= 0) return t;
            return -1;
        } else {
            return std::max(0.0, - C / B);
        }
    }
    double D = std::pow(B, 2) - 4 * A * C;
    if (D < 0) {
        if (A < 0) return 0;
        return -1;
    } else {
        double sqrtD = std::sqrt(D);
        double x1 = (- B - sqrtD) / (2 * A);
        double x2 = (- B + sqrtD) / (2 * A);
        if (x1 > x2) std::swap(x1, x2);
        if (A > 0) {
            if (x2 < 0) return -1;
            return std::max(0.0, x1);
        } else { 
            if (x2 < 0) {
                if (C <= 0) return 0;
                return -1;
            }
            if (x1 < 0) return x2;
            if (C <= 0) return 0;
            return -1;
        }
    }
}




/// DH link transform
SX link_transform(double a, double alpha, double d, const SX &theta)
{
    SX cth = cos(theta);
    SX sth = sin(theta);
 
    double cal = std::cos(alpha);
    double sal = std::sin(alpha);
 
    SX T = SX::zeros(4,4);
 
    T(0,0) = cth;
    T(0,1) = -sth;
    T(0,2) = 0.0;
    T(0,3) = a;
 
    T(1,0) = sth*cal;
    T(1,1) = cth*cal;
    T(1,2) = -sal;
    T(1,3) = -d*sal;
 
    T(2,0) = sth*sal;
    T(2,1) = cth*sal;
    T(2,2) = cal;
    T(2,3) = d*cal;
 
    T(3,0) = 0.0;
    T(3,1) = 0.0;
    T(3,2) = 0.0;
    T(3,3) = 1.0;
 
    return T;
}
 
// Orientation error via Angle-Axis
SX orientationErrorAngleAxis(const SX &R_current, const SX &R_des)
{
    SX R_err = mtimes(transpose(R_des), R_current);
    SX trace_val = R_err(0,0) + R_err(1,1) + R_err(2,2);
    SX cos_phi = 0.5 * (trace_val - 1.0);
 
    // clamp cos_phi to [-1, 1]
    cos_phi = fmax(fmin(cos_phi, 1.0), -1.0);
    SX phi = acos(cos_phi);
    SX sin_phi = sqrt(1.0 - cos_phi*cos_phi);
 
    SX r21 = R_err(2,1) - R_err(1,2);
    SX r02 = R_err(0,2) - R_err(2,0);
    SX r10 = R_err(1,0) - R_err(0,1);
 
    const double eps = 1e-8;
    SX denom = 2.0 * sin_phi;
    SX denom_safe = if_else(fabs(denom) < eps, eps, denom);
 
    SX axis_raw = vertcat(r21, r02, r10) / denom_safe;
 
    // If phi is tiny, set axis to zero
    SX axis = if_else(phi < eps, vertcat(SX(0.0), SX(0.0), SX(0.0)), axis_raw);
 
    return phi * axis;
}
 
std::pair<SX,SX> forward_kinematics(const SX &q)
{
    SX T = SX::eye(4);
 
    T = mtimes(T, link_transform(0.0,       0.0,        0.333, q(0)));
    T = mtimes(T, link_transform(0.0,      -M_PI/2,     0.0,   q(1)));
    T = mtimes(T, link_transform(0.0,       M_PI/2,     0.316, q(2)));
    T = mtimes(T, link_transform(0.0825,    M_PI/2,     0.0,   q(3)));
    T = mtimes(T, link_transform(-0.0825,  -M_PI/2,     0.384, q(4)));
    T = mtimes(T, link_transform(0.0,       M_PI/2,     0.0,   q(5)));
    T = mtimes(T, link_transform(0.088,     M_PI/2,     0.0,   q(6)));
    T = mtimes(T, link_transform(0.0,       0.0,        0.107, -M_PI/4));
    T = mtimes(T, link_transform(0.0,       0.0,        0.1034, 0.0));
 
    SX pos = T(Slice(0,3), Slice(3,4));
    SX R   = T(Slice(0,3), Slice(0,3));
    return std::make_pair(pos, R);
}

double computeScore(Eigen::VectorXd q, int ndof, std::vector<double>& upperJointsLimits, std::vector<double>& lowerJointsLimits) {
    double margin = 10000000;
    for (int i = 0; i < ndof; i ++) {
        margin = std::min(margin, std::min(q(i) - lowerJointsLimits[i], upperJointsLimits[i] - q(i)) / (upperJointsLimits[i] - lowerJointsLimits[i]));
    }
    std::cout << "Score: " << margin << std::endl;
    return margin;
}

std::pair<std::vector<Eigen::VectorXd>, bool> generate_joint_waypoint(
        int N,
        double total_time,
        Eigen::Isometry3d obj_init_pose,
        Eigen::Vector3d obj_vel,
        Eigen::VectorXd robot_joint_config,
        Eigen::Vector3d offset,
        Eigen::Matrix3d graspingTransformation
        )
{ 
    Eigen::AngleAxisd rotZPI(M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d matirxRotZPI = rotZPI.toRotationMatrix();
    std::vector<Eigen::Matrix3d> symmetricGripperTransform = { Eigen::Matrix3d::Identity(), matirxRotZPI };
    std::vector<std::vector<Eigen::VectorXd>> result(2, std::vector<Eigen::VectorXd>(N));
    int ndof = 7;
    std::vector<double> upperJointsLimits = {2.8973,  1.7628,  2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
    std::vector<double> lowerJointsLimits = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
    std::vector<double> upperVelocityLimits = {2, 2, 2, 2, 2, 2, 2};
    std::vector<double> lowerVelocityLimits = {-2, -2, -2, -2, -2, -2, -2};

    for (int ind = 0; ind < 2; ind ++) {  
    // -------------------------------------------------------------------------
    Eigen::Vector3d obj_init_pos = obj_init_pose.translation();
    SX robotJointPositionSX = SX::zeros(7);
    for (int i = 0; i < ndof; i ++) robotJointPositionSX(i) = robot_joint_config(i);

    // -------------------------------------------------------------------------
    Eigen::Vector3d robot_pos_fin = obj_init_pos + obj_vel * total_time + offset;
 
    Eigen::Matrix3d orientationFinalEigen = obj_init_pose.linear() * graspingTransformation * symmetricGripperTransform[ind];
    DM orient_fin = DM::zeros(3 ,3);
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            orient_fin(i, j) = orientationFinalEigen(i, j);
        }
    }
  
    double delta_t = total_time / (N - 1);

    SX q_sym = SX::sym("q", ndof * N * 2);

    auto get_q = [&](int i)
    {
        SX out = SX::zeros(ndof);
        for (int j = 0; j < ndof; ++j) {
            out(j) = q_sym(i*ndof + j);
        }
        return out;
    };

    auto get_dq = [&](int i)
    {
        SX out = SX::zeros(ndof);
        for (int j = 0; j < ndof; ++j) {
            out(j) = q_sym(N * ndof + i*ndof + j);
        }
        return out;
    };

    std::vector<Eigen::Vector3d> intermediate_obj_pos(N);
    for (int i = 0; i < N; i ++){
        intermediate_obj_pos[i] = obj_init_pos + i * delta_t * obj_vel;
    }
  
    SX obj = SX::zeros(1);
 
    std::vector<SX> g_list;
    std::vector<double> lbg_list, ubg_list;
    
    for (int i=0; i<N; i++)
    {
        SX q_i = get_q(i);
        SX dq_i = get_dq(i);
        auto fwd_ee = forward_kinematics(q_i);
        SX pos_ee = fwd_ee.first;
        SX R_ee   = fwd_ee.second;
 
        if (i == 0)
        {
            SX err = q_i - robotJointPositionSX;
            obj += 10 * dot(err, err);
            obj += 1.5 * dot(dq_i, dq_i);
        }
        else if (i == N-1)
        {
            SX pos_err = pos_ee - vertcat(SX(robot_pos_fin(0)), SX(robot_pos_fin(1)), SX(robot_pos_fin(2)));
            obj += 50* dot(pos_err, pos_err);
            SX err = orientationErrorAngleAxis(R_ee, orient_fin);   
            obj += 50 * dot(err, err);
            obj += 1.5 * dot(dq_i, dq_i);
        }
        else
        {
            SX z_vec = vertcat(SX(intermediate_obj_pos[i](0)), SX(intermediate_obj_pos[i](1)), SX(intermediate_obj_pos[i](2)))  - pos_ee;
            SX z_norm_ee = sqrt(dot(z_vec, z_vec));
            SX z_unit = z_vec / z_norm_ee;
 
            SX err = 1 - dot(R_ee(Slice(),2), z_unit);
            obj += 0.5 * dot(err, err);
            obj += 0.2 * dot(dq_i, dq_i);

        }
 
        if (i > 0)
        {
            SX q_prev = get_q(i - 1);
            SX dq_prev = get_dq(i - 1);
            SX error = q_i - q_prev - dq_prev * delta_t;
            obj += 50 * dot(error, error);
            
            SX ddq = dq_i - dq_prev;
            obj += 0.2 * dot(ddq, ddq);
        }
    }
 
    // -------------------------------------------------------------------------
    SX g;
    if (!g_list.empty())
    {
        g = vertcat(g_list);
    }
    else
    {
        g = SX::zeros(1);
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
    }
    DM lbg = DM(lbg_list);
    DM ubg = DM(ubg_list);
 
    // -------------------------------------------------------------------------
    std::vector<double> lbq, ubq;
    lbq.reserve(2*N*ndof);
    ubq.reserve(2*N*ndof);
    for (int i=0; i<N; i++){
        for (int j=0; j<ndof; j++){
            lbq.push_back(lowerJointsLimits[j]);
            ubq.push_back(upperJointsLimits[j]);
        }
    }
    for (int i = 0; i < N; i ++){
        for (int j=0; j<ndof; j++){
            lbq.push_back(lowerVelocityLimits[j]);
            ubq.push_back(upperVelocityLimits[j]);
        }
    }

    // -------------------------------------------------------------------------
    std::vector<double> x0(2 * N * ndof, 0.2);
    SXDict nlp;
    nlp["x"] = q_sym;
    nlp["f"] = obj;
    nlp["g"] = g;
    Dict opts_dict=Dict();
    opts_dict["ipopt.print_level"] = 0;
    opts_dict["ipopt.sb"] = "yes";
    opts_dict["print_time"] = 0;

    Function solver = nlpsol("solver", "ipopt", nlp, opts_dict);
 
    // -------------------------------------------------------------------------
    std::map<std::string, DM> arg, res;
    arg["lbx"] = lbq;
    arg["ubx"] = ubq;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["x0"]  = x0;
 
    res = solver(arg);

    std::vector<double> q_opt = std::vector<double>(res["x"]);
    for (int i = 0; i < N; i ++){
        Eigen::Map<Eigen::VectorXd> vec(&q_opt[i * ndof], ndof);
        result[ind][i] = vec;
    }
    }
    return computeScore(result[0].back(), ndof, upperJointsLimits, lowerJointsLimits) > computeScore(result[1].back(), ndof, upperJointsLimits, lowerJointsLimits) ? std::make_pair(result[0], false) : std::make_pair(result[1], true);
}


Eigen::VectorXd solveInverseKinematics(
        Eigen::Matrix<double, 7, 1> initialJointConfiguration,
        Eigen::Matrix<double, 4, 4> desiredPose
        )
{ 
    int ndof = 7;
    SX q_sym = SX::sym("q", ndof * 2);
    auto get_q = [&](int i)
    {
        SX out = SX::zeros(ndof);
        for (int j = 0; j < ndof; ++j) {
            out(j) = q_sym(i*ndof + j);
        }
        return out;
    };
       
    SX qInitial = get_q(0);
    SX qFinal = get_q(1);

    SX obj = SX::zeros(1);
    SX difference = qInitial - qFinal;
    obj = dot(difference, difference); 

    std::vector<SX> g_list;
    std::vector<double> lbg_list, ubg_list;

    for (int i = 0; i < ndof; i ++) {
        g_list.push_back(qInitial(i) - initialJointConfiguration(i));
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
    }
   
   auto endEffectorPose = forward_kinematics(qFinal);
   SX endEffectorPosition = endEffectorPose.first;
   SX endEffectorOrientation = endEffectorPose.second;
   SX orientationDesiredCA = SX::zeros(3,3);
   for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            orientationDesiredCA(i, j) = desiredPose(i, j);
        }
   }
   SX angleAxisErr = orientationErrorAngleAxis(endEffectorOrientation, orientationDesiredCA);
   for (int i = 0; i < 3; i ++) {
        g_list.push_back(angleAxisErr(i));
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
   }
    
   for (int i = 0; i < 3; i ++) {
        g_list.push_back(endEffectorPosition(i) - desiredPose(i, 3));
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
   }

   std::vector<double> upperJointsLimits = {2.8973,  1.7628,  2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
   std::vector<double> lowerJointsLimits = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
   
   SX g;
   if (!g_list.empty())
   {
        g = vertcat(g_list);
   }
   else
   {
        g = SX::zeros(1);
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
   }
   DM lbg = DM(lbg_list);
   DM ubg = DM(ubg_list);
 
   std::vector<double> lbq, ubq;
   lbq.reserve(2 * ndof);
   ubq.reserve(2 * ndof);
   for (int i = 0; i < 2; i ++){
       for (int j = 0; j < ndof; j ++){
            lbq.push_back(lowerJointsLimits[j]);
            ubq.push_back(upperJointsLimits[j]);
        }
   }

   std::vector<double> x0(2 * ndof);
   for (int i = 0; i < ndof; i ++) {
        x0[i] = initialJointConfiguration(i);
        x0[i + ndof] = initialJointConfiguration(i);
   }
   SXDict nlp;
   nlp["x"] = q_sym;
   nlp["f"] = obj;
   nlp["g"] = g;
   Dict opts_dict=Dict();
   opts_dict["ipopt.print_level"] = 0;
   opts_dict["ipopt.sb"] = "yes";
   opts_dict["print_time"] = 0;
   Function solver = nlpsol("solver", "ipopt", nlp, opts_dict);
 
   std::map<std::string, DM> arg, res;
   arg["lbx"] = lbq;
   arg["ubx"] = ubq;
   arg["lbg"] = lbg;
   arg["ubg"] = ubg;
   arg["x0"]  = x0;
 
   res = solver(arg);
   std::vector<double> q_opt = std::vector<double>(res["x"]);
   Eigen::VectorXd result(7);
   for (int i = 0; i < ndof; i ++) result(i) = q_opt[ndof + i];
   return result;
}

