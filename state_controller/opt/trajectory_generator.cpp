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
 
    // alpha is a constant in each call, so use std::cos/std::sin
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
 
//------------------------------------------------------------------------------
 
/// Orientation error via Angle-Axis
SX orientationErrorAngleAxis(const SX &R_current, const SX &R_des)
{
    SX R_err = mtimes(transpose(R_des), R_current);  // R_des^T * R_current
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
 
//------------------------------------------------------------------------------
 
/// Forward kinematics for the described manipulator
std::pair<SX,SX> forward_kinematics(const SX &q)
{
    SX T = SX::eye(4);
 
    // link_transform( a,      alpha,      d,       q[i] )
    T = mtimes(T, link_transform(0.0,       0.0,        0.333, q(0)));
    T = mtimes(T, link_transform(0.0,      -M_PI/2,     0.0,   q(1)));
    T = mtimes(T, link_transform(0.0,       M_PI/2,     0.316, q(2)));
    T = mtimes(T, link_transform(0.0825,    M_PI/2,     0.0,   q(3)));
    T = mtimes(T, link_transform(-0.0825,  -M_PI/2,     0.384, q(4)));
    T = mtimes(T, link_transform(0.0,       M_PI/2,     0.0,   q(5)));
    T = mtimes(T, link_transform(0.088,     M_PI/2,     0.0,   q(6)));
    T = mtimes(T, link_transform(0.0,       0.0,        0.107, -M_PI/4));
    T = mtimes(T, link_transform(0.0,       0.0,        0.1034, 0.0));
 
    SX pos = T(Slice(0,3), Slice(3,4));      // top-left 3x1
    SX R   = T(Slice(0,3), Slice(0,3));      // top-left 3x3
    return std::make_pair(pos, R);
}

std::vector<Eigen::VectorXd> generate_joint_waypoint(
        int N,
        double total_time,
        Eigen::Isometry3d obj_init_pose,
        Eigen::Vector3d obj_vel,
        Eigen::VectorXd robot_joint_config,
        Eigen::Vector3d offset
        )
{ 
    // -------------------------------------------------------------------------
    // 2. Compute initial orientation
    int ndof = 7;

    Eigen::Vector3d obj_init_pos = obj_init_pose.translation();
    
    SX robotJointPositionSX = SX::zeros(7);
    for (int i = 0; i < ndof; i ++) robotJointPositionSX(i) = robot_joint_config(i);

    // -------------------------------------------------------------------------
    // 4. Compute final pose
    Eigen::Vector3d robot_pos_fin = obj_init_pos + obj_vel * total_time + offset;
 
    // We'll define some axis to orient the end-effector at final
    // (From your snippet: x_fin = [0,1,0], z_fin = - offset / ||offset||, etc.)
    /*
    Eigen::Vector3d x_fin(0.0, 1.0, 0.0);
 
    // z_fin = -offset / norm(offset)
    
    Eigen::Vector3d z_fin = - offset / offset.norm();
  
    // y_fin = cross(z_fin,x_fin)
    Eigen::Vector3d y_fin = z_fin.cross(x_fin);
    DM orient_fin = DM::zeros(3 ,3);
    for (int i = 0; i < 3; i ++) {
        orient_fin(0, i) = x_fin(i);
        orient_fin(1, i) = y_fin(i);
        orient_fin(2, i) = z_fin(i);
    }
    */	

    double delta_t = total_time / (N - 1);
 
    // Create CasADi decision variable: q_sym in R^(7*N)
    SX q_sym = SX::sym("q", ndof * N * 2);
 
    // Helper to extract q_i (7x1) from the flattened q_sym
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

 
    // We define some weights
    double w_pos_terminal = 3.0;
    double w_ori_terminal = 2.0;
    double w_ori          = 1.0;
    double w_smooth       = 0.2;
 
    // Precompute the object position at each time step
    std::vector<Eigen::Vector3d> intermediate_obj_pos(N);
    for (int i = 0; i < N; i ++){
        intermediate_obj_pos[i] = obj_init_pos + i * delta_t * obj_vel;
    }
    std::cout << "obj" << intermediate_obj_pos[19].transpose() << std::endl; 
    // Build objective
    SX obj = SX::zeros(1);
 
    // (We won't add constraints in g_list for this snippet, but you could.)
    std::vector<SX> g_list;
    std::vector<double> lbg_list, ubg_list;
 
    // Joint limits
    std::vector<double> upperJointsLimits = {2.8973,  1.7628,  2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
    std::vector<double> lowerJointsLimits = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
    std::vector<double> upperVelocityLimits = {2, 2, 2, 2, 2, 2, 2};
    std::vector<double> lowerVelocityLimits = {-2, -2, -2, -2, -2, -2, -2};

    
    
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
            // final position error
            SX pos_err = pos_ee - vertcat(SX(robot_pos_fin(0)), SX(robot_pos_fin(1)), SX(robot_pos_fin(2)));
            obj += 10* dot(pos_err, pos_err);
 
            // Instead of orientationErrorAngleAxis(R_ee, orient_fin), you used
            // the direct alignment with the direction to the object:
            SX z_vec = vertcat(SX(intermediate_obj_pos[i](0)), SX(intermediate_obj_pos[i](1)), SX(intermediate_obj_pos[i](2))) - pos_ee;
            SX z_norm_ee = sqrt(dot(z_vec, z_vec));
            SX z_unit = z_vec / z_norm_ee;
 
            SX err = 1.0 - dot(R_ee(Slice(),2), z_unit);  // 3rd column is 'z' axis
            obj += 2 * dot(err, err);
            obj += 1.5 * dot(dq_i, dq_i);
        }
        else
        {
            // intermediate orientation term
            SX z_vec = vertcat(SX(intermediate_obj_pos[i](0)), SX(intermediate_obj_pos[i](1)), SX(intermediate_obj_pos[i](2)))  - pos_ee;
            SX z_norm_ee = sqrt(dot(z_vec, z_vec));
            SX z_unit = z_vec / z_norm_ee;
 
            SX err = 1.0 - dot(R_ee(Slice(),2), z_unit);
            obj += w_ori * dot(err, err);
            obj += 0.2 * dot(dq_i, dq_i);

        }
 
        // Smoothness
        if (i > 0)
        {
            SX q_prev = get_q(i - 1);
            SX dq_prev = get_dq(i - 1);
            SX error = q_i - q_prev - dq_prev * delta_t;
            obj += 5 * dot(error, error);
            
            SX ddq = dq_i - dq_prev;
            obj += 0.2 * dot(ddq, ddq);
        }
    }
 
    // -------------------------------------------------------------------------
    // 6. Create constraints container g (dummy, if none)
    SX g;
    if (!g_list.empty())
    {
        g = vertcat(g_list);
    }
    else
    {
        // If truly no constraints, put a dummy to keep IPOPT happy
        g = SX::zeros(1);
        lbg_list.push_back(0.0);
        ubg_list.push_back(0.0);
    }
 
    // Build final lbg, ubg
    DM lbg = DM(lbg_list);
    DM ubg = DM(ubg_list);
 
    // -------------------------------------------------------------------------
    // 7. Build bounds for decision variables
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
    // 8. Initial guess
    std::vector<double> x0(2 * N * ndof, 0.2);
    /*
    for (int i = 0; i < N; i ++) x0.insert(x0.end(), robot_joint_config.data(), robot_joint_config.data() + robot_joint_config.size());
    std::vector<double> temp(N * ndof, 0);
    x0.insert(x0.end(), temp.begin(), temp.end());
    */
    // -------------------------------------------------------------------------
    // 9. Build the NLP solver
    //    nlp  = {'x': q_sym, 'f': obj, 'g': g}
    //    solver = nlpsol('solver', 'ipopt', nlp)
    SXDict nlp;
    nlp["x"] = q_sym;
    nlp["f"] = obj;
    nlp["g"] = g;
    Dict opts_dict=Dict();
    opts_dict["ipopt.print_level"] = 0;
    opts_dict["ipopt.sb"] = "yes";
    opts_dict["print_time"] = 0;

    Function solver = nlpsol("solver", "ipopt", nlp); //opts_dict);
 
    // -------------------------------------------------------------------------
    // 10. Solve
    std::map<std::string, DM> arg, res;
    arg["lbx"] = lbq;
    arg["ubx"] = ubq;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["x0"]  = x0;
 
    // Solve
    res = solver(arg);
 
    // Extract solution
    std::vector<double> q_opt = std::vector<double>(res["x"]);
 
    // q_opt is in row-major flattening.  If you want it reshaped:
    //    q_opt_mat = q_opt.reshape((ndof, N), 'F').T in Python
    // In C++, you can manually reorder if needed. For demonstration:
    // Just show first few or do your own reshaping as needed.
    std::vector<Eigen::VectorXd> result(N);
    for (int i = 0; i < N; i ++){
        Eigen::Map<Eigen::VectorXd> vec(&q_opt[i * ndof], ndof);
        result[i] = vec;
    }
    //std::cout << "result ";
    //std::cout << result[0].transpose() << std::endl << result[N-2].transpose() ;
    return result;
}

