// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness; // sdk order
    std::vector<float> joint_damping; // sdk order

    // Joint positions of all joints.
    Eigen::VectorXf joint_pos;
    
    // Default joint positions of all joints.
    Eigen::VectorXf default_joint_pos;

    // Joint velocities of all joints.
    Eigen::VectorXf joint_vel;

    // Root angular velocity in base world frame.
    Eigen::Vector3f root_ang_vel_b;

    // Projection of the gravity direction on base frame.
    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    // State estimate (sources/odom_source.h), valid when has_odom: position
    // of the pelvis IMU site in the world and its linear velocity in the
    // site frame (mjlab robot/imu_lin_vel). Not derivable from lowstate.
    Eigen::Vector3f root_pos_w = Eigen::Vector3f::Zero();
    Eigen::Vector3f root_lin_vel_b = Eigen::Vector3f::Zero();
    bool has_odom = false;
    float odom_age_ms = 1e9f;

    std::vector<float> joint_ids_map;

    unitree::common::UnitreeJoystick* joystick = nullptr;
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};