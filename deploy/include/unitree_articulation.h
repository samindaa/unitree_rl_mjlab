// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include "sources/odom_source.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_), odom(unitree_rl::odom_source())
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        if (odom) {
            unitree_rl::msgs::OdomSample s;
            if (odom->sample(s)) {
                data.root_pos_w = Eigen::Vector3f(s.pos[0], s.pos[1], s.pos[2]);
                data.root_lin_vel_b = Eigen::Vector3f(s.lin_vel_b[0], s.lin_vel_b[1], s.lin_vel_b[2]);
                data.has_odom = true;
            }
            data.odom_age_ms = static_cast<float>(odom->age_ms());
        }
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions and velocities
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
    }

    LowStatePtr lowstate;
    std::shared_ptr<unitree_rl::OdomSource> odom;  // optional state estimate
};

}