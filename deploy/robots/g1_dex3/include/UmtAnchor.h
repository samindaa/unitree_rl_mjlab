#pragma once

// Shared by the UMT-based FSM states (State_UmtMimic, State_HiphiStudent):
// the robot's tracking-anchor (torso_link) pose and the clip->robot world
// alignment set when a state enters.
//
// In training every episode starts with the robot teleported onto the
// reference (RSI), so the clip's world frame and the robot's coincide. On the
// robot the clip lives in its own world frame and the IMU/estimator in
// another, so at enter() the clip is yaw-rotated (init_quat) and, when a
// state estimate is available, translated (init_pos) so that its anchor
// starts on the robot's anchor: reference world pose = init_quat * ref +
// init_pos. z stays absolute unless align_z (the estimator's z origin is the
// ground in sim; on hardware that depends on the estimator).

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <spdlog/spdlog.h>

#include "Types.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/utils/utils.h"
#include "unitree_articulation.h"

namespace umt
{

inline Eigen::Quaternionf init_quat = Eigen::Quaternionf::Identity();
inline Eigen::Vector3f init_pos = Eigen::Vector3f::Zero();

// G1 kinematic constants (mjlab g1.xml) for the anchor FK below.
inline const Eigen::Vector3f kImuInPelvisPos(0.04525f, 0.0f, -0.08339f);   // imu_in_pelvis site, pelvis frame
inline const Eigen::Vector3f kWaistRollLinkPos(-0.0039635f, 0.0f, 0.044f); // waist_roll_link in waist_yaw_link; torso_link at its origin

/// Robot torso (tracking anchor) orientation in world: pelvis IMU quaternion
/// composed with the three waist joints (yaw, roll, pitch = SDK motors 12-14).
inline Eigen::Quaternionf robot_anchor_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto & motors = robot->lowstate->msg_.motor_state();

    return root_quat
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());
}

/// Robot torso (tracking anchor) origin in world, from the streamed pelvis
/// IMU site pose and the waist yaw joint: site -> pelvis origin (fixed
/// offset) -> waist_roll_link / torso_link origin (yaw-rotated offset).
/// Valid only with a state estimate (ArticulationData::has_odom).
inline Eigen::Vector3f robot_anchor_pos_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());
    const auto & data = env->robot->data;
    const float waist_yaw = robot->lowstate->msg_.motor_state()[12].q();

    const Eigen::Vector3f pelvis_origin = data.root_pos_w - data.root_quat_w * kImuInPelvisPos;
    const Eigen::Vector3f torso_in_pelvis = Eigen::AngleAxisf(waist_yaw, Eigen::Vector3f::UnitZ()) * kWaistRollLinkPos;
    return pelvis_origin + data.root_quat_w * torso_in_pelvis;
}

/// Set init_quat / init_pos from the clip's anchor pose at its start frame
/// and the robot's current anchor pose (call with the loader on that frame).
inline void align_clip_to_robot(isaaclab::ManagerBasedRLEnv* env,
                                const Eigen::Quaternionf& ref_anchor_quat,
                                const Eigen::Vector3f& ref_anchor_pos,
                                bool align_z)
{
    const auto ref_yaw = isaaclab::yawQuaternion(ref_anchor_quat).toRotationMatrix();
    const auto robot_yaw = isaaclab::yawQuaternion(robot_anchor_quat_w(env)).toRotationMatrix();
    init_quat = Eigen::Quaternionf(robot_yaw * ref_yaw.transpose());
    init_pos.setZero();
    const Eigen::Vector3f robot_pos = robot_anchor_pos_w(env);
    if (env->robot->data.has_odom) {
        init_pos = robot_pos - init_quat * ref_anchor_pos;
        if (!align_z) init_pos.z() = 0.0f;
    }
    const Eigen::Vector3f ref_w = init_quat * ref_anchor_pos + init_pos;
    spdlog::info("align: robot anchor [{:.3f} {:.3f} {:.3f}] (odom {}), clip anchor [{:.3f} {:.3f} {:.3f}] -> world [{:.3f} {:.3f} {:.3f}], "
                 "yaw offset {:.1f} deg, init_pos [{:.3f} {:.3f} {:.3f}]",
                 robot_pos.x(), robot_pos.y(), robot_pos.z(), env->robot->data.has_odom ? "yes" : "no",
                 ref_anchor_pos.x(), ref_anchor_pos.y(), ref_anchor_pos.z(), ref_w.x(), ref_w.y(), ref_w.z(),
                 2.0f * std::atan2(init_quat.z(), init_quat.w()) * 180.0f / static_cast<float>(M_PI),
                 init_pos.x(), init_pos.y(), init_pos.z());
}

}  // namespace umt
