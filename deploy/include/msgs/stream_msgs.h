#pragma once

// Contract for the two inputs streamed to the deploy controller — by the
// simulator today, by robot-side nodes later — over the unitree_sdk2 DDS
// domain. Both use ROS2 types that ship precompiled in unitree_sdk2
// (unitree/idl/ros2), so neither the simulator nor the controller needs IDL
// codegen. This header is the single place the conventions are written down;
// every publisher and subscriber goes through the helpers below.
//
//   rt/odom_pelvis   nav_msgs::msg::dds_::Odometry_
//     pose  : pelvis IMU site in the world (odom) frame. The orientation is
//             stored in the ROS field order (x, y, z, w); OdomSample carries
//             it as (w, x, y, z), the MuJoCo / Eigen constructor order used
//             throughout the controller.
//     twist : linear and angular velocity of that site expressed IN THE SITE
//             FRAME (velocimeter + gyro semantics = mjlab's robot/imu_lin_vel
//             and robot/imu_ang_vel sensors on imu_in_pelvis). Not the pelvis
//             body origin: the two differ by omega x r (r ~ 0.095 m).
//     header.stamp : source clock (simulator time / estimator time).
//
//   rt/depth_camera  sensor_msgs::msg::dds_::PointCloud2_ used as an organized
//                    2D depth image (height x width, one UINT16 field
//                    "depth_mm", point_step 2, row_step 2*width, row 0 = top of
//                    the image, x increasing to the image right).
//     value : planar z-depth along the optical axis in millimetres — what a
//             D435 reports and what mujoco_warp renders in training
//             (render.py: dist * -ray_dir.z). 0 = no return. The training
//             normaliser (tasks/hiphi_tracking_multi_distill/mdp.py
//             camera_depth) is clamp(d, 0.01, cutoff) / cutoff, so a 0 lands at
//             0.01/cutoff, NOT at 1: background must be published as 0, never
//             as the far plane.

#include <unitree/idl/ros2/Odometry_.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace unitree_rl {
namespace msgs {

inline constexpr const char* kOdomTopic = "rt/odom_pelvis";
inline constexpr const char* kOdomFrameId = "odom";
inline constexpr const char* kOdomChildFrameId = "pelvis_imu";

inline constexpr const char* kDepthTopic = "rt/depth_camera";
inline constexpr const char* kDepthFrameId = "depth_camera";
inline constexpr const char* kDepthFieldName = "depth_mm";
inline constexpr uint8_t kPointFieldUint16 = 4;  // sensor_msgs/PointField UINT16

// Training camera (tasks/hiphi_tracking_multi_distill/env_cfg.py).
inline constexpr uint32_t kDepthWidth = 64;
inline constexpr uint32_t kDepthHeight = 36;

using Odometry = ::nav_msgs::msg::dds_::Odometry_;
using DepthImage = ::sensor_msgs::msg::dds_::PointCloud2_;

struct OdomSample
{
    double t = 0.0;                          // header.stamp, seconds
    double pos[3] = {0, 0, 0};               // site position, world
    double quat[4] = {1, 0, 0, 0};           // site orientation, world, (w, x, y, z)
    double lin_vel_b[3] = {0, 0, 0};         // site frame
    double ang_vel_b[3] = {0, 0, 0};         // site frame
};

inline void set_stamp(::std_msgs::msg::dds_::Header_& header, double t)
{
    double sec = std::floor(t);
    header.stamp().sec() = static_cast<int32_t>(sec);
    header.stamp().nanosec() = static_cast<uint32_t>(std::lround((t - sec) * 1e9));
}

inline double stamp_seconds(const ::std_msgs::msg::dds_::Header_& header)
{
    return static_cast<double>(header.stamp().sec()) + 1e-9 * header.stamp().nanosec();
}

inline void write_odom(Odometry& msg, const OdomSample& s)
{
    set_stamp(msg.header(), s.t);
    msg.header().frame_id() = kOdomFrameId;
    msg.child_frame_id() = kOdomChildFrameId;
    auto& p = msg.pose().pose().position();
    p.x() = s.pos[0]; p.y() = s.pos[1]; p.z() = s.pos[2];
    auto& q = msg.pose().pose().orientation();
    q.w() = s.quat[0]; q.x() = s.quat[1]; q.y() = s.quat[2]; q.z() = s.quat[3];
    auto& v = msg.twist().twist().linear();
    v.x() = s.lin_vel_b[0]; v.y() = s.lin_vel_b[1]; v.z() = s.lin_vel_b[2];
    auto& w = msg.twist().twist().angular();
    w.x() = s.ang_vel_b[0]; w.y() = s.ang_vel_b[1]; w.z() = s.ang_vel_b[2];
}

inline OdomSample read_odom(const Odometry& msg)
{
    OdomSample s;
    s.t = stamp_seconds(msg.header());
    const auto& p = msg.pose().pose().position();
    s.pos[0] = p.x(); s.pos[1] = p.y(); s.pos[2] = p.z();
    const auto& q = msg.pose().pose().orientation();
    s.quat[0] = q.w(); s.quat[1] = q.x(); s.quat[2] = q.y(); s.quat[3] = q.z();
    const auto& v = msg.twist().twist().linear();
    s.lin_vel_b[0] = v.x(); s.lin_vel_b[1] = v.y(); s.lin_vel_b[2] = v.z();
    const auto& w = msg.twist().twist().angular();
    s.ang_vel_b[0] = w.x(); s.ang_vel_b[1] = w.y(); s.ang_vel_b[2] = w.z();
    return s;
}

/// One-time layout of the depth message (fields, steps, frame id).
inline void init_depth_image(DepthImage& msg, uint32_t width, uint32_t height)
{
    msg.header().frame_id() = kDepthFrameId;
    msg.height() = height;
    msg.width() = width;
    ::sensor_msgs::msg::dds_::PointField_ field;
    field.name() = kDepthFieldName;
    field.offset() = 0;
    field.datatype() = kPointFieldUint16;
    field.count() = 1;
    msg.fields() = {field};
    msg.is_bigendian() = false;
    msg.point_step() = sizeof(uint16_t);
    msg.row_step() = width * sizeof(uint16_t);
    msg.is_dense() = false;  // 0 = no return
    msg.data().assign(static_cast<size_t>(width) * height * sizeof(uint16_t), 0);
}

/// Copies width*height little-endian uint16 millimetre values (row 0 = top).
inline void write_depth_image(DepthImage& msg, double t, const uint16_t* depth_mm)
{
    set_stamp(msg.header(), t);
    const size_t n = static_cast<size_t>(msg.width()) * msg.height();
    msg.data().resize(n * sizeof(uint16_t));
    std::memcpy(msg.data().data(), depth_mm, n * sizeof(uint16_t));
}

/// Validates the layout and unpacks the millimetre values. Returns false (and
/// leaves `depth_mm` untouched) when the message is not a depth image in the
/// convention above.
inline bool read_depth_image(const DepthImage& msg, std::vector<uint16_t>& depth_mm)
{
    if (msg.fields().size() != 1 || msg.fields()[0].name() != kDepthFieldName ||
        msg.fields()[0].datatype() != kPointFieldUint16 || msg.point_step() != sizeof(uint16_t) ||
        msg.is_bigendian()) {
        return false;
    }
    const size_t n = static_cast<size_t>(msg.width()) * msg.height();
    if (msg.data().size() != n * sizeof(uint16_t)) return false;
    depth_mm.resize(n);
    std::memcpy(depth_mm.data(), msg.data().data(), n * sizeof(uint16_t));
    return true;
}

}  // namespace msgs
}  // namespace unitree_rl
