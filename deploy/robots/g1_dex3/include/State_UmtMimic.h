#pragma once

// UMT (universal motion tracking) deploy state for G1 + Dex3.
//
// Mirrors smp_v2 task `Umt-Tracking-G1-No-State-Estimation`
// (smp_v2/src/tasks/umt): ZEST-style residual-on-reference tracking of a
// body-only clip on the G1+Dex3 entity (43 joints, 46 bodies). The policy
// observes / actuates the 29 body joints only; the 14 Dex3 finger joints are
// frozen at the clip's default pose (no hand command is sent yet — hook in
// `MotionLoader_::hand_joint_pos()` when the Dex3 hands arrive).
//
// Actor observation (154, in this order):
//   zest_ref            55 = root z(1) + roll/pitch(2) + anchor lin vel(3)
//                            + anchor ang vel(3) + gravity in anchor frame(3)
//                            + reference joint pos (43, ALL joints)
//   motion_anchor_ori_b  6
//   base_ang_vel         3
//   projected_gravity    3
//   joint_pos_rel       29   (relative to KNEES_BENT default pose)
//   joint_vel_rel       29
//   last_action         29
// Action (29): q_cmd = q_ref[body_joint] + scale * a   (ReferenceJointPositionAction)

#include "FSM/State_RLBase.h"
#include <cnpy.h>
#include <numeric>


class State_UmtMimic : public FSMState
{
public:
    State_UmtMimic(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs / action computation
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::array<float, 2> time_range_;
};


/**
 * Clip loader for the mjlab MotionLoader npz format as produced by
 * scripts/umt_bundle_to_deploy_npz.py (numeric arrays only, uncompressed —
 * the vendored cnpy cannot read deflate+zip64 entries or <U string arrays):
 *
 *   joint_pos       [frame, 43]
 *   joint_vel       [frame, 43]
 *   body_pos_w      [frame, 46, 3]
 *   body_quat_w     [frame, 46, 4]   (w, x, y, z)
 *   body_lin_vel_w  [frame, 46, 3]
 *   body_ang_vel_w  [frame, 46, 3]
 *   fps             [1]              (optional, default 50)
 *
 * Joint order is the mjlab G1+Dex3 entity order (fingers interleaved after
 * each wrist):
 *   0-11  legs, 12-14 waist, 15-21 left arm, 22-28 left hand,
 *   29-35 right arm, 36-42 right hand.
 * Body order: 0 = pelvis (root), 15 = torso_link (tracking anchor).
 */
class State_UmtMimic::MotionLoader_
{
public:
    struct Layout
    {
        int root_body_index = 0;
        int anchor_body_index = 15;
        std::vector<int> body_joint_ids = {
            0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14,
            15,16,17,18,19,20,21,
            29,30,31,32,33,34,35};
        std::vector<int> hand_joint_ids = {
            22,23,24,25,26,27,28,
            36,37,38,39,40,41,42};
    };

    explicit MotionLoader_(const std::string& motion_file, const Layout& layout)
    : layout_(layout)
    {
        load_data_from_npz(motion_file);
        num_frames = static_cast<int>(dof_positions.size());
        duration = num_frames * dt;

        update(0.0f);
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        auto require = [&](const char* key) -> cnpy::NpyArray& {
            auto it = npz_data.find(key);
            if (it == npz_data.end()) {
                throw std::runtime_error(std::string("motion npz is missing array '") + key + "'");
            }
            if (it->second.word_size != sizeof(float)) {
                throw std::runtime_error(std::string("motion npz array '") + key + "' must be float32");
            }
            return it->second;
        };

        auto& body_pos_w     = require("body_pos_w");     // [frame, body, 3]
        auto& body_quat_w    = require("body_quat_w");    // [frame, body, 4]
        auto& body_lin_vel_w = require("body_lin_vel_w"); // [frame, body, 3]
        auto& body_ang_vel_w = require("body_ang_vel_w"); // [frame, body, 3]
        auto& joint_pos      = require("joint_pos");      // [frame, dof]
        auto& joint_vel      = require("joint_vel");      // [frame, dof]

        dt = 1.0f / 50.0f;
        if (npz_data.find("fps") != npz_data.end()) {
            auto& fps = npz_data["fps"];
            if (fps.word_size == sizeof(float)) {
                dt = 1.0f / fps.data<float>()[0];
            } else if (fps.word_size == sizeof(double)) {
                dt = 1.0f / static_cast<float>(fps.data<double>()[0]);
            }
        }

        const size_t num_frames_npz = body_pos_w.shape[0];
        const size_t num_bodies = body_pos_w.shape[1];
        num_joints = static_cast<int>(joint_pos.shape[1]);

        auto check_body = [&](int idx, const char* what) {
            if (idx < 0 || static_cast<size_t>(idx) >= num_bodies) {
                throw std::runtime_error(fmt::format("{} body index {} out of range (npz has {} bodies)", what, idx, num_bodies));
            }
        };
        check_body(layout_.root_body_index, "root");
        check_body(layout_.anchor_body_index, "anchor");
        for (int j : layout_.body_joint_ids) {
            if (j < 0 || j >= num_joints) {
                throw std::runtime_error(fmt::format("body joint id {} out of range (npz has {} joints)", j, num_joints));
            }
        }

        root_positions.clear();
        anchor_quaternions.clear();
        anchor_lin_velocities.clear();
        anchor_ang_velocities.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t stride_pos  = body_pos_w.shape[1] * body_pos_w.shape[2];
        const size_t stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];
        const size_t stride_lin  = body_lin_vel_w.shape[1] * body_lin_vel_w.shape[2];
        const size_t stride_ang  = body_ang_vel_w.shape[1] * body_ang_vel_w.shape[2];

        auto read_vec3 = [](const float* base, size_t frame, size_t stride, int body) {
            return Eigen::Vector3f::Map(base + frame * stride + body * 3);
        };
        auto read_quat = [](const float* base, size_t frame, size_t stride, int body) {
            const float* q = base + frame * stride + body * 4;
            return Eigen::Quaternionf(q[0], q[1], q[2], q[3]); // w, x, y, z
        };

        for (size_t i = 0; i < num_frames_npz; i++)
        {
            root_positions.push_back(read_vec3(body_pos_w.data<float>(), i, stride_pos, layout_.root_body_index));
            root_quaternions.push_back(read_quat(body_quat_w.data<float>(), i, stride_quat, layout_.root_body_index));

            anchor_quaternions.push_back(read_quat(body_quat_w.data<float>(), i, stride_quat, layout_.anchor_body_index));
            anchor_lin_velocities.push_back(read_vec3(body_lin_vel_w.data<float>(), i, stride_lin, layout_.anchor_body_index));
            anchor_ang_velocities.push_back(read_vec3(body_ang_vel_w.data<float>(), i, stride_ang, layout_.anchor_body_index));

            dof_positions.push_back(Eigen::VectorXf::Map(joint_pos.data<float>() + i * num_joints, num_joints));
            dof_velocities.push_back(Eigen::VectorXf::Map(joint_vel.data<float>() + i * num_joints, num_joints));
        }
    }

    void update(float time)
    {
        float phase = std::clamp(time, 0.0f, duration);
        float f = phase / dt;
        frame = static_cast<int>(std::floor(f));
        frame = std::min(frame, num_frames - 1);
    }

    // --- per-frame accessors (current frame) ---
    Eigen::Vector3f root_position() const { return root_positions[frame]; }
    Eigen::Quaternionf root_quaternion() const { return root_quaternions[frame]; }
    Eigen::Quaternionf anchor_quaternion() const { return anchor_quaternions[frame]; }
    Eigen::Vector3f anchor_lin_vel_w() const { return anchor_lin_velocities[frame]; }
    Eigen::Vector3f anchor_ang_vel_w() const { return anchor_ang_velocities[frame]; }

    /// All 43 reference joint positions (entity order) — what zest_ref feeds the policy.
    const Eigen::VectorXf& joint_pos() const { return dof_positions[frame]; }
    const Eigen::VectorXf& joint_vel() const { return dof_velocities[frame]; }

    /// Reference positions of the 29 body joints, in policy/SDK order.
    Eigen::VectorXf body_joint_pos() const { return gather(dof_positions[frame], layout_.body_joint_ids); }
    /// Reference positions of the 14 Dex3 finger joints (frozen at default in UMT).
    Eigen::VectorXf hand_joint_pos() const { return gather(dof_positions[frame], layout_.hand_joint_ids); }

    const Layout& layout() const { return layout_; }

    float dt;
    int num_frames;
    int num_joints;
    float duration;

    int frame;
    std::vector<Eigen::Vector3f> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::Quaternionf> anchor_quaternions;
    std::vector<Eigen::Vector3f> anchor_lin_velocities;
    std::vector<Eigen::Vector3f> anchor_ang_velocities;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

private:
    static Eigen::VectorXf gather(const Eigen::VectorXf& v, const std::vector<int>& ids)
    {
        Eigen::VectorXf out(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) out[i] = v[ids[i]];
        return out;
    }

    Layout layout_;
};


REGISTER_FSM(State_UmtMimic)
