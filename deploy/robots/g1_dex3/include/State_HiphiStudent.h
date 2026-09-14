#pragma once

// Depth-distilled hiphi student on the frozen UMT base (smp_v2 task
// `Hiphi-Tracking-Multi-V2-Distill-G1`, tasks/hiphi_tracking_multi_distill +
// tasks/hiphi_tracking_umt/actions.py UmtResidualAction).
//
// Two networks run every policy step (50 Hz) on the same robot state:
//
//   UMT base (160 obs -> 29):  zest_ref (hands masked to the default pose) |
//       motion_anchor_pos_b | motion_anchor_ori_b | base_lin_vel |
//       base_ang_vel | projected_gravity | joint_pos_rel (29) |
//       joint_vel_rel (29) | its own last action (29)
//   student (196 + 1x36x64 depth -> 43):  zest_ref (unmasked) |
//       motion_anchor_ori_b | base_ang_vel | projected_gravity |
//       joint_pos_rel over all 43 joints (entity order, fingers interleaved) |
//       joint_vel_rel (43) | its own last action (43) ; camera_depth
//
// and the targets are composed as in training:
//
//   body (29, SDK order):  q = clamp(q_ref + SIGMA * pi_umt + res_scale * a[0:29], joint limits)
//   hand (14, Dex3 order): q = clamp(q_ref_hand + hand_scale * a[29:43], joint limits)
//
// SIGMA (G1_ACTION_SCALE) and res_scale / hand_scale live in the two
// deploy.yaml action terms (`umt_policy_dir`, `student_policy_dir`); the
// joint limits in the student's deploy.yaml. Both networks' inputs come from
// the streamed state estimate (rt/odom_pelvis) and depth image
// (rt/depth_camera): without them the state refuses to run.
//
// Body targets go to rt/lowcmd like every other state; finger targets to the
// Dex3Hands publisher.

#include "FSM/State_RLBase.h"
#include "State_UmtMimic.h"  // MotionLoader_ (clip format / layout)
#include "Dex3Hands.h"

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>


class State_HiphiStudent : public FSMState
{
public:
    State_HiphiStudent(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        dex3_hands().set_open_pose();
        probe_dump_();
    }

private:
    using MotionLoader_ = State_UmtMimic::MotionLoader_;

    void compose_and_store_(double t);
    void probe_dump_();

    std::shared_ptr<MotionLoader_> motion_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> umt_env_;      // frozen base
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> student_env_;  // residual student

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::array<float, 2> time_range_;
    bool align_z_ = false;
    std::string depth_sensor_ = "depth_camera";
    bool require_streams_ = true;
    float odom_max_age_ms_ = 100.0f;
    float depth_max_age_ms_ = 200.0f;
    std::atomic<int> stale_steps_{0};

    // Joint limits over all 43 joints in entity order (student deploy.yaml).
    std::vector<float> limit_lo_, limit_hi_;

    // Composed targets, written by the policy thread and read by run().
    std::mutex cmd_mutex_;
    std::vector<float> body_cmd_;  // 29, SDK order
    std::vector<float> hand_cmd_;  // 14, left then right, Dex3 motor order
    bool have_cmd_ = false;

    // Probe (dumped as npz on exit): per policy step t, reference (43, entity
    // order), UMT offset (29), scaled residual (43, [body|hand]), commanded
    // targets (43, [body|hand]) and measured joints (43, [body|hand]).
    std::vector<float> probe_t_, probe_ref_, probe_umt_, probe_res_, probe_cmd_, probe_q_;
    std::string probe_path_;
};


REGISTER_FSM(State_HiphiStudent)
