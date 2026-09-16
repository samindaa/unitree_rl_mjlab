#pragma once

// The hiphi policy stack (frozen UMT base + depth student, see
// State_HiphiStudent.h for the composition), shared by the three FSM states
// HiphiEaseIn -> HiphiStudent -> HiphiEaseOut so that the UMT's own
// last-action history, the clip alignment and the hand targets stay
// continuous through the hand-overs. Each state only selects the PHASE the
// 50 Hz policy thread runs:
//
//   EaseIn   synthetic reference from the robot's measured pose into the
//            clip's first frame (ease_in_cubic, duration_s), UMT only.
//   Clip     the clip itself, student residual on (faded in over
//            residual_fade_s).
//   EaseOut  synthetic reference from the clip's last frame into the hold
//            pose (ease_out_cubic, duration_s) and then the hold pose with
//            zero reference velocities for as long as the state lasts;
//            residual faded out, fingers frozen at their last targets.
//
// This is the online form of g1_spinkick_example/pkl_to_csv.py's padding
// (ease-in from a standing pose, ease-out into a hold, hold), starting from
// the measured pose and with the balance controller (UMT base) in the loop
// throughout — the deployment analogue of RSI.
//
// The thread is stopped by every family state's exit() and restarted by the
// next enter(); `continue_sequence` keeps the stack's buffers across such a
// restart (EaseIn -> Clip -> EaseOut), a fresh start resets them.

#include "FSM/State_RLBase.h"
#include "State_UmtMimic.h"  // MotionLoader_
#include "Dex3Hands.h"
#include "unitree_articulation.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class HiphiStack
{
public:
    using MotionLoader_ = State_UmtMimic::MotionLoader_;
    enum class Phase { None, EaseIn, Clip, EaseOut };
    enum class Easing { InCubic, InOutCubic, OutCubic };

    struct EaseConfig
    {
        float duration_s = 1.0f;
        Easing easing = Easing::InCubic;
        std::string hold_pose = "last";   // EaseOut: "last" | "default"
        float residual_fade_s = 0.2f;
    };

    /// Builds the envs from the FSM.HiphiStudent config block (called once by
    /// the first family state constructed).
    static HiphiStack& instance();
    void configure(const YAML::Node& student_cfg);
    bool configured() const { return configured_; }

    /// Start the policy thread in `phase`. `continue_sequence` keeps the
    /// alignment, the UMT buffers and the hand targets from the previous
    /// phase (EaseIn -> Clip -> EaseOut); otherwise everything is reset and
    /// the clip is re-aligned to the robot.
    void start(Phase phase, bool continue_sequence, const EaseConfig& ease);
    void stop();

    Phase phase() const { return phase_; }
    Phase last_phase() const { return last_phase_; }
    /// True once the running phase reached its end (EaseIn: blend done;
    /// Clip: time_end; EaseOut: never).
    bool phase_done() const { return phase_done_; }
    bool streams_stale() const { return stale_steps_ >= 25; }
    float max_joint_error() const { return max_joint_err_; }

    /// Latest composed targets for run(): false until the first policy step
    /// of this start().
    bool latest_commands(std::vector<float>& body, std::vector<float>& hand);

    void set_gains(LowCmd_t& lowcmd) const;
    const std::vector<float>& joint_ids_map() const { return articulation_->data.joint_ids_map; }
    void probe_dump();

private:
    struct Pose
    {
        Eigen::Vector3f root_pos, anchor_pos;
        Eigen::Quaternionf root_quat, anchor_quat;
        Eigen::VectorXf q;           // all joints, entity order
        Eigen::VectorXf qd;          // all joints
        Eigen::Vector3f anchor_v, anchor_w;
    };

    static float ease(Easing e, float t);
    Pose clip_pose(const std::shared_ptr<MotionLoader_>& clip, int frame) const;
    Pose robot_pose_in_clip_frame(const Eigen::VectorXf& q_fallback) const;
    std::shared_ptr<MotionLoader_> build_transition(const Pose& a, const Pose& b, float duration_s, Easing easing,
                                                    float hold_s, bool carry_velocity) const;

    void loop();
    void step();
    void compose_and_store(double t);

    bool configured_ = false;
    YAML::Node cfg_;
    std::shared_ptr<unitree::BaseArticulation<LowState_t::SharedPtr>> articulation_;
    std::shared_ptr<MotionLoader_> clip_;
    std::shared_ptr<MotionLoader_> synthetic_;
    std::shared_ptr<MotionLoader_> active_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> umt_env_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> student_env_;

    std::array<float, 2> time_range_{0.0f, 0.0f};
    bool align_z_ = false;
    std::string depth_sensor_ = "depth_camera";
    bool require_streams_ = true;
    float odom_max_age_ms_ = 100.0f, depth_max_age_ms_ = 200.0f;
    std::vector<float> limit_lo_, limit_hi_;
    std::vector<float> default_pose_;  // 43, entity order (hold_pose: default)

    // phase state
    std::atomic<Phase> phase_{Phase::None};
    Phase last_phase_ = Phase::None;
    EaseConfig ease_cfg_;
    long phase_steps_ = 0;
    std::atomic<bool> phase_done_{false};
    float residual_gain_ = 0.0f, residual_gain_target_ = 0.0f;
    std::atomic<int> stale_steps_{0};
    std::atomic<float> max_joint_err_{0.0f};

    // outputs
    std::mutex cmd_mutex_;
    std::vector<float> body_cmd_, hand_cmd_;
    bool have_cmd_ = false;
    std::vector<float> hand_hold_;     // fingers frozen during EaseOut
    std::vector<float> last_body_ref_; // for the ease-out start pose

    std::thread thread_;
    std::atomic<bool> running_{false};

    // probe: per policy step (all phases)
    std::vector<float> probe_t_, probe_phase_, probe_ref_, probe_umt_, probe_res_, probe_cmd_, probe_q_;
    std::string probe_path_;
};

/// Config parsing helper shared by the ease states.
HiphiStack::EaseConfig parse_ease_config(const YAML::Node& node, HiphiStack::Easing default_easing);
