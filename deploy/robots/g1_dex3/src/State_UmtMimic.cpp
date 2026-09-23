#include "State_UmtMimic.h"
#include <ctime>
#include "Dex3Hands.h"
#include "MotionLibrary.h"
#include "UmtAnchor.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

std::shared_ptr<State_UmtMimic::MotionLoader_> State_UmtMimic::motion = nullptr;


namespace isaaclab
{
namespace mdp
{

/**
 * mjlab tracking `motion_anchor_pos_b`: the clip's anchor position relative
 * to the robot's anchor, in the robot anchor frame (3). Needs the odom
 * source (sources.odom in config.yaml); zeros with a one-time warning
 * otherwise.
 */
REGISTER_OBSERVATION(motion_anchor_pos_b)
{
    static bool warned = false;
    if (!env->robot->data.has_odom) {
        if (!warned) {
            spdlog::warn("motion_anchor_pos_b: no state estimate received (sources.odom); returning zeros");
            warned = true;
        }
        return std::vector<float>(3, 0.0f);
    }
    auto loader = State_UmtMimic::motion;
    const Eigen::Vector3f ref_pos_w = umt::init_quat * loader->anchor_position() + umt::init_pos;
    const Eigen::Quaternionf robot_quat_w = umt::robot_anchor_quat_w(env);
    const Eigen::Vector3f robot_pos_w = umt::robot_anchor_pos_w(env);
    const Eigen::Vector3f pos_b = robot_quat_w.conjugate() * (ref_pos_w - robot_pos_w);
    return {pos_b.x(), pos_b.y(), pos_b.z()};
}

/**
 * ZEST Table S3 reference observation (smp_v2 tasks/zest_tracking/mdp/observations.py):
 *   [root z, root roll, root pitch, anchor lin vel (anchor frame), anchor ang vel
 *    (anchor frame), gravity (anchor frame), reference joint pos (all 43 joints)]
 * Purely a function of the clip — no robot state, no world xy / heading.
 */
REGISTER_OBSERVATION(zest_ref)
{
    auto loader = State_UmtMimic::motion;
    std::vector<float> data;
    data.reserve(12 + loader->num_joints);

    const Eigen::Vector3f root_pos = loader->root_position();
    const Eigen::Quaternionf root_quat = loader->root_quaternion();
    const Eigen::Quaternionf anchor_quat = loader->anchor_quaternion();

    // Roll / pitch: XYZ extrinsic convention (mjlab euler_xyz_from_quat).
    const float qw = root_quat.w(), qx = root_quat.x(), qy = root_quat.y(), qz = root_quat.z();
    const float roll = std::atan2(2.0f * (qw * qx + qy * qz), 1.0f - 2.0f * (qx * qx + qy * qy));
    float sin_pitch = 2.0f * (qw * qy - qz * qx);
    const float pitch = std::fabs(sin_pitch) >= 1.0f
        ? std::copysign(static_cast<float>(M_PI) / 2.0f, sin_pitch)
        : std::asin(sin_pitch);

    const Eigen::Quaternionf inv_anchor = anchor_quat.conjugate();
    const Eigen::Vector3f lin_vel_b = inv_anchor * loader->anchor_lin_vel_w();
    const Eigen::Vector3f ang_vel_b = inv_anchor * loader->anchor_ang_vel_w();
    const Eigen::Vector3f gravity_b = inv_anchor * Eigen::Vector3f(0.0f, 0.0f, -1.0f);

    data.push_back(root_pos.z());
    data.push_back(roll);
    data.push_back(pitch);
    data.insert(data.end(), lin_vel_b.data(), lin_vel_b.data() + 3);
    data.insert(data.end(), ang_vel_b.data(), ang_vel_b.data() + 3);
    data.insert(data.end(), gravity_b.data(), gravity_b.data() + 3);

    // The frozen UMT base inside the hiphi stack sees the DEFAULT finger pose
    // in place of the clip's finger reference (UmtResidualActionCfg.mask_hand_ref:
    // the UMT bundles' hands were frozen there).
    Eigen::VectorXf joint_pos = loader->joint_pos();
    if (params["mask_hand_ref"] && params["mask_hand_ref"].as<bool>()) {
        std::vector<float> hand_default;
        if (params["hand_default_pos"]) hand_default = params["hand_default_pos"].as<std::vector<float>>();
        const auto & ids = loader->layout().hand_joint_ids;
        for (size_t k = 0; k < ids.size(); ++k) {
            joint_pos[ids[k]] = k < hand_default.size() ? hand_default[k] : 0.0f;
        }
    }
    data.insert(data.end(), joint_pos.data(), joint_pos.data() + joint_pos.size());
    return data;
}

/**
 * mjlab tracking `motion_anchor_ori_b`: first two columns of
 * R(robot_anchor^-1 * motion_anchor), flattened row-major (6).
 */
REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_UmtMimic::motion;

    const Eigen::Quaternionf real_quat_w = umt::robot_anchor_quat_w(env);
    const Eigen::Quaternionf ref_quat_w  = umt::init_quat * loader->anchor_quaternion();

    const Eigen::Matrix3f rot = (real_quat_w.conjugate() * ref_quat_w).toRotationMatrix();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}

/**
 * ZEST residual-on-reference action (smp_v2 tasks/zest_tracking/mdp/actions.py):
 *   q_cmd[i] = q_ref[body_joint_ids[i]] + scale[i] * a[i]
 * The `offset` key is ignored (the time-varying reference IS the offset).
 */
class ReferenceJointPositionAction : public JointAction
{
public:
    ReferenceJointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : JointAction(cfg, env)
    {
        if (!_offset.empty()) {
            spdlog::warn("ReferenceJointPositionAction: 'offset' is ignored (reference pose is the offset).");
            _offset.clear();
        }
    }

    void process_actions(std::vector<float> actions) override
    {
        _raw_actions = actions;
        const Eigen::VectorXf ref = State_UmtMimic::motion->body_joint_pos();
        if (ref.size() != _action_dim) {
            throw std::runtime_error(fmt::format(
                "ReferenceJointPositionAction: reference has {} body joints but action dim is {}",
                ref.size(), _action_dim));
        }
        for (int i(0); i < _action_dim; ++i)
        {
            float a = _raw_actions[i];
            if (!_scale.empty()) a *= _scale[i];
            _processed_actions[i] = ref[i] + a;
        }
        if (!_clip.empty())
        {
            for (int i(0); i < _action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }
};

REGISTER_ACTION(ReferenceJointPositionAction);

}


State_UmtMimic::State_UmtMimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion layout (optional overrides for clips with a different body/joint ordering)
    MotionLoader_::Layout layout;
    if (cfg["root_body_index"])   layout.root_body_index   = cfg["root_body_index"].as<int>();
    if (cfg["anchor_body_index"]) layout.anchor_body_index = cfg["anchor_body_index"].as<int>();
    if (cfg["body_joint_ids"])    layout.body_joint_ids    = cfg["body_joint_ids"].as<std::vector<int>>();
    if (cfg["hand_joint_ids"])    layout.hand_joint_ids    = cfg["hand_joint_ids"].as<std::vector<int>>();

    if (!motion_library().empty()) {
        motion_ = motion_library().current();  // re-picked on every enter()
        spdlog::info("UMT motion from the motion library: '{}' (select with the viser dropdown / udp command port)", motion_library().current_name());
    } else {
        motion_ = std::make_shared<MotionLoader_>(motion_file.string(), layout);
        spdlog::info("Loaded UMT motion '{}': {} frames @ {:.0f} fps ({:.2f}s), {} joints",
                     motion_file.stem().string(), motion_->num_frames, 1.0f / motion_->dt,
                     motion_->duration, motion_->num_joints);
    }
    motion = motion_;

    if(cfg["time_start"]) {
        float time_start = cfg["time_start"].as<float>();
        time_range_[0] = std::clamp(time_start, 0.0f, motion_->duration);
    } else {
        time_range_[0] = 0.0f;
    }
    if(cfg["time_end"]) {
        float time_end = cfg["time_end"].as<float>();
        time_range_[1] = std::clamp(time_end, 0.0f, motion_->duration);
    } else {
        time_range_[1] = motion_->duration;
    }
    std::string end_state = "Velocity";
    if (cfg["end_state"]) {
        end_state = cfg["end_state"].as<std::string>();
    }
    if (cfg["align_z"]) {
        align_z_ = cfg["align_z"].as<bool>();
    }
    if (cfg["action_delay_ms"]) {
        action_delay_ms_ = cfg["action_delay_ms"].as<float>();
        if (action_delay_ms_ > 0.0f) {
            spdlog::warn("UMT latency injection ACTIVE: commands applied {} ms late "
                         "(sim2sim experiment knob — set action_delay_ms: 0 for normal use)",
                         action_delay_ms_);
        }
    }

    // The action term reads the reference through State_UmtMimic::motion,
    // so `motion` must be set before the env (and its ActionManager) is built.
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if (static_cast<int>(motion_->layout().body_joint_ids.size()) != static_cast<int>(env->robot->data.joint_ids_map.size())) {
        spdlog::critical("body_joint_ids ({}) and deploy.yaml joint_ids_map ({}) differ in size.",
                         motion_->layout().body_joint_ids.size(), env->robot->data.joint_ids_map.size());
        std::exit(-1);
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > time_range_[1]; }, // time out
            FSMStringMap.right.at(end_state)
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_UmtMimic::enter()
{
    if (!motion_library().empty()) {
        // the clip selected since the last entry; time range re-clamped to it
        motion_ = motion_library().current();
        auto cfg = param::config["FSM"][getStateString()];
        time_range_[0] = cfg["time_start"] ? std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration) : 0.0f;
        time_range_[1] = cfg["time_end"] ? std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration) : motion_->duration;
        spdlog::info("Umt: playing clip '{}' ({:.2f} s, {} frames)", motion_library().current_name(), motion_->duration, motion_->num_frames);
    }
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    motion = motion_; // set for specific motion
    env->reset();

    probe_t_.clear();
    probe_ref_.clear();
    probe_raw_.clear();
    probe_cmd_.clear();
    probe_q_.clear();
    probe_dq_.clear();
    probe_path_ = fmt::format("/tmp/umt_probe_{}.npz", std::time(nullptr));
    delay_buf_.clear();

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        motion->update(time_range_[0]);
        umt::align_clip_to_robot(env.get(), motion->anchor_quaternion(), motion->anchor_position(), align_z_);
        env->reset();
        {
            const auto & data = env->robot->data;
            const Eigen::Vector3f ap = umt::robot_anchor_pos_w(env.get());
            const Eigen::Vector3f init_pos = umt::init_pos;
            const auto err = isaaclab::mdp::motion_anchor_pos_b(env.get(), YAML::Node());
            spdlog::info("UMT enter: odom {} (age {:.0f} ms) pelvis_imu [{:.3f} {:.3f} {:.3f}] -> torso anchor [{:.3f} {:.3f} {:.3f}]; "
                         "clip offset [{:.3f} {:.3f} {:.3f}]; motion_anchor_pos_b [{:.3f} {:.3f} {:.3f}]",
                         data.has_odom ? "yes" : "no", data.odom_age_ms, data.root_pos_w.x(), data.root_pos_w.y(), data.root_pos_w.z(),
                         ap.x(), ap.y(), ap.z(), init_pos.x(), init_pos.y(), init_pos.z(), err[0], err[1], err[2]);
        }

        while (policy_thread_running)
        {
            env->robot->update();
            motion->update(env->episode_length * env->step_dt + time_range_[0]);
            env->step();

            {   // limit-probe sample (same frame the action was computed from)
                const Eigen::VectorXf ref = motion->body_joint_pos();
                const auto raw = env->action_manager->action();
                const auto cmd = env->action_manager->processed_actions();
                const auto& q  = env->robot->data.joint_pos;
                const auto& dq = env->robot->data.joint_vel;
                probe_t_.push_back(env->episode_length * env->step_dt);
                probe_ref_.insert(probe_ref_.end(), ref.data(), ref.data() + ref.size());
                probe_raw_.insert(probe_raw_.end(), raw.begin(), raw.end());
                probe_cmd_.insert(probe_cmd_.end(), cmd.begin(), cmd.end());
                probe_q_.insert(probe_q_.end(), q.data(), q.data() + q.size());
                probe_dq_.insert(probe_dq_.end(), dq.data(), dq.data() + dq.size());
            }

            // Sleep
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_UmtMimic::probe_dump_()
{
    if (probe_t_.empty()) return;
    const size_t T = probe_t_.size();
    const size_t J = probe_cmd_.size() / T;
    cnpy::npz_save(probe_path_, "t", probe_t_.data(), {T}, "w");
    cnpy::npz_save(probe_path_, "ref", probe_ref_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "action_raw", probe_raw_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "q_cmd", probe_cmd_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "q_meas", probe_q_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "dq_meas", probe_dq_.data(), {T, J}, "a");
    spdlog::info("UMT limit-probe: {} steps ({} joints) -> {}", T, J, probe_path_);
    probe_t_.clear();
    probe_ref_.clear();
    probe_raw_.clear();
    probe_cmd_.clear();
    probe_q_.clear();
    probe_dq_.clear();
}


void State_UmtMimic::run()
{
    auto action = env->action_manager->processed_actions();
    if (action_delay_ms_ > 0.0f) {
        // apply the newest command published at or before now - delay; while
        // the buffer is younger than the delay (right after enter), the
        // oldest available command is used, so the lag ramps up to the target
        const auto now = std::chrono::steady_clock::now();
        delay_buf_.emplace_back(now, action);
        const auto cutoff = now - std::chrono::microseconds(
            static_cast<long>(action_delay_ms_ * 1000.0f));
        while (delay_buf_.size() > 1 && delay_buf_[1].first <= cutoff) {
            delay_buf_.pop_front();
        }
        action = delay_buf_.front().second;
    }
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
    // Dex3 fingers follow the clip's reference directly (no policy residual);
    // for the body-only UMT clips that is the frozen open pose. Published by
    // the Dex3Hands thread on rt/dex3/<side>/cmd.
    if (dex3_hands().enabled()) {
        const Eigen::VectorXf hand_ref = motion->hand_joint_pos();
        dex3_hands().set_targets(hand_ref.data(), static_cast<int>(hand_ref.size()));
    }
}
