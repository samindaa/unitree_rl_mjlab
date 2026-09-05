#include "State_UmtMimic.h"
#include <ctime>
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

// Yaw offset between the robot's anchor (torso) and the clip's anchor at
// enter(): the reference lives in the clip's world frame, the robot in the
// IMU's world frame, so the reference is rotated into the robot's heading
// (in training the robot is teleported onto the reference at RSI, so the
// frames coincide by construction).
static Eigen::Quaternionf init_quat;
std::shared_ptr<State_UmtMimic::MotionLoader_> State_UmtMimic::motion = nullptr;


/// Robot torso (tracking anchor) orientation in world: pelvis IMU quaternion
/// composed with the three waist joints (yaw, roll, pitch = SDK motors 12-14).
static Eigen::Quaternionf robot_anchor_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto & motors = robot->lowstate->msg_.motor_state();

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());

    return torso_quat;
}


namespace isaaclab
{
namespace mdp
{

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

    const auto & joint_pos = loader->joint_pos();
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

    const Eigen::Quaternionf real_quat_w = robot_anchor_quat_w(env);
    const Eigen::Quaternionf ref_quat_w  = init_quat * loader->anchor_quaternion();

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

    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), layout);
    spdlog::info("Loaded UMT motion '{}': {} frames @ {:.0f} fps ({:.2f}s), {} joints",
                 motion_file.stem().string(), motion_->num_frames, 1.0f / motion_->dt,
                 motion_->duration, motion_->num_joints);
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
        auto ref_yaw = isaaclab::yawQuaternion(motion->anchor_quaternion()).toRotationMatrix();
        auto robot_yaw = isaaclab::yawQuaternion(robot_anchor_quat_w(env.get())).toRotationMatrix();
        init_quat = robot_yaw * ref_yaw.transpose();
        env->reset();

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
    // TODO(dex3): publish motion->hand_joint_pos() on the Dex3 hand command
    // topics once the hands are mounted (UMT keeps them at the default pose).
}
