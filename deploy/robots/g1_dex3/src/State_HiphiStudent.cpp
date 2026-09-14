#include "State_HiphiStudent.h"
#include <ctime>
#include "UmtAnchor.h"
#include "unitree_articulation.h"
#include "sources/depth_source.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"


namespace isaaclab
{
namespace mdp
{

// All 43 joints in entity order (fingers interleaved after each wrist, see
// MotionLoader_::Layout): body joints from lowstate, fingers from the Dex3
// hand state topics (zeros until the hands report).
static void entity_joint_state(ManagerBasedRLEnv* env, std::vector<float>& q, std::vector<float>& dq)
{
    auto loader = State_UmtMimic::motion;
    const auto & L = loader->layout();
    const auto & data = env->robot->data;
    q.assign(loader->num_joints, 0.0f);
    dq.assign(loader->num_joints, 0.0f);
    for (size_t i = 0; i < L.body_joint_ids.size(); ++i) {
        q[L.body_joint_ids[i]] = data.joint_pos[i];
        dq[L.body_joint_ids[i]] = data.joint_vel[i];
    }
    Dex3Hands::Targets hq, hdq;
    dex3_hands().joint_state(hq, hdq);
    for (size_t k = 0; k < L.hand_joint_ids.size() && k < hq.size(); ++k) {
        q[L.hand_joint_ids[k]] = hq[k];
        dq[L.hand_joint_ids[k]] = hdq[k];
    }
}

// mjlab `joint_pos_rel` / `joint_vel_rel` over the whole entity (43): the
// student's proprioception. Body defaults come from deploy.yaml
// default_joint_pos (29, SDK order), finger defaults from `hand_default_pos`
// (default 0 = the hands' qpos0).
REGISTER_OBSERVATION(joint_pos_rel_entity)
{
    std::vector<float> q, dq;
    entity_joint_state(env, q, dq);
    auto loader = State_UmtMimic::motion;
    const auto & L = loader->layout();
    const auto & data = env->robot->data;
    for (size_t i = 0; i < L.body_joint_ids.size(); ++i) {
        q[L.body_joint_ids[i]] -= data.default_joint_pos[i];
    }
    if (params["hand_default_pos"]) {
        const auto hand_default = params["hand_default_pos"].as<std::vector<float>>();
        for (size_t k = 0; k < L.hand_joint_ids.size() && k < hand_default.size(); ++k) {
            q[L.hand_joint_ids[k]] -= hand_default[k];
        }
    }
    return q;
}

REGISTER_OBSERVATION(joint_vel_rel_entity)
{
    std::vector<float> q, dq;
    entity_joint_state(env, q, dq);
    return dq;
}

}
}


State_HiphiStudent::State_HiphiStudent(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto umt_dir = param::parser_policy_dir(cfg["umt_policy_dir"].as<std::string>());
    auto student_dir = param::parser_policy_dir(cfg["student_policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }
    MotionLoader_::Layout layout;
    if (cfg["root_body_index"])   layout.root_body_index   = cfg["root_body_index"].as<int>();
    if (cfg["anchor_body_index"]) layout.anchor_body_index = cfg["anchor_body_index"].as<int>();
    if (cfg["body_joint_ids"])    layout.body_joint_ids    = cfg["body_joint_ids"].as<std::vector<int>>();
    if (cfg["hand_joint_ids"])    layout.hand_joint_ids    = cfg["hand_joint_ids"].as<std::vector<int>>();

    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), layout);
    spdlog::info("Loaded hiphi clip '{}': {} frames @ {:.0f} fps ({:.2f}s), {} joints",
                 motion_file.stem().string(), motion_->num_frames, 1.0f / motion_->dt,
                 motion_->duration, motion_->num_joints);
    // The observation / action terms read the clip through the shared slot.
    State_UmtMimic::motion = motion_;

    time_range_[0] = cfg["time_start"] ? std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration) : 0.0f;
    time_range_[1] = cfg["time_end"] ? std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration) : motion_->duration;
    std::string end_state = cfg["end_state"] ? cfg["end_state"].as<std::string>() : "Velocity";
    if (cfg["align_z"]) align_z_ = cfg["align_z"].as<bool>();
    if (cfg["depth_sensor"]) depth_sensor_ = cfg["depth_sensor"].as<std::string>();
    if (cfg["require_streams"]) require_streams_ = cfg["require_streams"].as<bool>();
    if (cfg["odom_max_age_ms"]) odom_max_age_ms_ = cfg["odom_max_age_ms"].as<float>();
    if (cfg["depth_max_age_ms"]) depth_max_age_ms_ = cfg["depth_max_age_ms"].as<float>();

    const auto umt_yaml = YAML::LoadFile(umt_dir / "params" / "deploy.yaml");
    const auto student_yaml = YAML::LoadFile(student_dir / "params" / "deploy.yaml");

    umt_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(umt_yaml, articulation);
    umt_env_->alg = std::make_unique<isaaclab::OrtRunner>(umt_dir / "exported" / "policy.onnx");
    student_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(student_yaml, articulation);
    student_env_->alg = std::make_unique<isaaclab::OrtRunner>(student_dir / "exported" / "policy.onnx");

    const int n_body = static_cast<int>(layout.body_joint_ids.size());
    const int n_hand = static_cast<int>(layout.hand_joint_ids.size());
    if (umt_env_->action_manager->total_action_dim() != n_body) {
        spdlog::critical("UMT base action dim {} != body joints {}", umt_env_->action_manager->total_action_dim(), n_body);
        std::exit(-1);
    }
    if (student_env_->action_manager->total_action_dim() != n_body + n_hand) {
        spdlog::critical("student action dim {} != body + hand joints {}", student_env_->action_manager->total_action_dim(), n_body + n_hand);
        std::exit(-1);
    }
    if (static_cast<int>(articulation->data.joint_ids_map.size()) != n_body) {
        spdlog::critical("deploy.yaml joint_ids_map ({}) != body_joint_ids ({})", articulation->data.joint_ids_map.size(), n_body);
        std::exit(-1);
    }
    limit_lo_ = student_yaml["joint_pos_limits_lo"].as<std::vector<float>>();
    limit_hi_ = student_yaml["joint_pos_limits_hi"].as<std::vector<float>>();
    if (static_cast<int>(limit_lo_.size()) != motion_->num_joints || static_cast<int>(limit_hi_.size()) != motion_->num_joints) {
        spdlog::critical("joint_pos_limits_lo/hi must have {} entries (entity order)", motion_->num_joints);
        std::exit(-1);
    }
    if (require_streams_) {
        if (!unitree_rl::odom_source()) {
            spdlog::critical("HiphiStudent needs the state estimate stream: enable sources.odom in config.yaml");
            std::exit(-1);
        }
        if (!unitree_rl::depth_sources().count(depth_sensor_)) {
            spdlog::critical("HiphiStudent needs the depth stream: configure sources.depth.{} in config.yaml", depth_sensor_);
            std::exit(-1);
        }
    }
    body_cmd_.assign(n_body, 0.0f);
    hand_cmd_.assign(n_hand, 0.0f);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (umt_env_->episode_length * umt_env_->step_dt) > time_range_[1]; },
            FSMStringMap.right.at(end_state)
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(umt_env_.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
    // Streams stale for 0.5 s of policy steps -> hand back to Velocity.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                if (stale_steps_ < 25) return false;
                spdlog::warn("HiphiStudent: state estimate / depth stream stale, leaving");
                return true;
            },
            FSMStringMap.right.at("Velocity")
        )
    );
}

void State_HiphiStudent::compose_and_store_(double t)
{
    const auto umt = umt_env_->action_manager->processed_actions();        // SIGMA * pi_umt (29)
    const auto res = student_env_->action_manager->processed_actions();    // scaled residual [body 29 | hand 14]
    const Eigen::VectorXf ref = motion_->joint_pos();                      // 43, entity order
    const auto & L = motion_->layout();
    const int n_body = static_cast<int>(L.body_joint_ids.size());
    const int n_hand = static_cast<int>(L.hand_joint_ids.size());

    std::vector<float> body(n_body), hand(n_hand);
    for (int i = 0; i < n_body; ++i) {
        const int j = L.body_joint_ids[i];
        body[i] = std::clamp(ref[j] + umt[i] + res[i], limit_lo_[j], limit_hi_[j]);
    }
    for (int k = 0; k < n_hand; ++k) {
        const int j = L.hand_joint_ids[k];
        hand[k] = std::clamp(ref[j] + res[n_body + k], limit_lo_[j], limit_hi_[j]);
    }
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        body_cmd_ = body;
        hand_cmd_ = hand;
        have_cmd_ = true;
    }

    // probe
    std::vector<float> q, dq;
    isaaclab::mdp::entity_joint_state(umt_env_.get(), q, dq);
    probe_t_.push_back(static_cast<float>(t));
    probe_ref_.insert(probe_ref_.end(), ref.data(), ref.data() + ref.size());
    probe_umt_.insert(probe_umt_.end(), umt.begin(), umt.end());
    probe_res_.insert(probe_res_.end(), res.begin(), res.end());
    probe_cmd_.insert(probe_cmd_.end(), body.begin(), body.end());
    probe_cmd_.insert(probe_cmd_.end(), hand.begin(), hand.end());
    for (int i = 0; i < n_body; ++i) probe_q_.push_back(q[L.body_joint_ids[i]]);
    for (int k = 0; k < n_hand; ++k) probe_q_.push_back(q[L.hand_joint_ids[k]]);
}

void State_HiphiStudent::enter()
{
    for (int i = 0; i < umt_env_->robot->data.joint_stiffness.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = umt_env_->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = umt_env_->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    State_UmtMimic::motion = motion_;
    umt_env_->reset();
    student_env_->reset();
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        have_cmd_ = false;
    }
    stale_steps_ = 0;
    probe_t_.clear(); probe_ref_.clear(); probe_umt_.clear(); probe_res_.clear(); probe_cmd_.clear(); probe_q_.clear();
    probe_path_ = fmt::format("/tmp/hiphi_probe_{}.npz", std::time(nullptr));

    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(umt_env_->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
        auto sleepTill = clock::now() + dt;

        umt_env_->robot->update();
        motion_->update(time_range_[0]);
        umt::align_clip_to_robot(umt_env_.get(), motion_->anchor_quaternion(), motion_->anchor_position(), align_z_);
        umt_env_->reset();
        student_env_->reset();
        {
            const auto & data = umt_env_->robot->data;
            const auto err = isaaclab::observations_map().at("motion_anchor_pos_b")(umt_env_.get(), YAML::Node());
            auto depth = unitree_rl::depth_sources().count(depth_sensor_) ? unitree_rl::depth_sources().at(depth_sensor_) : nullptr;
            spdlog::info("HiphiStudent enter: odom {} (age {:.0f} ms), depth {} (age {:.0f} ms, {} frames); "
                         "clip offset [{:.3f} {:.3f} {:.3f}]; motion_anchor_pos_b [{:.3f} {:.3f} {:.3f}]",
                         data.has_odom ? "yes" : "NO", data.odom_age_ms,
                         depth && depth->received() ? "yes" : "NO", depth ? depth->age_ms() : -1.0, depth ? depth->seq() : 0,
                         umt::init_pos.x(), umt::init_pos.y(), umt::init_pos.z(), err[0], err[1], err[2]);
        }

        while (policy_thread_running)
        {
            umt_env_->robot->update();
            const double t = umt_env_->episode_length * umt_env_->step_dt + time_range_[0];
            motion_->update(static_cast<float>(t));
            // Same state for both networks (each step() refreshes the
            // articulation; a few microseconds apart).
            umt_env_->step();
            student_env_->step();
            compose_and_store_(t);

            if (require_streams_) {
                const auto & data = umt_env_->robot->data;
                auto depth = unitree_rl::depth_sources().at(depth_sensor_);
                const bool stale = !data.has_odom || data.odom_age_ms > odom_max_age_ms_ ||
                                   !depth->received() || depth->age_ms() > depth_max_age_ms_;
                stale_steps_ = stale ? stale_steps_ + 1 : 0;
            }

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_HiphiStudent::run()
{
    std::vector<float> body, hand;
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (!have_cmd_) return;  // hold the previous state's targets until the first policy step
        body = body_cmd_;
        hand = hand_cmd_;
    }
    const auto & ids = umt_env_->robot->data.joint_ids_map;
    for (size_t i = 0; i < ids.size(); i++) {
        lowcmd->msg_.motor_cmd()[ids[i]].q() = body[i];
    }
    if (dex3_hands().enabled()) {
        dex3_hands().set_targets(hand.data(), static_cast<int>(hand.size()));
    }
}

void State_HiphiStudent::probe_dump_()
{
    if (probe_t_.empty()) return;
    const size_t T = probe_t_.size();
    const size_t J = probe_ref_.size() / T, B = probe_umt_.size() / T, A = probe_res_.size() / T;
    cnpy::npz_save(probe_path_, "t", probe_t_.data(), {T}, "w");
    cnpy::npz_save(probe_path_, "ref_entity", probe_ref_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "umt_offset", probe_umt_.data(), {T, B}, "a");
    cnpy::npz_save(probe_path_, "residual", probe_res_.data(), {T, A}, "a");
    cnpy::npz_save(probe_path_, "q_cmd", probe_cmd_.data(), {T, A}, "a");
    cnpy::npz_save(probe_path_, "q_meas", probe_q_.data(), {T, A}, "a");
    spdlog::info("HiphiStudent probe: {} steps -> {}", T, probe_path_);
    probe_t_.clear(); probe_ref_.clear(); probe_umt_.clear(); probe_res_.clear(); probe_cmd_.clear(); probe_q_.clear();
}
