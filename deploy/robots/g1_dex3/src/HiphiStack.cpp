#include "HiphiStack.h"
#include <cmath>
#include <ctime>
#include "MotionLibrary.h"
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


HiphiStack::EaseConfig parse_ease_config(const YAML::Node& node, HiphiStack::Easing default_easing)
{
    HiphiStack::EaseConfig c;
    c.easing = default_easing;
    if (!node) return c;
    if (node["duration_s"]) c.duration_s = node["duration_s"].as<float>();
    if (node["residual_fade_s"]) c.residual_fade_s = node["residual_fade_s"].as<float>();
    if (node["hold_pose"]) c.hold_pose = node["hold_pose"].as<std::string>();
    if (node["ease"]) {
        const auto e = node["ease"].as<std::string>();
        if (e == "in_cubic") c.easing = HiphiStack::Easing::InCubic;
        else if (e == "in_out_cubic") c.easing = HiphiStack::Easing::InOutCubic;
        else if (e == "out_cubic") c.easing = HiphiStack::Easing::OutCubic;
        else throw std::runtime_error("ease must be in_cubic | in_out_cubic | out_cubic, got " + e);
    }
    return c;
}


HiphiStack& HiphiStack::instance()
{
    static HiphiStack stack;
    return stack;
}

void HiphiStack::configure(const YAML::Node& cfg)
{
    if (configured_) return;
    cfg_ = cfg;
    auto umt_dir = param::parser_policy_dir(cfg["umt_policy_dir"].as<std::string>());
    auto student_dir = param::parser_policy_dir(cfg["student_policy_dir"].as<std::string>());

    articulation_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) motion_file = param::proj_dir / motion_file;
    MotionLoader_::Layout layout;
    if (cfg["root_body_index"])   layout.root_body_index   = cfg["root_body_index"].as<int>();
    if (cfg["anchor_body_index"]) layout.anchor_body_index = cfg["anchor_body_index"].as<int>();
    if (cfg["body_joint_ids"])    layout.body_joint_ids    = cfg["body_joint_ids"].as<std::vector<int>>();
    if (cfg["hand_joint_ids"])    layout.hand_joint_ids    = cfg["hand_joint_ids"].as<std::vector<int>>();
    clip_ = std::make_shared<MotionLoader_>(motion_file.string(), layout);
    spdlog::info("Loaded hiphi clip '{}': {} frames @ {:.0f} fps ({:.2f}s), {} joints",
                 motion_file.stem().string(), clip_->num_frames, 1.0f / clip_->dt, clip_->duration, clip_->num_joints);
    State_UmtMimic::motion = clip_;
    active_ = clip_;

    time_range_[0] = cfg["time_start"] ? std::clamp(cfg["time_start"].as<float>(), 0.0f, clip_->duration) : 0.0f;
    time_range_[1] = cfg["time_end"] ? std::clamp(cfg["time_end"].as<float>(), 0.0f, clip_->duration) : clip_->duration;
    if (cfg["align_z"]) align_z_ = cfg["align_z"].as<bool>();
    if (cfg["depth_sensor"]) depth_sensor_ = cfg["depth_sensor"].as<std::string>();
    if (cfg["require_streams"]) require_streams_ = cfg["require_streams"].as<bool>();
    if (cfg["odom_max_age_ms"]) odom_max_age_ms_ = cfg["odom_max_age_ms"].as<float>();
    if (cfg["depth_max_age_ms"]) depth_max_age_ms_ = cfg["depth_max_age_ms"].as<float>();

    const auto umt_yaml = YAML::LoadFile(umt_dir / "params" / "deploy.yaml");
    const auto student_yaml = YAML::LoadFile(student_dir / "params" / "deploy.yaml");
    umt_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(umt_yaml, articulation_);
    umt_env_->alg = std::make_unique<isaaclab::OrtRunner>(umt_dir / "exported" / "policy.onnx");
    student_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(student_yaml, articulation_);
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
    if (static_cast<int>(articulation_->data.joint_ids_map.size()) != n_body) {
        spdlog::critical("deploy.yaml joint_ids_map ({}) != body_joint_ids ({})", articulation_->data.joint_ids_map.size(), n_body);
        std::exit(-1);
    }
    limit_lo_ = student_yaml["joint_pos_limits_lo"].as<std::vector<float>>();
    limit_hi_ = student_yaml["joint_pos_limits_hi"].as<std::vector<float>>();
    if (static_cast<int>(limit_lo_.size()) != clip_->num_joints || static_cast<int>(limit_hi_.size()) != clip_->num_joints) {
        spdlog::critical("joint_pos_limits_lo/hi must have {} entries (entity order)", clip_->num_joints);
        std::exit(-1);
    }
    // hold_pose: default = deploy.yaml default_joint_pos (body) + open fingers
    default_pose_.assign(clip_->num_joints, 0.0f);
    for (int i = 0; i < n_body; ++i) default_pose_[layout.body_joint_ids[i]] = articulation_->data.default_joint_pos[i];
    if (require_streams_) {
        if (!unitree_rl::odom_source()) {
            spdlog::critical("Hiphi stack needs the state estimate stream: enable sources.odom in config.yaml");
            std::exit(-1);
        }
        if (!unitree_rl::depth_sources().count(depth_sensor_)) {
            spdlog::critical("Hiphi stack needs the depth stream: configure sources.depth.{} in config.yaml", depth_sensor_);
            std::exit(-1);
        }
    }
    body_cmd_.assign(n_body, 0.0f);
    hand_cmd_.assign(n_hand, 0.0f);
    configured_ = true;
}

float HiphiStack::ease(Easing e, float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    switch (e) {
        case Easing::InCubic: return t * t * t;
        case Easing::OutCubic: return 1.0f - std::pow(1.0f - t, 3.0f);
        case Easing::InOutCubic: return t < 0.5f ? 4.0f * t * t * t : 1.0f - std::pow(-2.0f * t + 2.0f, 3.0f) / 2.0f;
    }
    return t;
}

HiphiStack::Pose HiphiStack::clip_pose(const std::shared_ptr<MotionLoader_>& clip, int frame) const
{
    Pose p;
    p.root_pos = clip->root_positions[frame];
    p.root_quat = clip->root_quaternions[frame];
    p.anchor_pos = clip->anchor_positions[frame];
    p.anchor_quat = clip->anchor_quaternions[frame];
    p.q = clip->dof_positions[frame];
    p.qd = clip->dof_velocities[frame];
    p.anchor_v = clip->anchor_lin_velocities[frame];
    p.anchor_w = clip->anchor_ang_velocities[frame];
    return p;
}

// The robot's current pose expressed in the CLIP's world frame (the frames
// the loader / observation terms live in): world -> clip is the inverse of
// the alignment, x_clip = init_quat^-1 (x_w - init_pos). Without a state
// estimate the positions come from the clip frame the transition targets.
HiphiStack::Pose HiphiStack::robot_pose_in_clip_frame(const Eigen::VectorXf& q_fallback) const
{
    const auto & data = articulation_->data;
    const auto & L = clip_->layout();
    Pose p;
    std::vector<float> q, dq;
    isaaclab::mdp::entity_joint_state(umt_env_.get(), q, dq);
    Dex3Hands::Targets hq, hdq;
    if (!dex3_hands().joint_state(hq, hdq)) {
        // hands not reporting: assume they sit at their current targets (rest pose)
        const auto & rest = dex3_hands().open_pose();
        for (size_t k = 0; k < L.hand_joint_ids.size(); ++k) {
            q[L.hand_joint_ids[k]] = k < rest.size() ? rest[k] : q_fallback[L.hand_joint_ids[k]];
            dq[L.hand_joint_ids[k]] = 0.0f;
        }
    }
    p.q = Eigen::VectorXf::Map(q.data(), q.size());
    p.qd = Eigen::VectorXf::Map(dq.data(), dq.size());

    const Eigen::Quaternionf inv = umt::init_quat.conjugate();
    const Eigen::Quaternionf anchor_q_w = umt::robot_anchor_quat_w(umt_env_.get());
    p.anchor_quat = inv * anchor_q_w;
    p.root_quat = inv * data.root_quat_w;
    const Eigen::Vector3f w_w = data.root_quat_w * data.root_ang_vel_b;   // gyro -> world
    p.anchor_w = inv * w_w;
    if (data.has_odom) {
        p.anchor_pos = inv * (umt::robot_anchor_pos_w(umt_env_.get()) - umt::init_pos);
        const Eigen::Vector3f pelvis_w = data.root_pos_w - data.root_quat_w * umt::kImuInPelvisPos;
        p.root_pos = inv * (pelvis_w - umt::init_pos);
        p.anchor_v = inv * (data.root_quat_w * data.root_lin_vel_b);
    } else {
        p.anchor_pos = clip_->anchor_position();
        p.root_pos = clip_->root_position();
        p.anchor_v.setZero();
    }
    return p;
}

// Frames a -> b over duration_s with the easing, then `hold_s` of b. Joint
// positions / positions lerp, orientations slerp (pkl_to_csv.py); velocities
// by finite differences at the clip rate (as csv_to_npz produced the clips'),
// plus, with carry_velocity, a's own velocities faded out with (1 - e) — for
// ease-out from a clip frame that is still moving.
std::shared_ptr<HiphiStack::MotionLoader_> HiphiStack::build_transition(
    const Pose& a, const Pose& b, float duration_s, Easing easing, float hold_s, bool carry_velocity) const
{
    const float dt = clip_->dt;
    const int n = std::max(2, static_cast<int>(std::lround(duration_s / dt)) + 1);
    const int n_hold = std::max(0, static_cast<int>(std::lround(hold_s / dt)));
    auto m = std::make_shared<MotionLoader_>(clip_->layout(), dt);
    std::vector<float> es;
    for (int i = 0; i < n + n_hold; ++i) {
        const float s = i < n ? static_cast<float>(i) / static_cast<float>(n - 1) : 1.0f;
        const float e = ease(easing, s);
        es.push_back(e);
        m->root_positions.push_back(a.root_pos + e * (b.root_pos - a.root_pos));
        m->anchor_positions.push_back(a.anchor_pos + e * (b.anchor_pos - a.anchor_pos));
        m->root_quaternions.push_back(a.root_quat.slerp(e, b.root_quat));
        m->anchor_quaternions.push_back(a.anchor_quat.slerp(e, b.anchor_quat));
        m->dof_positions.push_back(a.q + e * (b.q - a.q));
    }
    const int N = static_cast<int>(m->dof_positions.size());
    m->dof_velocities.resize(N);
    m->anchor_lin_velocities.resize(N);
    m->anchor_ang_velocities.resize(N);
    for (int i = 0; i < N; ++i) {
        const int i0 = std::max(0, i - 1), i1 = std::min(N - 1, i + 1);
        const float span = (i1 - i0) * dt;
        Eigen::VectorXf qd = (m->dof_positions[i1] - m->dof_positions[i0]) / span;
        Eigen::Vector3f v = (m->anchor_positions[i1] - m->anchor_positions[i0]) / span;
        const Eigen::AngleAxisf dq(m->anchor_quaternions[i1] * m->anchor_quaternions[i0].conjugate());
        Eigen::Vector3f w = dq.axis() * dq.angle() / span;
        if (carry_velocity) {
            const float k = 1.0f - es[i];
            qd += k * a.qd;
            v += k * a.anchor_v;
            w += k * a.anchor_w;
        }
        m->dof_velocities[i] = qd;
        m->anchor_lin_velocities[i] = v;
        m->anchor_ang_velocities[i] = w;
    }
    m->finalize();
    return m;
}

void HiphiStack::start(Phase phase, bool continue_sequence, const EaseConfig& ease_cfg)
{
    stop();
    ease_cfg_ = ease_cfg;
    phase_steps_ = 0;
    phase_done_ = false;
    stale_steps_ = 0;
    max_joint_err_ = 0.0f;
    articulation_->update();

    if (!continue_sequence) {
        probe_dump();
        if (!motion_library().empty()) {
            auto picked = motion_library().current();
            if (picked->num_joints != static_cast<int>(limit_lo_.size())) {
                spdlog::error("HiphiStack: clip '{}' has {} joints, stack expects {}; keeping the previous clip",
                              motion_library().current_name(), picked->num_joints, limit_lo_.size());
            } else {
                clip_ = picked;
                time_range_[0] = cfg_["time_start"] ? std::clamp(cfg_["time_start"].as<float>(), 0.0f, clip_->duration) : 0.0f;
                time_range_[1] = cfg_["time_end"] ? std::clamp(cfg_["time_end"].as<float>(), 0.0f, clip_->duration) : clip_->duration;
                spdlog::info("HiphiStack: playing clip '{}' ({:.2f} s, {} frames)", motion_library().current_name(), clip_->duration, clip_->num_frames);
            }
        }
        clip_->update(time_range_[0]);
        umt::align_clip_to_robot(umt_env_.get(), clip_->anchor_quaternion(), clip_->anchor_position(), align_z_);
        umt_env_->reset();
        student_env_->reset();
        hand_hold_.clear();
        residual_gain_ = 0.0f;
        probe_path_ = fmt::format("/tmp/hiphi_probe_{}.npz", std::time(nullptr));
    }

    const char* name = "?";
    switch (phase) {
        case Phase::EaseIn: {
            name = "EaseIn";
            clip_->update(time_range_[0]);
            const Pose b = clip_pose(clip_, clip_->frame);
            const Pose a = robot_pose_in_clip_frame(b.q);
            synthetic_ = build_transition(a, b, ease_cfg_.duration_s, ease_cfg_.easing, 0.0f, false);
            active_ = synthetic_;
            residual_gain_ = 0.0f;
            residual_gain_target_ = 0.0f;
            const float dq_max = (b.q - a.q).cwiseAbs().maxCoeff();
            spdlog::info("HiphiStack EaseIn: {:.2f} s, {} frames, max joint gap {:.2f} rad, anchor gap [{:.3f} {:.3f} {:.3f}] m",
                         ease_cfg_.duration_s, synthetic_->num_frames, dq_max,
                         (b.anchor_pos - a.anchor_pos).x(), (b.anchor_pos - a.anchor_pos).y(), (b.anchor_pos - a.anchor_pos).z());
            break;
        }
        case Phase::Clip: {
            name = "Clip";
            active_ = clip_;
            if (continue_sequence) student_env_->reset();  // fresh residual history on the clip, UMT continues
            residual_gain_target_ = 1.0f;
            break;
        }
        case Phase::EaseOut: {
            name = "EaseOut";
            clip_->update(time_range_[1]);
            const Pose a = clip_pose(clip_, clip_->frame);
            Pose b = a;
            b.qd.setZero(); b.anchor_v.setZero(); b.anchor_w.setZero();
            if (ease_cfg_.hold_pose == "default") {
                b.q = Eigen::VectorXf::Map(default_pose_.data(), default_pose_.size());
            }
            synthetic_ = build_transition(a, b, ease_cfg_.duration_s, Easing::OutCubic, 1.0f, true);
            active_ = synthetic_;
            residual_gain_target_ = 0.0f;
            // fingers: freeze at the last commanded targets (keeps a grasp)
            {
                std::lock_guard<std::mutex> lock(cmd_mutex_);
                if (have_cmd_) hand_hold_ = hand_cmd_;
            }
            if (hand_hold_.empty()) {
                const auto & L = clip_->layout();
                hand_hold_.resize(L.hand_joint_ids.size());
                for (size_t k = 0; k < L.hand_joint_ids.size(); ++k) hand_hold_[k] = a.q[L.hand_joint_ids[k]];
            }
            spdlog::info("HiphiStack EaseOut: {:.2f} s into hold pose '{}', then hold", ease_cfg_.duration_s, ease_cfg_.hold_pose);
            break;
        }
        case Phase::None: break;
    }
    State_UmtMimic::motion = active_;
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        have_cmd_ = false;
    }
    last_phase_ = phase_;
    phase_ = phase;
    spdlog::info("HiphiStack: start {} ({})", name, continue_sequence ? "continuing" : "fresh");
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void HiphiStack::stop()
{
    if (!running_) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
    last_phase_ = phase_;
    phase_ = Phase::None;
}

void HiphiStack::loop()
{
    using clock = std::chrono::high_resolution_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(umt_env_->step_dt));
    auto sleepTill = clock::now() + dt;
    while (running_) {
        step();
        std::this_thread::sleep_until(sleepTill);
        sleepTill += dt;
    }
}

void HiphiStack::step()
{
    const float dt = umt_env_->step_dt;
    articulation_->update();
    const Phase ph = phase_;
    double t = phase_steps_ * dt;
    if (ph == Phase::Clip) t += time_range_[0];
    active_->update(static_cast<float>(t));

    // residual fade towards the phase's target (avoids a step at hand-over)
    const float rate = ease_cfg_.residual_fade_s > 0 ? dt / ease_cfg_.residual_fade_s : 1.0f;
    if (residual_gain_ < residual_gain_target_) residual_gain_ = std::min(residual_gain_target_, residual_gain_ + rate);
    else if (residual_gain_ > residual_gain_target_) residual_gain_ = std::max(residual_gain_target_, residual_gain_ - rate);

    umt_env_->step();
    student_env_->step();
    compose_and_store(t);
    phase_steps_++;

    if (ph == Phase::EaseIn && t >= ease_cfg_.duration_s) phase_done_ = true;
    if (ph == Phase::Clip && t >= time_range_[1]) phase_done_ = true;

    if (require_streams_) {
        const auto & data = articulation_->data;
        auto depth = unitree_rl::depth_sources().at(depth_sensor_);
        const bool stale = !data.has_odom || data.odom_age_ms > odom_max_age_ms_ ||
                           !depth->received() || depth->age_ms() > depth_max_age_ms_;
        stale_steps_ = stale ? stale_steps_ + 1 : 0;
    }
}

void HiphiStack::compose_and_store(double t)
{
    const auto umt = umt_env_->action_manager->processed_actions();        // SIGMA * pi_umt (29)
    const auto res = student_env_->action_manager->processed_actions();    // scaled residual [body 29 | hand 14]
    const Eigen::VectorXf ref = active_->joint_pos();                      // 43, entity order
    const auto & L = clip_->layout();
    const int n_body = static_cast<int>(L.body_joint_ids.size());
    const int n_hand = static_cast<int>(L.hand_joint_ids.size());
    const float g = residual_gain_;

    std::vector<float> body(n_body), hand(n_hand);
    for (int i = 0; i < n_body; ++i) {
        const int j = L.body_joint_ids[i];
        body[i] = std::clamp(ref[j] + umt[i] + g * res[i], limit_lo_[j], limit_hi_[j]);
    }
    if (phase_ == Phase::EaseOut && static_cast<int>(hand_hold_.size()) == n_hand) {
        hand = hand_hold_;
    } else {
        for (int k = 0; k < n_hand; ++k) {
            const int j = L.hand_joint_ids[k];
            hand[k] = std::clamp(ref[j] + g * res[n_body + k], limit_lo_[j], limit_hi_[j]);
        }
    }
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        body_cmd_ = body;
        hand_cmd_ = hand;
        have_cmd_ = true;
    }

    std::vector<float> q, dq;
    isaaclab::mdp::entity_joint_state(umt_env_.get(), q, dq);
    float err = 0.0f;
    for (int i = 0; i < n_body; ++i) err = std::max(err, std::fabs(q[L.body_joint_ids[i]] - ref[L.body_joint_ids[i]]));
    max_joint_err_ = err;

    probe_t_.push_back(static_cast<float>(t));
    probe_phase_.push_back(static_cast<float>(static_cast<int>(phase_.load())));
    probe_ref_.insert(probe_ref_.end(), ref.data(), ref.data() + ref.size());
    probe_umt_.insert(probe_umt_.end(), umt.begin(), umt.end());
    probe_res_.insert(probe_res_.end(), res.begin(), res.end());
    probe_cmd_.insert(probe_cmd_.end(), body.begin(), body.end());
    probe_cmd_.insert(probe_cmd_.end(), hand.begin(), hand.end());
    for (int i = 0; i < n_body; ++i) probe_q_.push_back(q[L.body_joint_ids[i]]);
    for (int k = 0; k < n_hand; ++k) probe_q_.push_back(q[L.hand_joint_ids[k]]);
}

bool HiphiStack::latest_commands(std::vector<float>& body, std::vector<float>& hand)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    if (!have_cmd_) return false;
    body = body_cmd_;
    hand = hand_cmd_;
    return true;
}

void HiphiStack::set_gains(LowCmd_t& lowcmd) const
{
    for (size_t i = 0; i < umt_env_->robot->data.joint_stiffness.size(); i++) {
        lowcmd.msg_.motor_cmd()[i].kp() = umt_env_->robot->data.joint_stiffness[i];
        lowcmd.msg_.motor_cmd()[i].kd() = umt_env_->robot->data.joint_damping[i];
        lowcmd.msg_.motor_cmd()[i].dq() = 0;
        lowcmd.msg_.motor_cmd()[i].tau() = 0;
    }
}

void HiphiStack::probe_dump()
{
    if (probe_t_.empty()) return;
    const size_t T = probe_t_.size();
    const size_t J = probe_ref_.size() / T, B = probe_umt_.size() / T, A = probe_res_.size() / T;
    cnpy::npz_save(probe_path_, "t", probe_t_.data(), {T}, "w");
    cnpy::npz_save(probe_path_, "phase", probe_phase_.data(), {T}, "a");  // 1 EaseIn, 2 Clip, 3 EaseOut
    cnpy::npz_save(probe_path_, "ref_entity", probe_ref_.data(), {T, J}, "a");
    cnpy::npz_save(probe_path_, "umt_offset", probe_umt_.data(), {T, B}, "a");
    cnpy::npz_save(probe_path_, "residual", probe_res_.data(), {T, A}, "a");
    cnpy::npz_save(probe_path_, "q_cmd", probe_cmd_.data(), {T, A}, "a");
    cnpy::npz_save(probe_path_, "q_meas", probe_q_.data(), {T, A}, "a");
    spdlog::info("Hiphi probe: {} steps -> {}", T, probe_path_);
    probe_t_.clear(); probe_phase_.clear(); probe_ref_.clear(); probe_umt_.clear(); probe_res_.clear(); probe_cmd_.clear(); probe_q_.clear();
}
