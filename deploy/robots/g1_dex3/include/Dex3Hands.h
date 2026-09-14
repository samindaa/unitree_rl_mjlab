#pragma once

// Dex3-1 hand command driver for the g1_dex3 controller.
//
// The Dex3 hands are not part of rt/lowcmd: each hand listens on its own DDS
// topic, rt/dex3/<side>/cmd (unitree_hg::msg::dds_::HandCmd_, 7 MotorCmd_ in
// Dex3 motor order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0,
// index_1), and reports on rt/dex3/<side>/state. That motor order is also the
// order of the 7 finger joints per hand in the deploy clips (entity ids 22-28
// left, 36-42 right, see State_UmtMimic.h), so reference values are passed
// through unchanged; right-hand values are already expressed in the right
// hand's mirrored joint ranges.
//
// A background thread publishes the current 14 targets with fixed PD gains at
// `publish_hz`, independent of the FSM state, so the hands always hold a pose:
// the rest pose from config (`open_pose`; default all joints 0 = the hand's
// qpos0, but config.yaml folds the thumbs so they clear the thighs at the
// stand pose) until an FSM state overrides it. State_UmtMimic /
// State_HiphiStudent stream the clip's finger reference while active and
// restore the rest pose on exit.
// Both the real hands and the simulator (simulate/src/unitree_sdk2_bridge.h)
// compute tau + kp (q - q_meas) + kd (dq - dq_meas) from these commands.
//
// The driver also subscribes to rt/dex3/<side>/state and exposes the 14
// measured finger positions / velocities (`joint_state`, same order) for
// policies whose proprioception covers the hands (hiphi student, 43 joints).
//
// Config (config.yaml, top level):
//   dex3:
//     enable: true
//     kp: 1.5          # avp_teleoperate Dex3_1_Controller gains
//     kd: 0.2
//     publish_hz: 100
//     open_pose: [14 floats]   # optional, default all 0

#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Dex3Hands
{
public:
    static constexpr int kMotorsPerHand = 7;
    static constexpr int kNumMotors = 2 * kMotorsPerHand; // left, then right
    using Targets = std::array<float, kNumMotors>;

    Dex3Hands() = default;
    ~Dex3Hands() { stop(); }
    Dex3Hands(const Dex3Hands&) = delete;
    Dex3Hands& operator=(const Dex3Hands&) = delete;

    /// Read the `dex3` config node and start publishing. A missing node or
    /// `enable: false` leaves the driver inert (every call becomes a no-op).
    void start(const YAML::Node& cfg)
    {
        if (!cfg || (cfg["enable"] && !cfg["enable"].as<bool>())) {
            spdlog::info("Dex3 hands: disabled (no hand commands will be sent)");
            return;
        }
        kp_ = cfg["kp"] ? cfg["kp"].as<float>() : 1.5f;
        kd_ = cfg["kd"] ? cfg["kd"].as<float>() : 0.2f;
        publish_hz_ = cfg["publish_hz"] ? cfg["publish_hz"].as<double>() : 100.0;
        open_pose_.fill(0.0f);
        if (cfg["open_pose"]) {
            auto v = cfg["open_pose"].as<std::vector<float>>();
            if (v.size() != kNumMotors) {
                throw std::runtime_error("dex3.open_pose must have 14 entries (left hand, then right)");
            }
            std::copy(v.begin(), v.end(), open_pose_.begin());
        }
        targets_ = open_pose_;

        left_ = std::make_unique<Publisher>("left");
        right_ = std::make_unique<Publisher>("right");
        left_state_ = std::make_unique<StateSub>("rt/dex3/left/state");
        right_state_ = std::make_unique<StateSub>("rt/dex3/right/state");
        running_ = true;
        thread_ = std::thread([this] { loop(); });
        spdlog::info("Dex3 hands: publishing rt/dex3/{{left,right}}/cmd at {} Hz, kp {} kd {}",
                     publish_hz_, kp_, kd_);
    }

    void stop()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    bool enabled() const { return running_; }

    /// Set the 14 joint targets (left hand motors 0-6, then right hand).
    void set_targets(const float* q, int n)
    {
        if (!running_) return;
        if (n != kNumMotors) {
            spdlog::error("Dex3 hands: got {} targets, expected {}", n, kNumMotors);
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::copy(q, q + n, targets_.begin());
    }

    void set_open_pose()
    {
        if (!running_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        targets_ = open_pose_;
    }

    const Targets& open_pose() const { return open_pose_; }

    /// Measured finger joint positions and velocities (left motors 0-6, then
    /// right), from rt/dex3/<side>/state. Zeros (and false) until both hands
    /// have reported; stale hands keep their last value.
    bool joint_state(Targets& q, Targets& dq)
    {
        q.fill(0.0f);
        dq.fill(0.0f);
        if (!running_) return false;
        bool ok = true;
        for (int side = 0; side < 2; side++) {
            auto& sub = side == 0 ? *left_state_ : *right_state_;
            std::lock_guard<std::mutex> lock(sub.mutex_);
            const auto& motors = sub.msg_.motor_state();
            if (motors.size() < static_cast<size_t>(kMotorsPerHand) || sub.isTimeout()) {
                ok = false;
                continue;
            }
            for (int i = 0; i < kMotorsPerHand; i++) {
                q[side * kMotorsPerHand + i] = motors[i].q();
                dq[side * kMotorsPerHand + i] = motors[i].dq();
            }
        }
        return ok;
    }

private:
    using StateSub = unitree::robot::SubscriptionBase<unitree_hg::msg::dds_::HandState_>;

    class Publisher : public unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::HandCmd_>
    {
    public:
        explicit Publisher(const std::string& side)
        : RealTimePublisher<MsgType>("rt/dex3/" + side + "/cmd")
        {
            msg_.motor_cmd().resize(kMotorsPerHand);
            for (int i = 0; i < kMotorsPerHand; i++) {
                // RIS mode byte (unitree_sdk2 example/g1/dex3): id in bits 0-3,
                // status 1 (enabled) in bits 4-6, timeout flag 0 in bit 7.
                const uint8_t id = static_cast<uint8_t>(i) & 0x0F;
                const uint8_t status = 0x01;
                const uint8_t timeout = 0x00;
                msg_.motor_cmd()[i].mode() = id | (status << 4) | (timeout << 7);
                msg_.motor_cmd()[i].q() = 0.0f;
                msg_.motor_cmd()[i].dq() = 0.0f;
                msg_.motor_cmd()[i].tau() = 0.0f;
            }
        }
    };

    void write(Publisher& pub, const float* q)
    {
        if (!pub.trylock()) return; // publisher thread still copying the last message
        for (int i = 0; i < kMotorsPerHand; i++) {
            auto& m = pub.msg_.motor_cmd()[i];
            m.q() = q[i];
            m.kp() = kp_;
            m.kd() = kd_;
        }
        pub.unlockAndPublish();
    }

    void loop()
    {
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / publish_hz_));
        auto next = clock::now();
        while (running_) {
            Targets q;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                q = targets_;
            }
            write(*left_, q.data());
            write(*right_, q.data() + kMotorsPerHand);
            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    float kp_ = 1.5f;
    float kd_ = 0.2f;
    double publish_hz_ = 100.0;
    Targets open_pose_{};
    Targets targets_{};
    std::mutex mutex_;

    std::unique_ptr<Publisher> left_;
    std::unique_ptr<Publisher> right_;
    std::unique_ptr<StateSub> left_state_;
    std::unique_ptr<StateSub> right_state_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

/// Process-wide driver, started from main() and used by the FSM states.
inline Dex3Hands& dex3_hands()
{
    static Dex3Hands hands;
    return hands;
}
