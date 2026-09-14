#pragma once

// State-estimate input (msgs/stream_msgs.h, rt/odom_pelvis): the pelvis IMU
// site pose in the world and its site-frame twist, from the simulator today
// and from a robot-side estimator adapter later. The DDS subscription is the
// only transport for now; keep consumers on the OdomSource interface so a
// different one (e.g. zenoh across hosts) can be dropped in.

#include <unitree/dds_wrapper/common/Subscription.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "msgs/stream_msgs.h"

namespace unitree_rl
{

class OdomSource : public unitree::robot::SubscriptionBase<msgs::Odometry>
{
public:
    explicit OdomSource(const std::string& topic = msgs::kOdomTopic, uint32_t timeout_ms = 100)
    : SubscriptionBase<msgs::Odometry>(topic), topic_(topic)
    {
        set_timeout_ms(timeout_ms);
    }

    const std::string& topic() const { return topic_; }

    /// Latest sample; false until the first message has arrived.
    bool sample(msgs::OdomSample& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!received_) return false;
        out = sample_;
        return true;
    }

    bool received() const { return received_; }

    /// Milliseconds since the last message (huge before the first one).
    double age_ms() const
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - last_update_time_).count();
    }

protected:
    void post_communication() override  // called under mutex_
    {
        sample_ = msgs::read_odom(msg_);
        received_ = true;
    }

private:
    std::string topic_;
    msgs::OdomSample sample_;
    std::atomic<bool> received_{false};
};

/// Process-wide slot, set by main() from config; null when no estimate is
/// streamed (the estimator-free UMT policies need none).
inline std::shared_ptr<OdomSource>& odom_source()
{
    static std::shared_ptr<OdomSource> instance;
    return instance;
}

}  // namespace unitree_rl
