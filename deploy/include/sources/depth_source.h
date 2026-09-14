#pragma once

// Depth-image input (msgs/stream_msgs.h, rt/depth_camera): an organized
// PointCloud2 of uint16 millimetre planar z-depth, from the simulator today
// and from a robot-side camera node later. Frames are decoded in the DDS
// callback; `latest_normalized()` produces the observation the training
// term `camera_depth` produced — clamp(d, min_depth, cutoff) / cutoff,
// (H, W) row-major with row 0 at the top — without blocking on the callback
// beyond a memcpy.

#include <unitree/dds_wrapper/common/Subscription.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "msgs/stream_msgs.h"

namespace unitree_rl
{

class DepthSource : public unitree::robot::SubscriptionBase<msgs::DepthImage>
{
public:
    explicit DepthSource(const std::string& topic = msgs::kDepthTopic, uint32_t timeout_ms = 200)
    : SubscriptionBase<msgs::DepthImage>(topic), topic_(topic)
    {
        set_timeout_ms(timeout_ms);
    }

    const std::string& topic() const { return topic_; }
    bool received() const { return received_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint64_t seq() const { return seq_; }
    double stamp() const { return stamp_; }

    double age_ms() const
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - last_update_time_).count();
    }

    /// Raw millimetre frame (row 0 = top). False until the first frame.
    bool latest_mm(std::vector<uint16_t>& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!received_) return false;
        out = mm_;
        return true;
    }

    /// Training normalisation of the latest frame into `out` (resized to
    /// height*width). Before the first frame `out` is filled with the value a
    /// no-return pixel maps to (min_depth / cutoff) and false is returned.
    bool latest_normalized(std::vector<float>& out, float cutoff_m, float min_depth_m)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = static_cast<size_t>(width_) * height_;
        out.resize(n);
        if (!received_) {
            std::fill(out.begin(), out.end(), min_depth_m / cutoff_m);
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            const float d = std::clamp(mm_[i] * 1e-3f, min_depth_m, cutoff_m);
            out[i] = std::clamp(d / cutoff_m, 0.0f, 1.0f);
        }
        return true;
    }

    /// Fixes the frame size before the first message so observation vectors
    /// have the right width from construction (must match the stream).
    void set_expected_size(uint32_t width, uint32_t height)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!received_) {
            width_ = width;
            height_ = height;
        }
    }

protected:
    void post_communication() override  // called under mutex_
    {
        if (!msgs::read_depth_image(msg_, mm_)) {
            if (!warned_) {
                spdlog::warn("{}: message is not a depth image in the stream_msgs convention; ignored", topic_);
                warned_ = true;
            }
            return;
        }
        if (received_ && (msg_.width() != width_ || msg_.height() != height_)) {
            spdlog::warn("{}: frame size changed {}x{} -> {}x{}", topic_, width_, height_, msg_.width(), msg_.height());
        }
        width_ = msg_.width();
        height_ = msg_.height();
        stamp_ = msgs::stamp_seconds(msg_.header());
        ++seq_;
        received_ = true;
    }

private:
    std::string topic_;
    std::vector<uint16_t> mm_;
    uint32_t width_ = msgs::kDepthWidth;
    uint32_t height_ = msgs::kDepthHeight;
    double stamp_ = 0.0;
    uint64_t seq_ = 0;
    std::atomic<bool> received_{false};
    bool warned_ = false;
};

/// Named depth sources, looked up by the `camera_depth` observation term's
/// `sensor_name` param. Populated by main() from config.
inline std::map<std::string, std::shared_ptr<DepthSource>>& depth_sources()
{
    static std::map<std::string, std::shared_ptr<DepthSource>> instance;
    return instance;
}

}  // namespace unitree_rl
