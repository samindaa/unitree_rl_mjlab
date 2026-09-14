// Probe for the streamed inputs (deploy/include/msgs/stream_msgs.h):
// subscribes to rt/odom_pelvis and rt/depth_camera, prints rate / age /
// content summaries once a second and optionally dumps everything received
// to an uncompressed npz (cnpy) for scripts/check_depth_stream.py.
//
//   topic_probe [-n iface] [-s seconds] [-d out.npz] [--odom-topic T] [--depth-topic T]

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <boost/program_options.hpp>
#include <cnpy.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include "msgs/stream_msgs.h"

namespace po = boost::program_options;
using clock_t_ = std::chrono::steady_clock;
using namespace unitree_rl::msgs;

struct OdomLog
{
    std::mutex mtx;
    std::vector<double> t, pos, quat, lin, ang;
    std::vector<double> rx_wall;  // receive time, seconds since start
    OdomSample last;
    size_t count = 0;
};

struct DepthLog
{
    std::mutex mtx;
    std::vector<double> t, rx_wall;
    std::vector<uint16_t> frames;  // [F, H, W]
    uint32_t width = 0, height = 0;
    std::vector<uint16_t> last;
    size_t count = 0;
    bool bad_layout = false;
};

static double wall_since(clock_t_::time_point t0)
{
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

int main(int argc, char** argv)
{
    std::string iface, odom_topic = kOdomTopic, depth_topic = kDepthTopic, dump;
    double seconds = 5.0;
    bool keep_frames = false;
    po::options_description desc("topic_probe");
    desc.add_options()
        ("help,h", "help")
        ("network,n", po::value<std::string>(&iface)->default_value(""), "dds network interface")
        ("seconds,s", po::value<double>(&seconds)->default_value(5.0), "run time; 0 = until Ctrl-C")
        ("dump,d", po::value<std::string>(&dump)->default_value(""), "write received data to this .npz")
        ("odom-topic", po::value<std::string>(&odom_topic)->default_value(kOdomTopic), "")
        ("depth-topic", po::value<std::string>(&depth_topic)->default_value(kDepthTopic), "");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
    keep_frames = !dump.empty();

    unitree::robot::ChannelFactory::Instance()->Init(0, iface);
    const auto t0 = clock_t_::now();

    OdomLog odom;
    DepthLog depth;

    unitree::robot::SubscriptionBase<Odometry> odom_sub(odom_topic, [&](const void* raw) {
        const auto s = read_odom(*static_cast<const Odometry*>(raw));
        std::lock_guard<std::mutex> lock(odom.mtx);
        odom.last = s;
        odom.count++;
        odom.rx_wall.push_back(wall_since(t0));
        odom.t.push_back(s.t);
        odom.pos.insert(odom.pos.end(), s.pos, s.pos + 3);
        odom.quat.insert(odom.quat.end(), s.quat, s.quat + 4);
        odom.lin.insert(odom.lin.end(), s.lin_vel_b, s.lin_vel_b + 3);
        odom.ang.insert(odom.ang.end(), s.ang_vel_b, s.ang_vel_b + 3);
    });

    unitree::robot::SubscriptionBase<DepthImage> depth_sub(depth_topic, [&](const void* raw) {
        const auto& msg = *static_cast<const DepthImage*>(raw);
        std::vector<uint16_t> mm;
        if (!read_depth_image(msg, mm)) { depth.bad_layout = true; return; }
        std::lock_guard<std::mutex> lock(depth.mtx);
        depth.width = msg.width();
        depth.height = msg.height();
        depth.last = mm;
        depth.count++;
        depth.rx_wall.push_back(wall_since(t0));
        depth.t.push_back(stamp_seconds(msg.header()));
        if (keep_frames) depth.frames.insert(depth.frames.end(), mm.begin(), mm.end());
    });

    spdlog::info("probe: odom <- {}   depth <- {}   ({}s{})", odom_topic, depth_topic,
                 seconds > 0 ? std::to_string(seconds) : "inf", dump.empty() ? "" : ", dumping to " + dump);

    size_t odom_prev = 0, depth_prev = 0;
    auto next = clock_t_::now() + std::chrono::seconds(1);
    while (seconds <= 0 || wall_since(t0) < seconds) {
        std::this_thread::sleep_until(next);
        next += std::chrono::seconds(1);
        {
            std::lock_guard<std::mutex> lock(odom.mtx);
            const size_t n = odom.count - odom_prev;
            odom_prev = odom.count;
            const double age = odom.rx_wall.empty() ? -1 : wall_since(t0) - odom.rx_wall.back();
            const auto& s = odom.last;
            spdlog::info("odom  {:4d} Hz  age {:6.1f} ms  t {:8.3f}  pos [{:+.3f} {:+.3f} {:+.3f}]  quat(wxyz) [{:+.3f} {:+.3f} {:+.3f} {:+.3f}]  v_b [{:+.3f} {:+.3f} {:+.3f}]  w_b [{:+.3f} {:+.3f} {:+.3f}]",
                         n, age * 1e3, s.t, s.pos[0], s.pos[1], s.pos[2], s.quat[0], s.quat[1], s.quat[2], s.quat[3],
                         s.lin_vel_b[0], s.lin_vel_b[1], s.lin_vel_b[2], s.ang_vel_b[0], s.ang_vel_b[1], s.ang_vel_b[2]);
        }
        {
            std::lock_guard<std::mutex> lock(depth.mtx);
            const size_t n = depth.count - depth_prev;
            depth_prev = depth.count;
            if (depth.bad_layout) spdlog::warn("depth: received a PointCloud2 that is not a stream_msgs depth image");
            if (depth.last.empty()) {
                spdlog::info("depth {:4d} Hz  (no frame yet)", n);
            } else {
                const double age = wall_since(t0) - depth.rx_wall.back();
                size_t valid = 0; uint32_t mn = 65535, mx = 0; double sum = 0;
                for (uint16_t v : depth.last) {
                    if (v == 0) continue;
                    valid++; sum += v; mn = std::min<uint32_t>(mn, v); mx = std::max<uint32_t>(mx, v);
                }
                const size_t W = depth.width, H = depth.height;
                const uint16_t center = depth.last[(H / 2) * W + W / 2];
                const uint16_t top = depth.last[W / 2], bottom = depth.last[(H - 1) * W + W / 2];
                spdlog::info("depth {:4d} Hz  age {:6.1f} ms  t {:8.3f}  {}x{}  valid {:5.1f}%  min {} mean {:.0f} max {} mm  top/center/bottom {} / {} / {} mm",
                             n, age * 1e3, depth.t.back(), W, H, 100.0 * valid / depth.last.size(),
                             valid ? mn : 0, valid ? sum / valid : 0.0, valid ? mx : 0, top, center, bottom);
            }
        }
    }

    if (!dump.empty()) {
        std::lock_guard<std::mutex> lo(odom.mtx);
        std::lock_guard<std::mutex> ld(depth.mtx);
        const size_t N = odom.t.size(), F = depth.t.size();
        bool first = true;
        auto mode = [&]() { const char* m = first ? "w" : "a"; first = false; return m; };
        if (N > 0) {
            cnpy::npz_save(dump, "odom_t", odom.t.data(), {N}, mode());
            cnpy::npz_save(dump, "odom_rx_wall", odom.rx_wall.data(), {N}, mode());
            cnpy::npz_save(dump, "odom_pos", odom.pos.data(), {N, 3}, mode());
            cnpy::npz_save(dump, "odom_quat_wxyz", odom.quat.data(), {N, 4}, mode());
            cnpy::npz_save(dump, "odom_lin_vel_b", odom.lin.data(), {N, 3}, mode());
            cnpy::npz_save(dump, "odom_ang_vel_b", odom.ang.data(), {N, 3}, mode());
        }
        if (F > 0) {
            cnpy::npz_save(dump, "depth_t", depth.t.data(), {F}, mode());
            cnpy::npz_save(dump, "depth_rx_wall", depth.rx_wall.data(), {F}, mode());
            cnpy::npz_save(dump, "depth_mm", depth.frames.data(), {F, depth.height, depth.width}, mode());
        }
        spdlog::info("wrote {} ({} odom samples, {} depth frames)", dump, N, F);
    }
    return 0;
}
