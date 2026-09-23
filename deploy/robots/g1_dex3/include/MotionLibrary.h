#pragma once

// Library of deploy motion clips with a UDP command port, so the active clip
// can be picked at run time (scripts/sim_viser_mirror.py "Motion clip"
// dropdown, or `echo -n "clip 1" | nc -u -w0 <host> 9873`). Umt and the
// hiphi stack take `current()` when they (re)start a motion; a selection
// made while a motion runs applies on the next entry.
//
// Config (config.yaml, top level):
//   motions:
//     files: [config/motions/a.npz, ...]   # relative to the project dir
//     dir: config/motions/pnp64            # optional: every *.npz in it, sorted (relative to the project dir)
//     default: 0                           # index or file stem
//     command_port: 9873                   # 0 = no command port
//     bind: 127.0.0.1                      # 0.0.0.0 to accept the viser mirror from another host
//
// Protocol (one datagram per command, ASCII):
//   clip <index|stem>   select        next / prev   step
//   list                reply: "current <index> <stem>\n<stem0>\n<stem1>..."
// Every command is answered with the `list` reply to the sender.
//
// Clips are deploy npz files (scripts/umt_bundle_to_deploy_npz.py): 43-joint
// entity order; the embedded layout_* arrays (root / anchor body index, body
// / hand joint ids) override the defaults, so bare-G1 (30-body) and Dex3
// (46-body) clips coexist.

#include "State_UmtMimic.h"  // MotionLoader_

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class MotionLibrary
{
public:
    using MotionLoader_ = State_UmtMimic::MotionLoader_;

    struct Entry
    {
        std::string name;   // file stem
        std::filesystem::path path;
        std::shared_ptr<MotionLoader_> clip;
    };

    /// Read the `motions` config node and load every clip. A missing node
    /// leaves the library empty (states fall back to their own motion_file).
    void start(const YAML::Node& cfg, const MotionLoader_::Layout& default_layout)
    {
        if (!cfg) return;
        std::vector<std::filesystem::path> files;
        if (cfg["files"]) {
            for (const auto& f : cfg["files"].as<std::vector<std::string>>()) {
                std::filesystem::path p = f;
                files.push_back(p.is_absolute() ? p : param::proj_dir / p);
            }
        }
        if (cfg["dir"]) {
            std::filesystem::path dir = cfg["dir"].as<std::string>();
            if (dir.is_relative()) dir = param::proj_dir / dir;
            std::vector<std::filesystem::path> found;
            for (const auto& e : std::filesystem::directory_iterator(dir)) {
                if (e.path().extension() == ".npz") found.push_back(e.path());
            }
            std::sort(found.begin(), found.end());
            files.insert(files.end(), found.begin(), found.end());
        }
        for (const auto& p : files) {
            Entry e;
            e.name = p.stem().string();
            e.path = p;
            try {
                e.clip = std::make_shared<MotionLoader_>(p.string(), default_layout);
            } catch (const std::exception& ex) {
                spdlog::error("motions: cannot load '{}': {}", p.string(), ex.what());
                continue;
            }
            spdlog::info("motions[{}]: '{}' {} frames ({:.2f} s), {} joints, root body {}, anchor body {}",
                         entries_.size(), e.name, e.clip->num_frames, e.clip->duration, e.clip->num_joints,
                         e.clip->layout().root_body_index, e.clip->layout().anchor_body_index);
            entries_.push_back(std::move(e));
        }
        if (entries_.empty()) {
            spdlog::warn("motions: no clips loaded");
            return;
        }
        if (cfg["default"]) {
            const auto d = cfg["default"];
            try { select(d.as<int>()); }
            catch (...) { select_by_name(d.as<std::string>()); }
        }
        const int port = cfg["command_port"] ? cfg["command_port"].as<int>() : 9873;
        const std::string bind_addr = cfg["bind"] ? cfg["bind"].as<std::string>() : "127.0.0.1";
        if (port > 0) start_command_port(port, bind_addr);
        spdlog::info("motions: {} clips, current '{}'{}", entries_.size(), current_name(),
                     port > 0 ? fmt::format(", commands on udp://{}:{}", bind_addr, port) : "");
    }

    ~MotionLibrary()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0) close(fd_);
    }

    bool empty() const { return entries_.empty(); }
    size_t size() const { return entries_.size(); }

    std::shared_ptr<MotionLoader_> current()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.empty() ? nullptr : entries_[current_].clip;
    }

    std::string current_name()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.empty() ? "" : entries_[current_].name;
    }

    bool select(int index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < 0 || index >= static_cast<int>(entries_.size())) return false;
        if (index != current_) spdlog::info("motions: selected [{}] '{}' (takes effect on the next motion entry)", index, entries_[index].name);
        current_ = index;
        return true;
    }

    bool select_by_name(const std::string& name)
    {
        for (size_t i = 0; i < entries_.size(); i++) {
            if (entries_[i].name == name) return select(static_cast<int>(i));
        }
        spdlog::warn("motions: no clip named '{}'", name);
        return false;
    }

private:
    void start_command_port(int port, const std::string& bind_addr)
    {
        fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = inet_addr(bind_addr.c_str());
        if (bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
            spdlog::error("motions: cannot bind udp://{}:{}", bind_addr, port);
            close(fd_);
            fd_ = -1;
            return;
        }
        timeval tv{0, 200000};
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        running_ = true;
        thread_ = std::thread([this] { loop(); });
    }

    std::string status_reply()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string s = fmt::format("current {} {}\n", current_, entries_.empty() ? "" : entries_[current_].name);
        for (const auto& e : entries_) s += e.name + "\n";
        return s;
    }

    void loop()
    {
        char buf[512];
        while (running_) {
            sockaddr_in from{};
            socklen_t from_len = sizeof(from);
            const ssize_t n = recvfrom(fd_, buf, sizeof(buf) - 1, 0, reinterpret_cast<sockaddr*>(&from), &from_len);
            if (n <= 0) continue;
            std::string cmd(buf, static_cast<size_t>(n));
            while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) cmd.pop_back();
            if (cmd.rfind("clip ", 0) == 0) {
                const std::string arg = cmd.substr(5);
                const bool numeric = !arg.empty() && std::all_of(arg.begin(), arg.end(), [](unsigned char c) { return std::isdigit(c); });
                const bool ok = numeric ? select(std::stoi(arg)) : select_by_name(arg);
                if (!ok) spdlog::warn("motions: bad selection '{}'", arg);
            } else if (cmd == "next" || cmd == "prev") {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!entries_.empty()) {
                    const int n_ = static_cast<int>(entries_.size());
                    current_ = (current_ + (cmd == "next" ? 1 : n_ - 1)) % n_;
                    spdlog::info("motions: selected [{}] '{}' (takes effect on the next motion entry)", current_, entries_[current_].name);
                }
            } else if (cmd != "list") {
                spdlog::warn("motions: unknown command '{}'", cmd);
            }
            const std::string reply = status_reply();
            ::sendto(fd_, reply.data(), reply.size(), MSG_DONTWAIT, reinterpret_cast<const sockaddr*>(&from), from_len);
        }
    }

    std::vector<Entry> entries_;
    int current_ = 0;
    std::mutex mutex_;
    int fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

/// Process-wide library, started from main().
inline MotionLibrary& motion_library()
{
    static MotionLibrary lib;
    return lib;
}
