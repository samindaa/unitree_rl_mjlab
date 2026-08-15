#pragma once

// UDP "state tap": sends the full simulator state (sim time + qpos) to a
// localhost port so external visualizers (scripts/sim_viser_mirror.py) can
// render the simulation without joining DDS. The DDS route is not an option
// for Python tooling here: the simulator links cyclonedds 0.10.2, which
// segfaults while parsing the XTypes TypeObjects that newer cyclonedds
// releases (the only ones with Python 3.12 wheels) emit during discovery.
//
// Packet layout, little-endian, one datagram per send:
//   uint32  magic  "MJQP" (0x4D4A5150)
//   uint32  nq     number of qpos entries
//   double  time   mjData.time
//   double  qpos[nq]
//
// The companion CommandTap listens on state_tap_port + 1 for key commands
// (magic "MJKY" + one byte), so the mirror can drive the keyboard joystick,
// the elastic band, and simulation reset remotely.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

class StateTap
{
public:
    static constexpr uint32_t MAGIC = 0x4D4A5150; // "MJQP"

    StateTap(const mjModel *model, int port) : nq_(model->nq)
    {
        fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        addr_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        buf_.resize(2 * sizeof(uint32_t) + sizeof(double) * (1 + nq_));
        const uint32_t magic = MAGIC;
        const uint32_t nq = static_cast<uint32_t>(nq_);
        std::memcpy(buf_.data(), &magic, sizeof(magic));
        std::memcpy(buf_.data() + sizeof(magic), &nq, sizeof(nq));
    }

    ~StateTap()
    {
        if (fd_ >= 0) close(fd_);
    }

    StateTap(const StateTap &) = delete;
    StateTap &operator=(const StateTap &) = delete;

    // Reads qpos without locking, like the rest of the bridge: a torn read is
    // acceptable for visualization and vanishingly rare in practice.
    void send(const mjData *data)
    {
        if (fd_ < 0) return;
        std::memcpy(buf_.data() + 8, &data->time, sizeof(double));
        std::memcpy(buf_.data() + 16, data->qpos, sizeof(double) * nq_);
        ::sendto(fd_, buf_.data(), buf_.size(), MSG_DONTWAIT,
                 reinterpret_cast<const sockaddr *>(&addr_), sizeof(addr_));
    }

private:
    int fd_ = -1;
    int nq_;
    sockaddr_in addr_{};
    std::vector<char> buf_;
};

// Receives single-key commands over UDP and hands them to a handler, mirroring
// what the GLFW window's key callback and the terminal keyboard accept. The
// handler runs on this tap's own thread.
class CommandTap
{
public:
    static constexpr uint32_t MAGIC = 0x4D4A4B59; // "MJKY"

    CommandTap(int port, std::function<void(char)> handler) : handler_(std::move(handler))
    {
        fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(fd_, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            std::perror("CommandTap bind");
            close(fd_);
            fd_ = -1;
            return;
        }
        timeval tv{0, 200000}; // so ~CommandTap() is noticed promptly
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        running_ = true;
        thread_ = std::thread([this] { loop(); });
    }

    ~CommandTap()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0) close(fd_);
    }

    CommandTap(const CommandTap &) = delete;
    CommandTap &operator=(const CommandTap &) = delete;

private:
    void loop()
    {
        char buf[8];
        while (running_)
        {
            const ssize_t n = recv(fd_, buf, sizeof(buf), 0);
            if (n != sizeof(uint32_t) + 1) continue;
            uint32_t magic;
            std::memcpy(&magic, buf, sizeof(magic));
            if (magic != MAGIC) continue;
            handler_(buf[sizeof(magic)]);
        }
    }

    std::function<void(char)> handler_;
    int fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
};
