#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>


/**
 * Synthesises a Unitree gamepad from terminal key presses, so the controller FSM
 * can be driven without a physical joystick on local ("lo") simulation runs.
 *
 * A terminal reports one key at a time, so a chord such as `LT + up` cannot be
 * typed literally. Each chord is therefore bound to a single key: pressing it
 * asserts every button of the chord for `press_duration` seconds, which the
 * controller's extract() sees as one clean rising edge (`.on_pressed`).
 *
 * LT and RT are Axis, not Button, so the controller's extract() low-passes them
 * (Axis::smooth = 0.03) and they need ~23 updates to cross the 0.5 press
 * threshold. A digital `.on_pressed` edge lasts exactly one update, so asserting
 * a whole chord at once means `LT + up.on_pressed` never sees both halves true on
 * the same tick. The triggers are therefore asserted `trigger_lead` seconds ahead
 * of the rest of the chord -- the same way a human holds L2 before tapping Up.
 *
 * The velocity axes are sticky rather than momentary -- `w` raises the forward
 * command and it stays there, which is what a velocity policy expects.
 */
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    KeyboardJoystick(const std::map<std::string, std::string> & chord_map,
                     double press_duration = 0.3, double axis_step = 0.1,
                     double trigger_lead = 0.15)
    : press_duration_(press_duration), axis_step_(axis_step),
      trigger_lead_(toDuration(trigger_lead)), hold_(toDuration(press_duration))
    {
        for(const auto & entry : chord_map)
        {
            if(entry.first.size() != 1) {
                std::cerr << "Error: keyboard chord key \"" << entry.first
                          << "\" must be a single character." << std::endl;
                exit(EXIT_FAILURE);
            }
            chords_[entry.first[0]] = Chord{parseChord(entry.second), entry.second};
        }

        // These axes carry digital values, so bypass the smoothing filter: without
        // this an asserted LT would need ~23 updates to cross the 0.5 threshold that
        // combine() uses. The controller applies its own smoothing on extract().
        for(auto * axis : {&lx, &ly, &rx, &ry, &LT, &RT}) {
            axis->smooth = 1.0;
        }

        enterRawMode();
        printHelp();

        running_ = true;
        read_thread_ = std::thread([this]{ readLoop(); });
        active() = this;
    }

    ~KeyboardJoystick()
    {
        active() = nullptr;
        running_ = false;
        if(read_thread_.joinable()) { read_thread_.join(); }
        restoreTerminal();
    }

    /// Feed one key as if it had been typed in the terminal (used by the
    /// command tap so scripts/sim_viser_mirror.py can drive the FSM). Safe to
    /// call from any thread.
    void inject(char c) { onKey(c); }

    /// The single live instance, for command routing; nullptr when the
    /// keyboard joystick is not in use.
    static KeyboardJoystick*& active()
    {
        static KeyboardJoystick* instance = nullptr;
        return instance;
    }

    void update() override
    {
        const auto now = Clock::now();

        bool held[kNumButtons];
        float lx_cmd, ly_cmd, rx_cmd, ry_cmd;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for(int i(0); i<kNumButtons; i++) {
                held[i] = (now >= press_at_[i]) && (now < release_at_[i]);
            }
            lx_cmd = lx_cmd_;
            ly_cmd = ly_cmd_;
            rx_cmd = rx_cmd_;
            ry_cmd = ry_cmd_;
        }

        back(held[kBack]);
        start(held[kStart]);
        LB(held[kLB]);
        RB(held[kRB]);
        A(held[kA]);
        B(held[kB]);
        X(held[kX]);
        Y(held[kY]);
        up(held[kUp]);
        down(held[kDown]);
        left(held[kLeft]);
        right(held[kRight]);
        F1(held[kF1]);
        F2(held[kF2]);
        LT(held[kLT] ? 1.f : 0.f);
        RT(held[kRT] ? 1.f : 0.f);
        lx(lx_cmd);
        ly(ly_cmd);
        rx(rx_cmd);
        ry(ry_cmd);
    }

    /**
     * @brief Put the terminal back in canonical mode.
     *
     * Async-signal-safe, and also registered with atexit(), so the shell is never
     * left without an echo after Ctrl-C.
     */
    static void restoreTerminal()
    {
        if(terminalModified().exchange(false)) {
            tcsetattr(fileno(stdin), TCSANOW, &savedTermios());
        }
    }

private:
    // Only the buttons that combine() actually transmits over wireless_remote.
    enum ButtonId {
        kBack, kStart, kLB, kRB, kA, kB, kX, kY,
        kUp, kDown, kLeft, kRight, kF1, kF2, kLT, kRT, kNumButtons
    };

    struct Chord
    {
        std::vector<int> buttons;
        std::string text;
    };

    using Clock = std::chrono::steady_clock;

    static Clock::duration toDuration(double seconds)
    {
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    }

    // LT/RT reach the controller as Axis values and are low-pass filtered there.
    static bool isTrigger(int id) { return id == kLT || id == kRT; }

    /**
     * @brief Map a button name to its id, accepting both the DSL spelling used in
     * the controller config (LT, RB) and the labels printed by the controller (L2, R1).
     */
    static int buttonId(std::string name)
    {
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

        static const std::unordered_map<std::string, int> kMap = {
            {"back", kBack}, {"select", kBack}, {"start", kStart},
            {"lb", kLB}, {"l1", kLB}, {"rb", kRB}, {"r1", kRB},
            {"lt", kLT}, {"l2", kLT}, {"rt", kRT}, {"r2", kRT},
            {"a", kA}, {"b", kB}, {"x", kX}, {"y", kY},
            {"up", kUp}, {"down", kDown}, {"left", kLeft}, {"right", kRight},
            {"f1", kF1}, {"f2", kF2},
        };

        auto it = kMap.find(name);
        return it == kMap.end() ? -1 : it->second;
    }

    static std::vector<int> parseChord(const std::string & chord)
    {
        std::vector<int> ids;
        std::stringstream ss(chord);
        std::string token;

        while(std::getline(ss, token, '+'))
        {
            const auto first = token.find_first_not_of(" \t");
            if(first == std::string::npos) { continue; }
            token = token.substr(first, token.find_last_not_of(" \t") - first + 1);

            int id = buttonId(token);
            if(id < 0) {
                std::cerr << "Error: unknown button \"" << token << "\" in keyboard chord \""
                          << chord << "\"." << std::endl;
                exit(EXIT_FAILURE);
            }
            ids.push_back(id);
        }

        if(ids.empty()) {
            std::cerr << "Error: empty keyboard chord." << std::endl;
            exit(EXIT_FAILURE);
        }
        return ids;
    }

    void readLoop()
    {
        while(running_)
        {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fileno(stdin), &fds);

            timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 50000; // so ~KeyboardJoystick() is noticed promptly

            if(select(fileno(stdin)+1, &fds, NULL, NULL, &tv) <= 0) { continue; }

            char c = '\0';
            if(read(fileno(stdin), &c, 1) != 1) { continue; }

            if(c == '\033') // arrow keys, aliased onto the translation axes
            {
                char seq = '\0';
                if(read(fileno(stdin), &seq, 1) != 1 || seq != '[') { continue; }
                if(read(fileno(stdin), &seq, 1) != 1) { continue; }
                switch(seq)
                {
                case 'A': c = 'w'; break;
                case 'B': c = 's'; break;
                case 'C': c = 'd'; break;
                case 'D': c = 'a'; break;
                default: continue;
                }
            }

            onKey(c);
        }
    }

    void onKey(char c)
    {
        auto chord = chords_.find(c);
        if(chord != chords_.end())
        {
            const auto now = Clock::now();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for(int id : chord->second.buttons)
                {
                    // LT/RT lead so they are above threshold by the time the digital
                    // buttons produce their one-tick .on_pressed edge.
                    const auto lead = isTrigger(id) ? Clock::duration::zero() : trigger_lead_;

                    // Only open a new window if one is not already running, otherwise
                    // auto-repeat would keep pushing the digital press further away.
                    if(now >= release_at_[id]) { press_at_[id] = now + lead; }
                    release_at_[id] = now + lead + hold_;
                }
            }
            std::cout << "[keyboard] " << chord->second.text << std::endl;
            return;
        }

        float vx, vy, wz;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            switch(c)
            {
            case 'w': ly_cmd_ += axis_step_; break;
            case 's': ly_cmd_ -= axis_step_; break;
            case 'a': lx_cmd_ -= axis_step_; break; // lin_vel_y is read as -lx
            case 'd': lx_cmd_ += axis_step_; break;
            case 'q': rx_cmd_ -= axis_step_; break; // ang_vel_z is read as -rx
            case 'e': rx_cmd_ += axis_step_; break;
            case ' ': lx_cmd_ = ly_cmd_ = rx_cmd_ = ry_cmd_ = 0.f; break;
            default: return;
            }

            lx_cmd_ = std::clamp(lx_cmd_, -1.f, 1.f);
            ly_cmd_ = std::clamp(ly_cmd_, -1.f, 1.f);
            rx_cmd_ = std::clamp(rx_cmd_, -1.f, 1.f);
            ry_cmd_ = std::clamp(ry_cmd_, -1.f, 1.f);

            vx = ly_cmd_;
            vy = -lx_cmd_;
            wz = -rx_cmd_;
        }

        printf("[keyboard] vx=%+.2f  vy=%+.2f  wz=%+.2f\n", vx, vy, wz);
        fflush(stdout);
    }

    void printHelp()
    {
        std::cout << "\n--- Keyboard joystick (no gamepad required) ---\n";
        for(const auto & entry : chords_) {
            std::cout << "  [" << entry.first << "]      " << entry.second.text << "\n";
        }
        std::cout << "  [w/s]    forward / backward\n"
                     "  [a/d]    strafe left / right\n"
                     "  [q/e]    turn left / right\n"
                     "  [space]  zero all velocity commands\n"
                     "-----------------------------------------------\n" << std::endl;
    }

    void enterRawMode()
    {
        if(!isatty(fileno(stdin))) {
            std::cerr << "Error: joystick_type \"keyboard\" needs a terminal on stdin." << std::endl;
            exit(EXIT_FAILURE);
        }

        tcgetattr(fileno(stdin), &savedTermios());

        termios raw = savedTermios();
        raw.c_lflag &= ~(ICANON | ECHO); // ISIG is kept, so Ctrl-C still works
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(fileno(stdin), TCSANOW, &raw);
        terminalModified() = true;

        std::atexit(&KeyboardJoystick::restoreTerminal);
        installSignalHandler(SIGINT);
        installSignalHandler(SIGTERM);
    }

    static void installSignalHandler(int sig)
    {
        struct sigaction current;
        if(sigaction(sig, nullptr, &current) != 0) { return; }
        if(current.sa_handler != SIG_DFL) { return; } // never stomp an existing handler

        struct sigaction sa;
        sa.sa_handler = &KeyboardJoystick::onSignal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(sig, &sa, nullptr);
    }

    static void onSignal(int sig)
    {
        restoreTerminal();
        signal(sig, SIG_DFL);
        raise(sig);
    }

    static termios & savedTermios()
    {
        static termios saved{};
        return saved;
    }

    static std::atomic<bool> & terminalModified()
    {
        static std::atomic<bool> modified{false};
        return modified;
    }

    const double press_duration_;
    const float axis_step_;
    const Clock::duration trigger_lead_;
    const Clock::duration hold_;

    std::map<char, Chord> chords_;
    Clock::time_point press_at_[kNumButtons];
    Clock::time_point release_at_[kNumButtons];

    float lx_cmd_{0.f}, ly_cmd_{0.f}, rx_cmd_{0.f}, ry_cmd_{0.f};

    std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::thread read_thread_;
};
