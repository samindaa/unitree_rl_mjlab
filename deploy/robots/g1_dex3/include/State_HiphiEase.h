#pragma once

// Eased entry into / held exit from the hiphi motion (see HiphiStack.h):
//
//   HiphiEaseIn   synthetic reference from the robot's measured pose into the
//                 clip's first frame (ease_in_cubic over duration_s, UMT base
//                 only), then hands over to HiphiStudent automatically.
//   HiphiEaseOut  synthetic reference from the clip's last frame into the
//                 hold pose (ease_out_cubic), then holds it with zero
//                 reference velocities until the operator leaves (LT+B ->
//                 Passive, RT+A -> Velocity). Fingers stay at their last
//                 targets, so a grasp survives.
//
// Config (config.yaml FSM.<state>): duration_s, ease (in_cubic |
// in_out_cubic), residual_fade_s, hold_pose (last | default, EaseOut),
// abort_joint_err (rad, EaseIn -> Passive when tracking breaks down).

#include "State_HiphiStudent.h"

class State_HiphiEaseIn : public State_HiphiBase
{
public:
    State_HiphiEaseIn(int state_mode, std::string state_string)
    : State_HiphiBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        ease_ = parse_ease_config(cfg, HiphiStack::Easing::InCubic);
        abort_joint_err_ = cfg["abort_joint_err"] ? cfg["abort_joint_err"].as<float>() : 0.6f;
        this->registered_checks.emplace_back(std::make_pair(
            [&]()->bool{ return stack_.phase_done(); },
            FSMStringMap.right.at("HiphiStudent")));
        this->registered_checks.emplace_back(std::make_pair(
            [&]()->bool{
                if (stack_.max_joint_error() <= abort_joint_err_) return false;
                spdlog::warn("HiphiEaseIn: joint tracking error {:.2f} rad > {:.2f}, aborting", stack_.max_joint_error(), abort_joint_err_);
                return true;
            },
            FSMStringMap.right.at("Passive")));
    }

    void enter()
    {
        stack_.set_gains(*lowcmd);
        stack_.start(HiphiStack::Phase::EaseIn, false, ease_);
    }

    void exit()
    {
        const bool to_student = stack_.phase_done();
        stack_.stop();
        if (!to_student) {
            dex3_hands().set_open_pose();
            stack_.probe_dump();
        }
    }

private:
    HiphiStack::EaseConfig ease_;
    float abort_joint_err_ = 0.6f;
};

REGISTER_FSM(State_HiphiEaseIn)


class State_HiphiEaseOut : public State_HiphiBase
{
public:
    State_HiphiEaseOut(int state_mode, std::string state_string)
    : State_HiphiBase(state_mode, state_string)
    {
        ease_ = parse_ease_config(param::config["FSM"][state_string], HiphiStack::Easing::OutCubic);
    }

    void enter()
    {
        stack_.set_gains(*lowcmd);
        const bool cont = stack_.last_phase() == HiphiStack::Phase::Clip;
        stack_.start(HiphiStack::Phase::EaseOut, cont, ease_);
    }

    void exit()
    {
        stack_.stop();
        dex3_hands().set_open_pose();
        stack_.probe_dump();
    }

private:
    HiphiStack::EaseConfig ease_;
};

REGISTER_FSM(State_HiphiEaseOut)
