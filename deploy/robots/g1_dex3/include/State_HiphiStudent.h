#pragma once

// Depth-distilled hiphi student on the frozen UMT base (smp_v2 task
// `Hiphi-Tracking-Multi-V2-Distill-G1`, tasks/hiphi_tracking_multi_distill +
// tasks/hiphi_tracking_umt/actions.py UmtResidualAction).
//
// Two networks run every policy step (50 Hz) on the same robot state:
//
//   UMT base (160 obs -> 29):  zest_ref (hands masked to the default pose) |
//       motion_anchor_pos_b | motion_anchor_ori_b | base_lin_vel |
//       base_ang_vel | projected_gravity | joint_pos_rel (29) |
//       joint_vel_rel (29) | its own last action (29)
//   student (196 + 1xHxW depth -> 43):  zest_ref (unmasked) |
//       motion_anchor_ori_b | base_ang_vel | projected_gravity |
//       joint_pos_rel over all 43 joints (entity order, fingers interleaved) |
//       joint_vel_rel (43) | its own last action (43) ; camera_depth
//
// and the targets are composed as in training:
//
//   body (29, SDK order):  q = clamp(q_ref + SIGMA * pi_umt + res_scale * a[0:29], joint limits)
//   hand (14, Dex3 order): q = clamp(q_ref_hand + hand_scale * a[29:43], joint limits)
//
// The stack itself (envs, clip, alignment, policy thread) lives in
// HiphiStack, shared with the eased entry / exit states:
//
//   Velocity --RB+B--> HiphiEaseIn --(blend done)--> HiphiStudent --(clip end)--> HiphiEaseOut (hold)
//
// This state runs the clip phase. Entered from HiphiEaseIn it continues the
// stack (UMT history, alignment, hand targets); entered directly it starts
// fresh (aligns the clip to the robot and jumps to frame 0).

#include "FSM/State_RLBase.h"
#include "HiphiStack.h"

class State_HiphiBase : public FSMState
{
public:
    State_HiphiBase(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string), stack_(HiphiStack::instance())
    {
        stack_.configure(param::config["FSM"]["HiphiStudent"]);
        this->registered_checks.emplace_back(std::make_pair(
            [&]()->bool{
                if (!stack_.streams_stale()) return false;
                spdlog::warn("{}: state estimate / depth stream stale, leaving", getStateString());
                return true;
            },
            FSMStringMap.right.at("Velocity")));
    }

    void run()
    {
        std::vector<float> body, hand;
        if (!stack_.latest_commands(body, hand)) return;  // hold the previous targets until the first step
        const auto & ids = stack_.joint_ids_map();
        for (size_t i = 0; i < ids.size() && i < body.size(); i++) {
            lowcmd->msg_.motor_cmd()[ids[i]].q() = body[i];
        }
        if (dex3_hands().enabled()) {
            dex3_hands().set_targets(hand.data(), static_cast<int>(hand.size()));
        }
    }

protected:
    HiphiStack& stack_;
};


class State_HiphiStudent : public State_HiphiBase
{
public:
    State_HiphiStudent(int state_mode, std::string state_string)
    : State_HiphiBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        end_state_ = cfg["end_state"] ? cfg["end_state"].as<std::string>() : "HiphiEaseOut";
        fade_ = parse_ease_config(cfg, HiphiStack::Easing::InCubic);
        this->registered_checks.emplace_back(std::make_pair(
            [&]()->bool{ return stack_.phase_done(); },
            FSMStringMap.right.at(end_state_)));
    }

    void enter()
    {
        stack_.set_gains(*lowcmd);
        const bool cont = stack_.last_phase() == HiphiStack::Phase::EaseIn && stack_.phase_done();
        stack_.start(HiphiStack::Phase::Clip, cont, fade_);
    }

    void exit()
    {
        const bool to_ease_out = stack_.phase_done();
        stack_.stop();
        if (!to_ease_out) {
            dex3_hands().set_open_pose();
            stack_.probe_dump();
        }
    }

private:
    std::string end_state_;
    HiphiStack::EaseConfig fade_;
};

REGISTER_FSM(State_HiphiStudent)
