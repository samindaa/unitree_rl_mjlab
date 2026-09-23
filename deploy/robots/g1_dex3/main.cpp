#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_UmtMimic.h"
#include "State_HiphiStudent.h"
#include "State_HiphiEase.h"
#include "Dex3Hands.h"
#include "sources/odom_source.h"
#include "sources/depth_source.h"
#include "MotionLibrary.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

// Streamed inputs (msgs/stream_msgs.h): the state estimate feeds
// ArticulationData (base_lin_vel / motion_anchor_pos_b), the depth cameras
// feed the camera_depth observation. Created before the FSM so states whose
// deploy.yaml uses those terms find them at construction.
void init_sources(const YAML::Node& cfg)
{
    if (!cfg) return;
    if (auto odom = cfg["odom"]; odom && odom["enable"].as<bool>(false)) {
        const auto topic = odom["topic"].as<std::string>(unitree_rl::msgs::kOdomTopic);
        const auto timeout = odom["timeout_ms"].as<uint32_t>(100);
        unitree_rl::odom_source() = std::make_shared<unitree_rl::OdomSource>(topic, timeout);
        spdlog::info("Source odom: {} (timeout {} ms)", topic, timeout);
    }
    if (auto depth = cfg["depth"]) {
        for (auto it = depth.begin(); it != depth.end(); ++it) {
            const auto name = it->first.as<std::string>();
            const auto node = it->second;
            if (!node["enable"].as<bool>(true)) continue;
            const auto topic = node["topic"].as<std::string>(unitree_rl::msgs::kDepthTopic);
            const auto timeout = node["timeout_ms"].as<uint32_t>(200);
            auto src = std::make_shared<unitree_rl::DepthSource>(topic, timeout);
            src->set_expected_size(node["width"].as<uint32_t>(unitree_rl::msgs::kDepthWidth),
                                   node["height"].as<uint32_t>(unitree_rl::msgs::kDepthHeight));
            unitree_rl::depth_sources()[name] = src;
            spdlog::info("Source depth '{}': {} {}x{} (timeout {} ms)", name, topic, src->width(), src->height(), timeout);
        }
    }
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof + Dex3 Controller (UMT tracking) \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // The body is the 29dof G1 on rt/lowcmd; the Dex3 hands live on their own
    // DDS topics and are driven by the Dex3Hands publisher (see Dex3Hands.h).
    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    dex3_hands().start(param::config["dex3"]);
    init_sources(param::config["sources"]);
    motion_library().start(param::config["motions"], State_UmtMimic::MotionLoader_::Layout());

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    std::cout << "And then press [R1 + A] to start the UMT motion, [R1 + B] the hiphi student (eased in, held at the end).\n";

    while (true)
    {
        sleep(1);
    }

    return 0;
}
