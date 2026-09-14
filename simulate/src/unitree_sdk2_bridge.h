#pragma once

#include <mujoco/mujoco.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/go2/go2.h>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>

#include "msgs/stream_msgs.h"

#include <iostream>
#include <string>
#include <vector>

#include "param.h"
#include "physics_joystick.h"
#include "keyboard_joystick.h"
#include "state_tap.h"

#define MOTOR_SENSOR_NUM 3

class UnitreeSDK2BridgeBase
{
public:
    UnitreeSDK2BridgeBase(mjModel *model, mjData *data)
    : mj_model_(model), mj_data_(data)
    {
        _check_sensor();
        if(param::config.print_scene_information == 1) {
            printSceneInformation();
        }
        if(param::config.use_joystick == 1) {
            if(param::config.joystick_type == "xbox") {
                joystick = std::make_shared<XBoxJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else if(param::config.joystick_type == "switch") {
                joystick  = std::make_shared<SwitchJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else if(param::config.joystick_type == "keyboard") {
                joystick = std::make_shared<KeyboardJoystick>(param::config.keyboard_map,
                                                              param::config.keyboard_press_duration,
                                                              param::config.keyboard_axis_step,
                                                              param::config.keyboard_trigger_lead);
            } else {
                std::cerr << "Unsupported joystick type: " << param::config.joystick_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if(param::config.state_tap_port > 0) {
            state_tap = std::make_unique<StateTap>(model, param::config.state_tap_port);
        }
    }

    virtual void start() {}

    void printSceneInformation()
    {
        auto printObjects = [this](const char* title, int count, int type, auto getIndex) {
            std::cout << "<<------------- " << title << " ------------->> " << std::endl;
            for (int i = 0; i < count; i++) {
                const char* name = mj_id2name(mj_model_, type, i);
                if (name) {
                    std::cout << title << "_index: " << getIndex(i) << ", " << "name: " << name;
                    if (type == mjOBJ_SENSOR) {
                        std::cout << ", dim: " << mj_model_->sensor_dim[i];
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        };
    
        printObjects("Link", mj_model_->nbody, mjOBJ_BODY, [](int i) { return i; });
        printObjects("Joint", mj_model_->njnt, mjOBJ_JOINT, [](int i) { return i; });
        printObjects("Actuator", mj_model_->nu, mjOBJ_ACTUATOR, [](int i) { return i; });
    
        int sensorIndex = 0;
        printObjects("Sensor", mj_model_->nsensor, mjOBJ_SENSOR, [&](int i) {
            int currentIndex = sensorIndex;
            sensorIndex += mj_model_->sensor_dim[i];
            return currentIndex;
        });
    }

protected:
    // All actuators. The scene's sensor list starts with three blocks of
    // num_motor_ entries (jointpos, jointvel, jointactuatorfrc) in actuator
    // order, so motor i reads sensordata[i], [i + num_motor_], [i + 2*num_motor_].
    int num_motor_ = 0;
    int dim_motor_sensor_ = 0;
    // Actuators driven through rt/lowcmd / reported on rt/lowstate: the
    // leading actuators that are not Dex3 hand motors. Dex3 hand motors
    // (scene_g1_dex3.xml: actuators named left_hand_* / right_hand_*, placed
    // after the body motors) go through the rt/dex3/<side>/{cmd,state} topics
    // instead — LowCmd only has 35 motor slots and the real hands live on
    // those topics too.
    int num_body_motor_ = 0;
    std::vector<int> left_hand_motor_ids_;
    std::vector<int> right_hand_motor_ids_;

    mjData *mj_data_;
    mjModel *mj_model_;

    // Sensor data indices
    int imu_quat_adr_ = -1;
    int imu_gyro_adr_ = -1;
    int imu_acc_adr_ = -1;
    int frame_pos_adr_ = -1;
    int frame_vel_adr_ = -1;

    int secondary_imu_quat_adr_ = -1;
    int secondary_imu_gyro_adr_ = -1;
    int secondary_imu_acc_adr_ = -1;

    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;
    std::unique_ptr<StateTap> state_tap = nullptr;
    int state_tap_counter_ = 0;

    void _check_sensor()
    {
        num_motor_ = mj_model_->nu;
        dim_motor_sensor_ = MOTOR_SENSOR_NUM * num_motor_;

        // Split off the Dex3 hand actuators by name.
        num_body_motor_ = 0;
        for (int i = 0; i < num_motor_; i++) {
            const char* cname = mj_id2name(mj_model_, mjOBJ_ACTUATOR, i);
            const std::string name = cname ? cname : "";
            if (name.rfind("left_hand_", 0) == 0) {
                left_hand_motor_ids_.push_back(i);
            } else if (name.rfind("right_hand_", 0) == 0) {
                right_hand_motor_ids_.push_back(i);
            } else {
                if (!left_hand_motor_ids_.empty() || !right_hand_motor_ids_.empty()) {
                    std::cerr << "Body actuator '" << name << "' listed after a Dex3 hand actuator; "
                              << "hand actuators must come last (see scripts/make_g1_dex3_scene.py)" << std::endl;
                    exit(EXIT_FAILURE);
                }
                num_body_motor_++;
            }
        }
        if (!left_hand_motor_ids_.empty() || !right_hand_motor_ids_.empty()) {
            std::cout << "Dex3 hands: " << num_body_motor_ << " body motors on rt/lowcmd, "
                      << left_hand_motor_ids_.size() << " left + " << right_hand_motor_ids_.size()
                      << " right hand motors on rt/dex3/<side>/cmd" << std::endl;
        }

        // Find sensor addresses by name
        int sensor_id = -1;
        
        // IMU quaternion
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_quat");
        if (sensor_id >= 0) {
            imu_quat_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // IMU gyroscope
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_gyro");
        if (sensor_id >= 0) {
            imu_gyro_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // IMU accelerometer
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_acc");
        if (sensor_id >= 0) {
            imu_acc_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // Frame position
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_pos");
        if (sensor_id >= 0) {
            frame_pos_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // Frame velocity
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_vel");
        if (sensor_id >= 0) {
            frame_vel_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Secondary IMU quaternion
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_quat");
        if (sensor_id >= 0) {
            secondary_imu_quat_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Secondary IMU gyroscope
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_gyro");
        if (sensor_id >= 0) {
            secondary_imu_gyro_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Secondary IMU accelerometer
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_acc");
        if (sensor_id >= 0) {
            secondary_imu_acc_adr_ = mj_model_->sensor_adr[sensor_id];
        }
    }
};

template <typename LowCmd_t, typename LowState_t>
class RobotBridge : public UnitreeSDK2BridgeBase
{
using HighState_t = unitree::robot::go2::publisher::SportModeState;
using WirelessController_t = unitree::robot::go2::publisher::WirelessController;

public:
    RobotBridge(mjModel *model, mjData *data) : UnitreeSDK2BridgeBase(model, data)
    {
        lowcmd = std::make_shared<LowCmd_t>("rt/lowcmd");
        lowstate = std::make_unique<LowState_t>();
        lowstate->joystick = joystick;
        highstate = std::make_unique<HighState_t>();
        wireless_controller = std::make_unique<WirelessController_t>();
        wireless_controller->joystick = joystick;
    }

    void start()
    {
        thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "unitree_bridge", UT_CPU_ID_NONE, 1000, [this]() { this->run(); });
    }

    virtual void run()
    {
        if(!mj_data_) return;
        if(lowstate->joystick) { lowstate->joystick->update(); }
        // lowcmd
        {
            std::lock_guard<std::mutex> lock(lowcmd->mutex_);
            for(int i(0); i<num_body_motor_; i++) {
                auto & m = lowcmd->msg_.motor_cmd()[i];
                mj_data_->ctrl[i] = m.tau() +
                                    m.kp() * (m.q() - mj_data_->sensordata[i]) +
                                    m.kd() * (m.dq() - mj_data_->sensordata[i + num_motor_]);
            }
        }

        // lowstate
        if(lowstate->trylock()) {
            for(int i(0); i<num_body_motor_; i++) {
                lowstate->msg_.motor_state()[i].q() = mj_data_->sensordata[i];
                lowstate->msg_.motor_state()[i].dq() = mj_data_->sensordata[i + num_motor_];
                lowstate->msg_.motor_state()[i].tau_est() = mj_data_->sensordata[i + 2 * num_motor_];
            }
            
            if(imu_quat_adr_ >= 0) {
                lowstate->msg_.imu_state().quaternion()[0] = mj_data_->sensordata[imu_quat_adr_ + 0];
                lowstate->msg_.imu_state().quaternion()[1] = mj_data_->sensordata[imu_quat_adr_ + 1];
                lowstate->msg_.imu_state().quaternion()[2] = mj_data_->sensordata[imu_quat_adr_ + 2];
                lowstate->msg_.imu_state().quaternion()[3] = mj_data_->sensordata[imu_quat_adr_ + 3];

                double w = lowstate->msg_.imu_state().quaternion()[0];
                double x = lowstate->msg_.imu_state().quaternion()[1];
                double y = lowstate->msg_.imu_state().quaternion()[2];
                double z = lowstate->msg_.imu_state().quaternion()[3];

                lowstate->msg_.imu_state().rpy()[0] = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                lowstate->msg_.imu_state().rpy()[1] = asin(2 * (w * y - z * x));
                lowstate->msg_.imu_state().rpy()[2] = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            }
            
            if(imu_gyro_adr_ >= 0) {
                lowstate->msg_.imu_state().gyroscope()[0] = mj_data_->sensordata[imu_gyro_adr_ + 0];
                lowstate->msg_.imu_state().gyroscope()[1] = mj_data_->sensordata[imu_gyro_adr_ + 1];
                lowstate->msg_.imu_state().gyroscope()[2] = mj_data_->sensordata[imu_gyro_adr_ + 2];
            }

            if(imu_acc_adr_ >= 0) {
                lowstate->msg_.imu_state().accelerometer()[0] = mj_data_->sensordata[imu_acc_adr_ + 0];
                lowstate->msg_.imu_state().accelerometer()[1] = mj_data_->sensordata[imu_acc_adr_ + 1];
                lowstate->msg_.imu_state().accelerometer()[2] = mj_data_->sensordata[imu_acc_adr_ + 2];
            }
            
            lowstate->msg_.tick() = std::round(mj_data_->time / 1e-3);
            lowstate->unlockAndPublish();
        }
        // highstate
        if(highstate->trylock()) {
            if(frame_pos_adr_ >= 0) {
                highstate->msg_.position()[0] = mj_data_->sensordata[frame_pos_adr_ + 0];
                highstate->msg_.position()[1] = mj_data_->sensordata[frame_pos_adr_ + 1];
                highstate->msg_.position()[2] = mj_data_->sensordata[frame_pos_adr_ + 2];
            }
            if(frame_vel_adr_ >= 0) {
                highstate->msg_.velocity()[0] = mj_data_->sensordata[frame_vel_adr_ + 0];
                highstate->msg_.velocity()[1] = mj_data_->sensordata[frame_vel_adr_ + 1];
                highstate->msg_.velocity()[2] = mj_data_->sensordata[frame_vel_adr_ + 2];
            }
            highstate->unlockAndPublish();
        }
        // wireless_controller
        if(wireless_controller->joystick) {
            wireless_controller->unlockAndPublish();
        }
        // state tap for external visualizers, at 1/10th of the bridge rate
        if(state_tap && ++state_tap_counter_ >= 10) {
            state_tap_counter_ = 0;
            state_tap->send(mj_data_);
        }
    }

    std::unique_ptr<HighState_t> highstate;
    std::unique_ptr<WirelessController_t> wireless_controller;
    std::shared_ptr<LowCmd_t> lowcmd;
    std::unique_ptr<LowState_t> lowstate;
    
private:
    unitree::common::RecurrentThreadPtr thread_;
};

using Go2Bridge = RobotBridge<unitree::robot::go2::subscription::LowCmd, unitree::robot::go2::publisher::LowState>;

class G1Bridge : public RobotBridge<unitree::robot::g1::subscription::LowCmd, unitree::robot::g1::publisher::LowState>
{
public:
    G1Bridge(mjModel *model, mjData *data) : RobotBridge(model, data)
    {
        if (param::config.robot.find("g1") != std::string::npos) {
            auto* g1_lowstate = dynamic_cast<unitree::robot::g1::publisher::LowState*>(lowstate.get());
            if (g1_lowstate) {
                auto scene = param::config.robot_scene.filename().string();
                g1_lowstate->msg_.mode_machine() = scene.find("23") != std::string::npos ? 4 : 5;
            }
        }

        bmsstate = std::make_unique<BmsState_t>("rt/lf/bmsstate");
        bmsstate->msg_.soc() = 100;

        secondary_imustate = std::make_unique<IMUState_t>("rt/secondary_imu");

        left_hand_.init(mj_model_, "left", left_hand_motor_ids_);
        right_hand_.init(mj_model_, "right", right_hand_motor_ids_);

        if (param::config.odom_enable) {
            odom_site_id_ = mj_name2id(mj_model_, mjOBJ_SITE, param::config.odom_site.c_str());
            if (odom_site_id_ < 0) {
                std::cerr << "odom: site '" << param::config.odom_site
                          << "' not in the scene; state estimate stream disabled" << std::endl;
            } else {
                odom_pub_ = std::make_unique<OdomPub_t>(param::config.odom_topic);
                std::cout << "odom: " << param::config.odom_topic << " <- site '" << param::config.odom_site
                          << "' at " << 1000 / param::config.odom_divider << " Hz" << std::endl;
            }
        }
    }

    void run() override
    {
        RobotBridge::run();

        run_hand(left_hand_);
        run_hand(right_hand_);
        run_odom();

        // secondary IMU state
        if (secondary_imustate->trylock()) {
            if(secondary_imu_quat_adr_ >= 0) {
                secondary_imustate->msg_.quaternion()[0] = mj_data_->sensordata[secondary_imu_quat_adr_ + 0];
                secondary_imustate->msg_.quaternion()[1] = mj_data_->sensordata[secondary_imu_quat_adr_ + 1];
                secondary_imustate->msg_.quaternion()[2] = mj_data_->sensordata[secondary_imu_quat_adr_ + 2];
                secondary_imustate->msg_.quaternion()[3] = mj_data_->sensordata[secondary_imu_quat_adr_ + 3];

                double w = secondary_imustate->msg_.quaternion()[0];
                double x = secondary_imustate->msg_.quaternion()[1];
                double y = secondary_imustate->msg_.quaternion()[2];
                double z = secondary_imustate->msg_.quaternion()[3];

                secondary_imustate->msg_.rpy()[0] = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                secondary_imustate->msg_.rpy()[1] = asin(2 * (w * y - z * x));
                secondary_imustate->msg_.rpy()[2] = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            }

            if(secondary_imu_gyro_adr_ >= 0) {
                secondary_imustate->msg_.gyroscope()[0] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 0];
                secondary_imustate->msg_.gyroscope()[1] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 1];
                secondary_imustate->msg_.gyroscope()[2] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 2];
            }

            if(secondary_imu_acc_adr_ >= 0) {
                secondary_imustate->msg_.accelerometer()[0] = mj_data_->sensordata[secondary_imu_acc_adr_ + 0];
                secondary_imustate->msg_.accelerometer()[1] = mj_data_->sensordata[secondary_imu_acc_adr_ + 1];
                secondary_imustate->msg_.accelerometer()[2] = mj_data_->sensordata[secondary_imu_acc_adr_ + 2];
            }

            secondary_imustate->unlockAndPublish();
        }

        // In practice, bmsstate is sent at a low frequency; here it is sent with the main loop
        bmsstate->unlockAndPublish();
    }

    using BmsState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::BmsState_>;
    using IMUState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::IMUState_>;
//    using BmsState_t = unitree::robot::RealTimePublisher<unitree_go::msg::dds_::BmsState_>;
//    using IMUState_t = unitree::robot::RealTimePublisher<unitree_go::msg::dds_::IMUState_>;
    std::unique_ptr<BmsState_t> bmsstate;
    std::unique_ptr<IMUState_t> secondary_imustate;

    /**
     * One Dex3-1 hand, mirroring the real hand's DDS interface:
     *   rt/dex3/<side>/cmd    HandCmd_   (7 MotorCmd_: mode, q, dq, tau, kp, kd)
     *   rt/dex3/<side>/state  HandState_ (7 MotorState_, 9 PressSensorState_)
     * Motor i of the message is actuator motor_ids[i] of the scene (Dex3 SDK
     * order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0,
     * index_1).
     *
     * Unlike the body (bridge-side PD on <motor> actuators), the hand
     * actuators are MuJoCo <position> servos (scripts/make_g1_dex3_scene.py):
     * the command's q goes to ctrl and its kp/kd are written into the servo's
     * gain/bias parameters, so MuJoCo evaluates the finger PD at every
     * physics step. A bridge-side PD held across the multi-step bursts of the
     * physics loop is unstable for the fingers' ~3e-4 kg.m^2 at kd 0.2. The
     * command's dq and tau, and the MotorCmd_ mode byte (motor id / enable /
     * timeout bits on the real hand), are ignored. When no command has
     * arrived for 1 s the servo gains are zeroed and the fingers go limp,
     * like an unconnected controller. Inactive when the scene has no hand
     * actuators (plain scene_g1.xml).
     */
    using HandCmdSub_t = unitree::robot::SubscriptionBase<unitree_hg::msg::dds_::HandCmd_>;
    using HandStatePub_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::HandState_>;
    static constexpr size_t kDex3PressSensors = 9;

    struct Dex3Hand
    {
        std::vector<int> motor_ids;
        std::unique_ptr<HandCmdSub_t> cmd;
        std::unique_ptr<HandStatePub_t> state;

        void init(const mjModel* model, const std::string& side, const std::vector<int>& ids)
        {
            motor_ids = ids;
            if (motor_ids.empty()) return;
            for (int a : motor_ids) {
                if (model->actuator_gaintype[a] != mjGAIN_FIXED || model->actuator_biastype[a] != mjBIAS_AFFINE) {
                    std::cerr << "Dex3 hand actuator '" << mj_id2name(model, mjOBJ_ACTUATOR, a)
                              << "' must be a <position> servo (see scripts/make_g1_dex3_scene.py)" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            cmd = std::make_unique<HandCmdSub_t>("rt/dex3/" + side + "/cmd");
            state = std::make_unique<HandStatePub_t>("rt/dex3/" + side + "/state");
            state->msg_.motor_state().resize(motor_ids.size());
            state->msg_.press_sensor_state().resize(kDex3PressSensors);
        }
    };

    void run_hand(Dex3Hand& hand)
    {
        if (hand.motor_ids.empty()) return;
        const size_t n = hand.motor_ids.size();
        {
            std::lock_guard<std::mutex> lock(hand.cmd->mutex_);
            const auto& cmds = hand.cmd->msg_.motor_cmd();
            const bool live = !hand.cmd->isTimeout() && cmds.size() >= n;
            for (size_t i = 0; i < n; i++) {
                const int a = hand.motor_ids[i];
                const double kp = live ? cmds[i].kp() : 0.0;
                const double kd = live ? cmds[i].kd() : 0.0;
                // <position> servo: force = kp * ctrl - kp * q - kd * dq
                mj_model_->actuator_gainprm[a * mjNGAIN + 0] = kp;
                mj_model_->actuator_biasprm[a * mjNBIAS + 1] = -kp;
                mj_model_->actuator_biasprm[a * mjNBIAS + 2] = -kd;
                mj_data_->ctrl[a] = live ? cmds[i].q() : 0.0;
            }
        }
        if (hand.state->trylock()) {
            auto& motors = hand.state->msg_.motor_state();
            for (size_t i = 0; i < n; i++) {
                const int a = hand.motor_ids[i];
                motors[i].q() = mj_data_->sensordata[a];
                motors[i].dq() = mj_data_->sensordata[a + num_motor_];
                motors[i].tau_est() = mj_data_->sensordata[a + 2 * num_motor_];
            }
            hand.state->unlockAndPublish();
        }
    }

    Dex3Hand left_hand_;
    Dex3Hand right_hand_;

    /**
     * State estimate stream (msgs/stream_msgs.h): the ground-truth pose of the
     * pelvis IMU site in the world and its twist in the site frame, i.e. what
     * a perfect estimator fed by that IMU would report. Taken straight from
     * mjData (site_xpos / site_xmat / mj_objectVelocity), not from the scene's
     * frame_* sensors, which sit on the `imu` site at the pelvis origin — the
     * training sensors (mjlab imu_lin_vel / imu_ang_vel) are on imu_in_pelvis
     * and the two differ by omega x r.
     */
    using OdomPub_t = unitree::robot::RealTimePublisher<unitree_rl::msgs::Odometry>;
    std::unique_ptr<OdomPub_t> odom_pub_;
    int odom_site_id_ = -1;
    int odom_counter_ = 0;

    void run_odom()
    {
        if (!odom_pub_) return;
        if (++odom_counter_ < param::config.odom_divider) return;
        odom_counter_ = 0;
        if (!odom_pub_->trylock()) return;

        unitree_rl::msgs::OdomSample s;
        s.t = mj_data_->time;
        const mjtNum* p = mj_data_->site_xpos + 3 * odom_site_id_;
        s.pos[0] = p[0]; s.pos[1] = p[1]; s.pos[2] = p[2];
        mjtNum q[4];
        mju_mat2Quat(q, mj_data_->site_xmat + 9 * odom_site_id_);
        for (int i = 0; i < 4; i++) s.quat[i] = q[i];
        mjtNum vel[6];  // [angular, linear], site frame
        mj_objectVelocity(mj_model_, mj_data_, mjOBJ_SITE, odom_site_id_, vel, /*flg_local=*/1);
        for (int i = 0; i < 3; i++) {
            s.ang_vel_b[i] = vel[i];
            s.lin_vel_b[i] = vel[3 + i];
        }
        unitree_rl::msgs::write_odom(odom_pub_->msg_, s);
        odom_pub_->unlockAndPublish();
    }
};
