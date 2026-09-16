#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <filesystem>
#include <map>
#include <string>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    // Only used when joystick_type == "keyboard"
    std::map<std::string, std::string> keyboard_map;
    double keyboard_press_duration = 0.3;
    double keyboard_axis_step = 0.1;
    double keyboard_trigger_lead = 0.15;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    // UDP port for the state tap consumed by scripts/sim_viser_mirror.py;
    // 0 disables it.
    int state_tap_port = 9870;

    // State estimate stream (msgs/stream_msgs.h): pose of `odom_site` in the
    // world and its site-frame twist, published every `odom_divider` bridge
    // ticks (bridge runs at 1 kHz).
    int odom_enable = 1;
    std::string odom_topic = "rt/odom_pelvis";
    std::string odom_site = "imu_in_pelvis";
    int odom_divider = 2;
    // State-estimator randomization (smp_v2 EstimatorNoiseCfg, applied to
    // the published pose/twist): per-"episode" uniform bias bounds and
    // per-message Gaussian noise, position in the site (pelvis) frame,
    // velocity in the site frame. All 0 = clean.
    struct OdomRandomizeConfig
    {
        double pos_bias_m = 0.0;
        double vel_bias = 0.0;
        double pos_noise_m = 0.0;
        double vel_noise = 0.0;
        double bias_redraw_s = 0.0;   // 0 = draw the bias once at start
        unsigned seed = 0;            // 0 = random
    } odom_randomize;

    // Depth camera stream (simulate/src/depth_camera.h). Silently inactive
    // when the scene has no camera named `camera_name`.
    struct DepthCameraConfig
    {
        int enable = 1;
        std::string gl = "egl";      // "egl" (GPU device, headless-capable) or "glfw" (hidden window)
        std::string camera_name = "depth_camera";
        std::string topic = "rt/depth_camera";
        int width = 64;
        int height = 36;
        double hz = 30.0;
        double delay_ms = 0.0;       // sim2sim latency injection
        std::string dump_dir = "";   // write every `dump_every`-th frame as 16-bit PNG + sidecar
        int dump_every = 0;
        // Depth randomization (smp_v2 DepthRandomizationCfg / RandomizedCameraDepth,
        // LadderMan recipe): shift -> noise -> dropout, per published frame.
        struct RandomizeConfig
        {
            double noise_std_m = 0.0;   // per-frame Gaussian noise on depth
            double dropout_p = 0.0;     // per-pixel probability of reading 0 (no return)
            int shift_px = 0;           // per-"episode" integer image shift bound
            double shift_redraw_s = 0.0;  // 0 = draw the shift once at start
            unsigned seed = 0;          // 0 = random
        } randomize;
    } depth_camera;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();

            // Optional; defaults above apply when absent.
            if(cfg["keyboard_map"]) {
                keyboard_map = cfg["keyboard_map"].as<std::map<std::string, std::string>>();
            }
            if(cfg["keyboard_press_duration"]) {
                keyboard_press_duration = cfg["keyboard_press_duration"].as<double>();
            }
            if(cfg["keyboard_axis_step"]) {
                keyboard_axis_step = cfg["keyboard_axis_step"].as<double>();
            }
            if(cfg["keyboard_trigger_lead"]) {
                keyboard_trigger_lead = cfg["keyboard_trigger_lead"].as<double>();
            }
            if(cfg["state_tap_port"]) {
                state_tap_port = cfg["state_tap_port"].as<int>();
            }
            if(cfg["odom_enable"]) odom_enable = cfg["odom_enable"].as<int>();
            if(cfg["odom_topic"]) odom_topic = cfg["odom_topic"].as<std::string>();
            if(cfg["odom_site"]) odom_site = cfg["odom_site"].as<std::string>();
            if(cfg["odom_divider"]) odom_divider = std::max(1, cfg["odom_divider"].as<int>());
            if(auto r = cfg["odom_randomize"]) {
                if(r["pos_bias_m"]) odom_randomize.pos_bias_m = r["pos_bias_m"].as<double>();
                if(r["vel_bias"]) odom_randomize.vel_bias = r["vel_bias"].as<double>();
                if(r["pos_noise_m"]) odom_randomize.pos_noise_m = r["pos_noise_m"].as<double>();
                if(r["vel_noise"]) odom_randomize.vel_noise = r["vel_noise"].as<double>();
                if(r["bias_redraw_s"]) odom_randomize.bias_redraw_s = r["bias_redraw_s"].as<double>();
                if(r["seed"]) odom_randomize.seed = r["seed"].as<unsigned>();
            }
            if(auto dc = cfg["depth_camera"]) {
                if(dc["enable"]) depth_camera.enable = dc["enable"].as<int>();
                if(dc["gl"]) depth_camera.gl = dc["gl"].as<std::string>();
                if(dc["camera_name"]) depth_camera.camera_name = dc["camera_name"].as<std::string>();
                if(dc["topic"]) depth_camera.topic = dc["topic"].as<std::string>();
                if(dc["width"]) depth_camera.width = dc["width"].as<int>();
                if(dc["height"]) depth_camera.height = dc["height"].as<int>();
                if(dc["hz"]) depth_camera.hz = dc["hz"].as<double>();
                if(dc["delay_ms"]) depth_camera.delay_ms = dc["delay_ms"].as<double>();
                if(dc["dump_dir"]) depth_camera.dump_dir = dc["dump_dir"].as<std::string>();
                if(dc["dump_every"]) depth_camera.dump_every = dc["dump_every"].as<int>();
                if(auto r = dc["randomize"]) {
                    if(r["noise_std_m"]) depth_camera.randomize.noise_std_m = r["noise_std_m"].as<double>();
                    if(r["dropout_p"]) depth_camera.randomize.dropout_p = r["dropout_p"].as<double>();
                    if(r["shift_px"]) depth_camera.randomize.shift_px = r["shift_px"].as<int>();
                    if(r["shift_redraw_s"]) depth_camera.randomize.shift_redraw_s = r["shift_redraw_s"].as<double>();
                    if(r["seed"]) depth_camera.randomize.seed = r["seed"].as<unsigned>();
                }
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}