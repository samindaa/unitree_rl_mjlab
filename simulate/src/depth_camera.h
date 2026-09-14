#pragma once

// Depth camera stream for the simulator (msgs/stream_msgs.h, rt/depth_camera).
//
// Renders the scene's fixed camera `camera_name` (added to torso_link by
// scripts/make_g1_dex3_scene.py with the training camera's pose / fovy) off
// screen on its own thread and publishes planar z-depth in millimetres as an
// organized PointCloud2_. Convention details that matter for parity with the
// training renderer (mujoco_warp) are in stream_msgs.h; the ones enforced
// here:
//   * background (no geometry) is published as 0, not the far plane;
//   * row 0 is the top of the image (mjr_readPixels is bottom-up, so rows are
//     flipped);
//   * only visual geom groups 0-2 are rendered (the training sensor's
//     enabled_geom_groups), no shadows / reflections / skybox.
//
// GL plumbing: the thread owns its own GL context and mjrContext / mjvScene,
// so the UI thread's renderer is untouched and the vendored simulate.cc stays
// as is. Default backend is EGL on the first GPU device (EGL_EXT_platform_
// device, what MuJoCo's own Python renderer does): hardware GL regardless of
// the window system — GLFW on a Wayland session lands on Mesa/llvmpipe, where
// this 64x36 render takes ~70 ms instead of ~0.2 ms. The alternative backend
// is a hidden GLFW window created on the main thread (GLFW requires that).
// mjv_updateScene runs under the simulator's model mutex; the render itself
// does not.
//
// `delay_ms` holds frames back before publishing — the same sim2sim latency
// knob as the controller's action_delay_ms, for camera-latency sweeps.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <unitree/dds_wrapper/common/Publisher.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lodepng.h"
#include "msgs/stream_msgs.h"

// Offscreen EGL context on a GPU device (falls back to the default display).
class EglOffscreenContext
{
public:
    bool init()
    {
        auto queryDevices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
        auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
        EGLint major = 0, minor = 0;
        if (queryDevices && getPlatformDisplay) {
            EGLDeviceEXT devices[16];
            EGLint n = 0;
            if (queryDevices(16, devices, &n)) {
                for (EGLint i = 0; i < n && display_ == EGL_NO_DISPLAY; i++) {
                    EGLDisplay d = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    if (d != EGL_NO_DISPLAY && eglInitialize(d, &major, &minor)) display_ = d;
                }
            }
        }
        if (display_ == EGL_NO_DISPLAY) {
            EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (d != EGL_NO_DISPLAY && eglInitialize(d, &major, &minor)) display_ = d;
        }
        if (display_ == EGL_NO_DISPLAY) return false;

        const EGLint cfg_attribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8, EGL_NONE};
        EGLConfig config;
        EGLint ncfg = 0;
        if (!eglChooseConfig(display_, cfg_attribs, &config, 1, &ncfg) || ncfg < 1) return false;
        if (!eglBindAPI(EGL_OPENGL_API)) return false;
        const EGLint pb_attribs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
        surface_ = eglCreatePbufferSurface(display_, config, pb_attribs);
        if (surface_ == EGL_NO_SURFACE) return false;
        context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, nullptr);
        return context_ != EGL_NO_CONTEXT;
    }

    bool make_current() { return eglMakeCurrent(display_, surface_, surface_, context_) == EGL_TRUE; }
    void release() { if (display_ != EGL_NO_DISPLAY) eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); }

    std::string describe() const
    {
        if (display_ == EGL_NO_DISPLAY) return "none";
        const char* vendor = eglQueryString(display_, EGL_VENDOR);
        const char* version = eglQueryString(display_, EGL_VERSION);
        return std::string(vendor ? vendor : "?") + " " + (version ? version : "?");
    }

    ~EglOffscreenContext()
    {
        if (display_ == EGL_NO_DISPLAY) return;
        release();
        if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
        if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
        eglTerminate(display_);
    }

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLContext context_ = EGL_NO_CONTEXT;
};

class DepthCameraStreamer
{
public:
    struct Config
    {
        std::string camera_name = "depth_camera";
        std::string topic = "rt/depth_camera";
        int width = 64;
        int height = 36;
        double hz = 30.0;
        double delay_ms = 0.0;
        std::string dump_dir;
        int dump_every = 0;
    };

    // `model` / `data` are the simulator's (reloadable) globals; `mtx` is the
    // simulator's model mutex. `window` is a hidden GLFW window for the glfw
    // backend, nullptr for EGL. The publisher is created in start(), which
    // must run after the DDS ChannelFactory is initialised.
    DepthCameraStreamer(GLFWwindow* window, mjModel** model, mjData** data,
                        std::recursive_mutex* mtx, Config cfg)
    : window_(window), model_(model), data_(data), mtx_(mtx), cfg_(std::move(cfg))
    {
        mjv_defaultOption(&opt_);
        for (int g = 0; g < mjNGROUP; g++) opt_.geomgroup[g] = (g <= 2);
        mjv_defaultCamera(&cam_);
        mjv_defaultScene(&scn_);
        mjr_defaultContext(&con_);
        depth_.resize(static_cast<size_t>(cfg_.width) * cfg_.height);
        depth_mm_.resize(depth_.size());
    }

    ~DepthCameraStreamer()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    void start()
    {
        pub_ = std::make_unique<Pub_t>(cfg_.topic);
        unitree_rl::msgs::init_depth_image(pub_->msg_, cfg_.width, cfg_.height);
        running_ = true;
        thread_ = std::thread([this] { loop(); });
    }

private:
    using Pub_t = unitree::robot::RealTimePublisher<unitree_rl::msgs::DepthImage>;
    using clock = std::chrono::steady_clock;

    struct Pending
    {
        clock::time_point release;
        double t;
        std::vector<uint16_t> mm;
    };

    void loop()
    {
        if (window_) {
            glfwMakeContextCurrent(window_);
            std::cout << "depth_camera: GL via hidden GLFW window" << std::endl;
        } else {
            if (!egl_.init() || !egl_.make_current()) {
                std::cerr << "depth_camera: EGL context creation failed; depth stream inactive "
                             "(try depth_camera.gl: glfw)" << std::endl;
                return;
            }
            std::cout << "depth_camera: GL via EGL (" << egl_.describe() << ")" << std::endl;
            // MuJoCo prints "OpenGL error 0x502 in or before mjr_makeContext" once
            // on this context; the rendered depth validates against the Python
            // renderer to <1 mm (scripts/check_depth_stream.py), so it is benign.
        }
        const auto period = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / cfg_.hz));
        auto next = clock::now();
        while (running_) {
            mjModel* m = *model_;
            mjData* d = *data_;
            if (m && d && ensure_context(m)) {
                render(m, d);
            }
            flush_pending();
            next += period;
            std::this_thread::sleep_until(next);
        }
        release_context();
        if (window_) glfwMakeContextCurrent(nullptr);
        else egl_.release();
    }

    // (Re)build the render context when the model (re)loads.
    bool ensure_context(mjModel* m)
    {
        if (m == ctx_model_) return cam_id_ >= 0;
        release_context();
        ctx_model_ = m;
        cam_id_ = mj_name2id(m, mjOBJ_CAMERA, cfg_.camera_name.c_str());
        if (cam_id_ < 0) {
            std::cerr << "depth_camera: no camera '" << cfg_.camera_name
                      << "' in the scene; depth stream inactive" << std::endl;
            return false;
        }
        mjv_makeScene(m, &scn_, 2000);
        scn_.flags[mjRND_SHADOW] = 0;
        scn_.flags[mjRND_REFLECTION] = 0;
        scn_.flags[mjRND_SKYBOX] = 0;
        scn_.flags[mjRND_FOG] = 0;
        scn_.flags[mjRND_HAZE] = 0;
        // No multisampling for this context: MSAA resolve averages depth
        // across silhouettes (phantom values between fore- and background)
        // and the full-buffer resolve dominates mjr_readPixels.
        const int offsamples = m->vis.quality.offsamples;
        m->vis.quality.offsamples = 0;
        mjr_makeContext(m, &con_, mjFONTSCALE_100);
        m->vis.quality.offsamples = offsamples;
        mjr_setBuffer(mjFB_OFFSCREEN, &con_);
        if (con_.currentBuffer != mjFB_OFFSCREEN) {
            std::cerr << "depth_camera: offscreen framebuffer unavailable; depth stream inactive" << std::endl;
            cam_id_ = -1;
            return false;
        }
        if (con_.offWidth < cfg_.width || con_.offHeight < cfg_.height) {
            std::cerr << "depth_camera: offscreen buffer " << con_.offWidth << "x" << con_.offHeight
                      << " smaller than " << cfg_.width << "x" << cfg_.height
                      << " (raise <visual><global offwidth/offheight>)" << std::endl;
            cam_id_ = -1;
            return false;
        }
        cam_.type = mjCAMERA_FIXED;
        cam_.fixedcamid = cam_id_;
        znear_ = m->vis.map.znear * m->stat.extent;
        zfar_ = m->vis.map.zfar * m->stat.extent;
        std::cout << "depth_camera: '" << cfg_.camera_name << "' " << cfg_.width << "x" << cfg_.height
                  << " @ " << cfg_.hz << " Hz -> " << cfg_.topic
                  << " (fovy " << m->cam_fovy[cam_id_] << ", clip " << znear_ << ".." << zfar_ << " m"
                  << (cfg_.delay_ms > 0 ? ", delay " + std::to_string(cfg_.delay_ms) + " ms" : "")
                  << ")" << std::endl;
        return true;
    }

    void release_context()
    {
        if (!ctx_model_) return;
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);
        mjr_defaultContext(&con_);
        mjv_defaultScene(&scn_);
        ctx_model_ = nullptr;
        cam_id_ = -1;
    }

    void render(mjModel* m, mjData* d)
    {
        double t;
        const auto t_start = clock::now();
        {
            std::lock_guard<std::recursive_mutex> lock(*mtx_);
            stat_lock_ += ms_since(t_start);
            mjv_updateScene(m, d, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
            t = d->time;
            if (cfg_.dump_every > 0) {
                qpos_snapshot_.assign(d->qpos, d->qpos + m->nq);
            }
        }
        const auto t_scene = clock::now();
        const mjrRect viewport{0, 0, cfg_.width, cfg_.height};
        mjr_render(viewport, &scn_, &con_);
        const auto t_render = clock::now();
        mjr_readPixels(nullptr, depth_.data(), viewport, &con_);
        stat_render_ += ms_since(t_scene, t_render);
        stat_read_ += ms_since(t_render);
        if (++stat_n_ == 100) {  // one-shot diagnostic: a GPU context takes ~1 ms here, llvmpipe ~70 ms
            std::cout << "depth_camera: per frame avg over first 100: lock wait " << stat_lock_ / stat_n_
                      << " ms, render " << stat_render_ / stat_n_ << " ms, readback " << stat_read_ / stat_n_
                      << " ms" << std::endl;
        }

        // GL depth -> metric z along the optical axis; bottom-up -> top-down.
        const int W = cfg_.width, H = cfg_.height;
        const double k = 1.0 - znear_ / zfar_;
        for (int r = 0; r < H; r++) {
            const float* row = depth_.data() + static_cast<size_t>(H - 1 - r) * W;
            uint16_t* out = depth_mm_.data() + static_cast<size_t>(r) * W;
            for (int c = 0; c < W; c++) {
                const double dn = row[c];
                double z = 0.0;  // no return
                if (dn < 1.0 - 1e-6) {
                    z = znear_ / (1.0 - dn * k);
                    if (z >= 0.99 * zfar_) z = 0.0;
                }
                const double mm = std::round(z * 1000.0);
                out[c] = static_cast<uint16_t>(std::min(mm, 65535.0));
            }
        }

        ++seq_;
        if (cfg_.dump_every > 0 && seq_ % cfg_.dump_every == 0) dump(t);

        Pending p;
        p.release = clock::now() + std::chrono::duration_cast<clock::duration>(
                                       std::chrono::duration<double>(cfg_.delay_ms * 1e-3));
        p.t = t;
        p.mm = depth_mm_;
        pending_.push_back(std::move(p));
    }

    void flush_pending()
    {
        const auto now = clock::now();
        while (!pending_.empty() && pending_.front().release <= now) {
            if (pub_->trylock()) {
                unitree_rl::msgs::write_depth_image(pub_->msg_, pending_.front().t, pending_.front().mm.data());
                pub_->unlockAndPublish();
            }
            pending_.pop_front();
        }
    }

    // 16-bit grey PNG (big-endian samples, as PNG requires) + a sidecar with
    // the sim time and qpos, so scripts/check_depth_stream.py can re-render
    // the identical state with MuJoCo's Python renderer.
    void dump(double t)
    {
        namespace fs = std::filesystem;
        fs::create_directories(cfg_.dump_dir);
        const std::string stem = (fs::path(cfg_.dump_dir) / ("depth_" + std::to_string(seq_))).string();
        std::vector<unsigned char> be(depth_mm_.size() * 2);
        for (size_t i = 0; i < depth_mm_.size(); i++) {
            be[2 * i] = static_cast<unsigned char>(depth_mm_[i] >> 8);
            be[2 * i + 1] = static_cast<unsigned char>(depth_mm_[i] & 0xff);
        }
        lodepng::encode(stem + ".png", be, cfg_.width, cfg_.height, LCT_GREY, 16);
        std::ofstream side(stem + ".txt");
        side.precision(12);
        side << "time " << t << "\nqpos";
        for (double q : qpos_snapshot_) side << " " << q;
        side << "\n";
    }

    GLFWwindow* window_;
    EglOffscreenContext egl_;
    mjModel** model_;
    mjData** data_;
    std::recursive_mutex* mtx_;
    Config cfg_;

    mjvOption opt_;
    mjvCamera cam_;
    mjvScene scn_;
    mjrContext con_;
    mjModel* ctx_model_ = nullptr;
    int cam_id_ = -1;
    double znear_ = 0.0, zfar_ = 0.0;

    std::vector<float> depth_;
    std::vector<uint16_t> depth_mm_;
    std::vector<double> qpos_snapshot_;
    std::deque<Pending> pending_;
    uint64_t seq_ = 0;

    static double ms_since(clock::time_point a, clock::time_point b = clock::now())
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    int stat_n_ = 0;
    double stat_lock_ = 0.0, stat_render_ = 0.0, stat_read_ = 0.0;

    std::unique_ptr<Pub_t> pub_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};
