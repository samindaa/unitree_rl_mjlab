# Plan: state-estimator and depth-image streaming for g1_dex3

Status: **implemented and validated in simulation (2026-09-14)**; robot-side
nodes (§4) are still to do. Deviations from the original proposal are marked
*Implemented:* below. Validation results are in §6.

Goal: the `g1_dex3` deploy controller must be able to consume two inputs it
does not have today, from the simulator now and from the real robot later,
through the same code path:

1. a **state estimate**: pelvis IMU pose in the world frame, plus linear and
   angular velocity expressed in the pelvis (IMU) frame;
2. a **depth image** matching the training camera of
   `smp_v2/src/tasks/hiphi_tracking_multi_distill/env_cfg.py`
   (64×36, fovy 58°, head-mounted on `torso_link`, cutoff 3 m).

Consumers: the future depth-distilled student (`student_state` +
`student_camera` ONNX inputs) and any estimator-aware UMT variant
(`motion_anchor_pos_b`, `base_lin_vel`). Today's `State_UmtMimic` needs
neither; this plan adds the plumbing and two observation terms, not a new
FSM state (that follows once a student ONNX exists).

---

## 1. Transport decision: DDS (unitree_sdk2), behind a small interface

**Recommendation: DDS now; no zenoh.**

| | DDS via unitree_sdk2 | zenoh |
| --- | --- | --- |
| Deps on this machine / Orin | already linked by sim and controller (cyclonedds 0.10.2, `idlc` + C++ codegen installed) | nothing installed anywhere; would need zenoh-c vendored for x86_64 **and** aarch64, Bazel + CMake rules |
| Message types | `nav_msgs::msg::dds_::Odometry_` ships in `/usr/local/include/unitree/idl/ros2/`; custom types via `idlc -l cxx` | define our own |
| Consumer code | `unitree::robot::SubscriptionBase<T>` (mutex + `isTimeout()`) already used for lowstate/hand state | new client code |
| Real robot | D435i is on the same Jetson as the controller; a local camera node publishes on the same DDS domain the SDK already uses | only pays off across hosts / NAT / WiFi |

Zenoh would only earn its cost if the camera or estimator lived on another
host. Keep that door open by hiding the transport behind two tiny interfaces
(`IOdomSource`, `IDepthSource`) so a zenoh backend is a drop-in later.

Known constraint carried over: Python tooling **cannot** join this DDS domain
(cyclonedds 11 wheels crash the 0.10.2 participants, see memory
`cyclonedds-version-crash`). Validation scripts therefore consume either the
UDP state tap or npz dumps written by the C++ probe tool (§6).

---

## 2. Message contracts (shared by sim, robot nodes, controller)

New directory `msgs/` at the repo root (referenced by both
`simulate/CMakeLists.txt` and `deploy/BUILD.bazel` / `deploy/robots/*/CMakeLists.txt`).

### 2.1 `rt/odom_pelvis` — `nav_msgs::msg::dds_::Odometry_` (existing IDL)

Deliberately **not** `rt/odommodestate` (Unitree's own topic) so the two never
collide and we own the convention.

| field | content |
| --- | --- |
| `header.stamp` | source time (sim: `mjData.time`; robot: estimator time) |
| `header.frame_id` | `"odom"` |
| `child_frame_id` | `"pelvis_imu"` |
| `pose.pose.position` | world position of the **pelvis IMU site** (m) |
| `pose.pose.orientation` | world orientation of that site (ROS field order x,y,z,w — the sim writes explicitly, the controller reads explicitly) |
| `twist.twist.linear` | linear velocity of the site **in the site frame** (velocimeter semantics = mjlab `robot/imu_lin_vel`) |
| `twist.twist.angular` | angular velocity in the site frame (= gyro) |

Why the IMU site and not the pelvis body origin: mjlab's `base_lin_vel`
term is the velocimeter at `imu_in_pelvis` (offset (0.04525, 0, −0.08339)
from the pelvis origin). Site velocity ≠ body-origin velocity by ω×r
(~0.1 m/s at 1 rad/s), so the topic carries what the policy was trained
on; the controller applies the fixed site→body offset when it needs body
FK (§5.2).

### 2.2 `rt/depth_camera` — `sensor_msgs::msg::dds_::PointCloud2_` as an organized depth image

*Implemented:* the installed `idlc` has no C++ generator plugin
(`libcycloneddsidlcxx.so` missing) and hand-writing the cyclonedds-cxx
serializer boilerplate is version-fragile, so instead of a custom IDL the
stream uses `PointCloud2_` — precompiled in `libunitree_sdk2.a`, readable by
ROS tooling — as an organized `height × width` image with one `UINT16` field
`depth_mm` (`point_step` 2, `row_step` 2·width, `is_dense=false`, row 0 =
top). fovy / cutoff are not transmitted: the consumer's `deploy.yaml` carries
`cutoff_distance` exactly as the training term does. Helpers
(`init/write/read_depth_image`) live in `deploy/include/msgs/stream_msgs.h`
(the contract header moved under `deploy/include/` so the Bazel module, whose
root is `deploy/`, can see it; the simulator's CMake adds that include dir).

**Depth semantics are fixed by training** (`mdp.camera_depth` +
mujoco_warp `render.py`):

- planar z-depth along the optical axis (warp line 743), the same thing a
  D435 reports;
- **no-hit pixel = 0** (warp line 724), and `camera_depth` then does
  `clamp(d, 0.01, 3.0) / 3.0`, so background lands at **0.0033, not 1.0**.
  The MuJoCo OpenGL renderer returns the far plane for background — the sim
  publisher must map `z ≥ zfar − ε` to 0. A D435 reports 0 for invalid
  pixels, so the real camera is consistent for free.
- image layout (B,1,H,W) row-major = the flat `H*W` vector the controller
  feeds ORT; top row first, pixel x increasing to the image right, where
  image-right is torso **−y** (see `_depth_camera_quat`).

---

## 3. Simulator side (`simulate/`)

### 3.1 Odometry publisher (small)

In `G1Bridge::run()` (`src/unitree_sdk2_bridge.h`), alongside the secondary
IMU: a `RealTimePublisher<Odometry_>` on `rt/odom_pelvis`, filled every
bridge tick (1 kHz; make the divider configurable, default 2 → 500 Hz) from
`mjData` directly — no new sensors needed:

- site = `imu_in_pelvis` (present in `scene_g1_dex3.xml:98`; **not** the
  `imu` site at the pelvis origin that the existing `frame_pos/frame_vel`
  sensors use);
- pose: `d->site_xpos`, `d->site_xmat` → quaternion;
- twist: `mj_objectVelocity(m, d, mjOBJ_SITE, id, vel, /*flg_local=*/1)`
  (angular first, then linear) — exactly velocimeter + gyro.

Config: `simulate/config.yaml` → `odom_topic`, `odom_divider`,
`odom_site: imu_in_pelvis`.

### 3.2 Depth camera streamer (new `src/depth_camera.h`)

Camera in the scene: extend `scripts/make_g1_dex3_scene.py` to add
`<camera name="depth_camera">` on `torso_link` with the training constants
copied verbatim from `hiphi_tracking_multi_distill/env_cfg.py`
(`HEAD_CAMERA_POS`, `HEAD_CAMERA_PITCH_RAD` → quat via the same 3×3 as
`_depth_camera_quat()`, `fovy=58`). Regenerate `scene_g1_dex3.xml`. Camera
identity with training is then by construction, not by transcription.

Rendering thread (does not touch the vendored `mujoco/simulate/simulate.cc`):

*Implemented:* the default GL backend is an **EGL context on the first GPU
device** (`EGL_EXT_platform_device`, what MuJoCo's Python renderer does),
`depth_camera.gl: egl`. On this workstation GLFW 3.4 selects its Wayland
backend and lands on Mesa/llvmpipe, where the 64×36 render took ~70 ms
(15 Hz); the EGL device context renders it in ~0.3 ms + ~0.7 ms readback at
a steady 30 Hz, and works headless. The hidden-GLFW-window variant below is
kept as `gl: glfw`. Multisampling is disabled for this context: MSAA-resolved
depth is biased +2–6 cm at grazing angles (measured), and the training ray
tracer samples pixel centres.

1. (`gl: glfw` only) `main()` creates a **hidden** GLFW window
   (`GLFW_VISIBLE=false`) before `RenderLoop()` — window creation must be
   on the main thread; the streamer thread then owns that context.
2. Streamer thread, lazily after the model loads: `mjr_makeContext`,
   `mjr_setBuffer(mjFB_OFFSCREEN)` (offscreen buffer default 640×480 ≥
   64×36), own `mjvScene`/`mjvOption` with `geomgroup = {0,1,2}` only,
   shadows/reflections/skybox off, `mjvCamera{mjCAMERA_FIXED, id}`.
3. Loop at `depth_hz` (default 30 — D435 depth stream; policy consumes the
   latest at 50 Hz, same as on the robot): lock `sim.mtx` →
   `mjv_updateScene` → unlock; `mjr_render` into a 64×36 viewport;
   `mjr_readPixels(nullptr, depth, …)`; convert normalized GL depth to
   metric `z = znear / (1 − d·(1 − znear/zfar))` with
   `znear = vis.map.znear·stat.extent`, `zfar = vis.map.zfar·stat.extent`;
   flip rows (GL is bottom-up); background → 0; pack `uint16 mm`; publish.
4. Optional `depth_delay_ms` (ring buffer before publish) — the same
   sim2sim latency knob as `action_delay_ms`, so the student's tolerance to
   camera latency can be swept in sim.
5. Debug: `depth_dump_png: path` writes frames via the already-vendored
   lodepng (§6 validation).

Config: `simulate/config.yaml` → `depth_camera: {enable, camera_name,
width, height, hz, cutoff_m, delay_ms, topic}`.

Headless note: the sim already requires a display (GLFW); the hidden window
adds no new requirement. EGL would be the path to a truly headless sim —
out of scope.

---

## 4. Robot side (later, hardware-dependent; separate binaries, controller unchanged)

- `deploy/tools/odom_adapter`: subscribes to whatever estimate the robot
  provides and republishes `rt/odom_pelvis` in §2.1 convention. **Open
  question to verify on the robot:** does `rt/odommodestate`
  (`unitree_go::msg::dds_::SportModeState_`) publish while we run low-level
  control, and in which frames are `position`/`velocity`? If it does not,
  the adapter becomes our own estimator (out of scope here) and the
  controller still does not change.
- `deploy/tools/depth_camera_node` (librealsense2 on the Jetson): D435i
  depth 848×480@30 → z-depth mm → FOV-match + area-downsample to 64×36,
  invalid stays 0 → `rt/depth_camera`. Intrinsics/extrinsics calibration
  vs the nominal URDF mount is a known-unknown flagged in the env cfg; plan
  a one-off check (render sim depth at the robot's measured pose vs a real
  frame of a known object).

---

## 5. Controller side (`deploy/`)

### 5.1 Sources (header-only, in `deploy/include/`)

- `sources/odom_source.h`: `IOdomSource { pose_w, lin_vel_b, ang_vel_b,
  stamp, fresh(max_age) }` + `DdsOdomSource : SubscriptionBase<Odometry_>`
  on `rt/odom_pelvis`.
- `sources/depth_source.h`: `IDepthSource { width, height, latest(frame&),
  stamp, fresh(max_age) }` + `DdsDepthSource : SubscriptionBase<DepthImage_>`.
  Double-buffered so the 50 Hz policy thread never blocks on a DDS callback.

### 5.2 Articulation data

`isaaclab::ArticulationData` gains `root_pos_w`, `root_lin_vel_b`,
`has_odom`, `odom_age`. `unitree::BaseArticulation::update()` copies from
the optional odom source under its lock. Torso (anchor) world position for
`motion_anchor_pos_b` = IMU-site pose → pelvis origin (fixed offset) →
waist FK (`waist_roll_link` at (−0.0039635, 0, 0.044) after `waist_yaw`;
`torso_link` at the origin of `waist_roll_link`), using the same waist-joint
composition `robot_anchor_quat_w()` already does for orientation.

### 5.3 Observation terms (registered next to the UMT ones)

| term | source | notes |
| --- | --- | --- |
| `base_lin_vel` (3) | `root_lin_vel_b` | mjlab `builtin_sensor robot/imu_lin_vel` |
| `motion_anchor_pos_b` (3) | odom + clip | needs an `init_pos` captured at `enter()` alongside `init_quat`, so the clip's world frame is translated onto the robot's start as well as yaw-aligned |
| `camera_depth` (H·W) | depth source | `clamp(d, min_depth, cutoff)/cutoff`, params `{sensor_name, cutoff_distance, min_depth}` mirroring `mdp.camera_depth`; stale frame → hold last, log once |

`deploy.yaml` for a student uses two groups, `student_state` and
`student_camera`, which the existing multi-input path of
`ObservationManager` already maps to ONNX input names; ORT reshapes the
flat 2304 floats to the model's `[1,1,36,64]`.

### 5.4 Safety

Registered checks in any state that consumes these: odom stale > 100 ms or
depth stale > 200 ms → transition to `Velocity` (or `Passive`), same
pattern as the bad-orientation check. Latency logged (source stamp vs
policy-step time) so the sim2sim delay sweep has a measured number to
match.

### 5.5 Build

- CMake (`deploy/robots/g1_dex3/CMakeLists.txt`) and Bazel
  (`deploy/BUILD.bazel: isaaclab_core`) both pick up `msgs/`.
- Only the ROS2 IDL header and the generated `DepthImage_` header are new
  compile inputs; no new libraries.

---

## 6. Validation

Results (2026-09-14, this workstation, `unitree_mujoco` + `g1_dex3_ctrl` in sim):

- `rt/odom_pelvis`: ~478 Hz received (500 Hz target; bridge-thread jitter),
  age ≤ 2 ms, unit quaternions, twist consistent with the robot swinging in
  the elastic band.
- `rt/depth_camera`: 30.0 Hz, age 6–7 ms, 100 % valid pixels looking at the
  floor; `scripts/check_depth_stream.py --dump-dir` vs MuJoCo's Python
  renderer (no MSAA): |dz| median 0.3–0.7 mm, max ≤ 4 mm, 0 % background
  mismatch, row order confirmed (row-flipped error ~1.1 m).
- Controller: sources created from `config.yaml`, `Umt` entered with a live
  estimate; the anchor FK (IMU site → pelvis origin → waist-yaw → torso_link)
  matches MuJoCo's `torso_link` to 0.000 mm over 200 random poses.
- Bazel (`//robots/g1_dex3:g1_dex3_ctrl`, `:topic_probe`) and CMake both build.

Procedure:

1. **Rates & conventions (sim):** new `deploy/robots/g1_dex3/tools/topic_probe`
   subscribes to both topics, prints Hz / age, and dumps N seconds to an
   uncompressed npz. Hold the robot in the elastic band: odom pose vs UDP
   state-tap qpos; rotate the pelvis → gyro sign; translate → lin-vel sign
   and site-vs-origin offset.
2. **Depth vs training renderer:** script that loads `scene_g1_dex3.xml` in
   mujoco_warp with the same camera, sets the qpos taken from the sim's npz
   dump, renders, and diffs per pixel against the streamed frame. Expect
   sub-cm agreement away from silhouettes; assert background == 0, top row
   is the far/upper part of the view, image-right is torso −y.
3. **`base_lin_vel` sanity:** run the UMT clip in unitree_mujoco with the
   probe on, compare the streamed site velocity statistics with mjlab play
   of the same clip.
4. **Robot:** confirm which estimator topic exists under low-level control
   (§4); measure camera-node end-to-end latency to size `depth_delay_ms`.

---

## 7. Work breakdown (order = dependency order)

| # | item | touches | size |
| --- | --- | --- | --- |
| 0 | ✅ contract header (`deploy/include/msgs/stream_msgs.h`) | `deploy/include/msgs/` | S |
| 1 | ✅ odom publisher | `simulate/src/unitree_sdk2_bridge.h`, `config.yaml`, `param.h` | S |
| 2 | ✅ camera in scene generator; depth streamer thread (EGL / hidden GLFW) | `scripts/make_g1_dex3_scene.py`, `simulate/src/depth_camera.h`, `main.cc`, `CMakeLists.txt` | M |
| 3 | ✅ sources, articulation fields, obs terms (`base_lin_vel`, `camera_depth`, `motion_anchor_pos_b`), probe tool, builds | `deploy/include/sources/`, `deploy/robots/g1_dex3/...`, `BUILD.bazel` | M |
| 4 | ✅ validation script (`scripts/check_depth_stream.py`, §6.1–6.2); §6.3 covered by the first consumer, `State_HiphiStudent` (depth student on the 160-obs UMT base, `scripts/sim_e2e_hiphi_student.py`) | `scripts/` | S–M |
| 5 | ⬜ robot nodes (odom adapter, realsense node) | `deploy/tools/` | M, blocked on §4 answers |

Items 1 and 2 are independent of 3; 4 needs 1–3.

## 8. Risks / open questions

- Real-robot estimator availability and frame convention (§4) — the single
  biggest unknown; everything else is under our control.
- OpenGL rasterization vs warp ray tracing differ at silhouettes and at the
  near plane; if the student turns out sensitive, add light depth noise in
  the sim publisher rather than chasing pixel parity.
- Sim IMU sensors (`imu_quat/gyro/acc`, `frame_*`) sit on the `imu` site at
  the pelvis origin, not on `imu_in_pelvis`; harmless for gyro/quat, but
  the odom twist must be taken from `imu_in_pelvis` (done in §3.1).
- GLFW hidden window under the headless `script` recipe: uses the same X
  display the visible window already needs — expected to work, to be
  confirmed in item 2.
