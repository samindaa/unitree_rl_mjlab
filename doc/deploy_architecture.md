# Deployment Architecture

An architectural walkthrough of [`deploy/`](../deploy), the C++ controller that runs
trained policies on physical Unitree robots. **Unitree G1 (29-dof)** is used as the
running example throughout; every other robot follows the same structure.

- [What it is](#what-it-is)
- [Layers](#layers)
- [Repository layout](#repository-layout)
- [Boot sequence](#boot-sequence)
- [The FSM graph](#the-fsm-graph)
- [The two-thread control loop](#the-two-thread-control-loop)
- [One policy step, concretely](#one-policy-step-concretely)
- [The sim to real contract](#the-sim-to-real-contract)
- [Mimic: motion tracking](#mimic-motion-tracking)
- [Build and run](#build-and-run)
- [Driving the FSM without a gamepad](#driving-the-fsm-without-a-gamepad)
- [Things to know before you trust it](#things-to-know-before-you-trust-it)

---

## What it is

A **standalone C++ real-robot controller**. No Python, no mjlab, no MuJoCo at runtime.
It re-implements Isaac Lab's manager-based environment API in C++ so that the exported
`policy.onnx` sees *byte-identical observation vectors* on hardware to what it saw in
simulation.

The only artifacts crossing from training to the robot are:

1. `policy.onnx` (+ `policy.onnx.data` for large models), exported during training
2. `deploy.yaml`, a hand-maintained mirror of the training environment configuration

---

## Layers

```
┌──────────────────────────────────────────────────────────────────┐
│  FSM layer — safety & mode switching        include/FSM/         │
│  CtrlFSM (1 kHz thread) → State_Passive / FixStand / RLBase /    │
│  Mimic.  Joystick DSL strings in config.yaml define transitions. │
└───────────────────────────┬──────────────────────────────────────┘
                            │  owns
┌───────────────────────────▼──────────────────────────────────────┐
│  isaaclab mirror — the C++ twin of the training env              │
│                                             include/isaaclab/    │
│  ManagerBasedRLEnv (50 Hz thread)                                │
│    ├── ObservationManager  registry of REGISTER_OBSERVATION fns  │
│    ├── ActionManager       JointPositionAction: scale + offset   │
│    ├── Articulation        joint_pos/vel, ang_vel, proj_gravity  │
│    └── Algorithms          OrtRunner = onnxruntime session       │
└───────────────────────────┬──────────────────────────────────────┘
                            │  reads/writes
┌───────────────────────────▼──────────────────────────────────────┐
│  Unitree SDK2 / DDS — hardware I/O          include/*.h, Types.h │
│  LowState_t (subscribe: IMU, motor q/dq, joystick)               │
│  LowCmd_t   (publish: per-motor q, dq, kp, kd, tau)              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

Everything under [`deploy/include/`](../deploy/include) is **robot-agnostic** framework
code. Everything under `deploy/robots/<name>/` is the **robot-specific** binding — only
five small pieces per robot:

| Path | Role |
| --- | --- |
| [`include/Types.h`](../deploy/robots/g1/include/Types.h) | Binds `LowCmd_t` / `LowState_t` to this robot's DDS topic types |
| [`main.cpp`](../deploy/robots/g1/main.cpp) | DDS init, robot handshake, build the FSM, idle forever |
| [`config/config.yaml`](../deploy/robots/g1/config/config.yaml) | The FSM graph: enabled states, gains, joystick transitions |
| `src/State_RLBase.cpp`, `src/State_Mimic.cpp` | Robot-specific observation terms and the action → motor write |
| `config/policy/<task>/<version>/` | `exported/policy.onnx` and `params/deploy.yaml` |

Shared framework files worth knowing:

| Path | Role |
| --- | --- |
| [`include/FSM/CtrlFSM.h`](../deploy/include/FSM/CtrlFSM.h) | The 1 kHz state machine driver |
| [`include/FSM/BaseState.h`](../deploy/include/FSM/BaseState.h) | State interface + `REGISTER_FSM` factory registry |
| [`include/FSM/FSMState.h`](../deploy/include/FSM/FSMState.h) | Adds shared `lowcmd`/`lowstate`/`keyboard` and parses transitions |
| [`include/FSM/State_Passive.h`](../deploy/include/FSM/State_Passive.h) | Damping-only safe state |
| [`include/FSM/State_FixStand.h`](../deploy/include/FSM/State_FixStand.h) | Interpolate to a fixed stand pose |
| [`include/FSM/State_RLBase.h`](../deploy/include/FSM/State_RLBase.h) | Generic "run an ONNX policy" state |
| [`include/isaaclab/`](../deploy/include/isaaclab) | The env / manager / articulation / onnxruntime mirror |
| [`include/unitree_articulation.h`](../deploy/include/unitree_articulation.h) | Fills `ArticulationData` from a DDS `LowState` |
| [`include/param.h`](../deploy/include/param.h) | CLI parsing, `config.yaml` loading, policy-directory resolution |
| [`include/unitree_joystick_dsl.hpp`](../deploy/include/unitree_joystick_dsl.hpp) | The joystick expression language used by transitions |
| [`thirdparty/`](../deploy/thirdparty) | Vendored onnxruntime (x64 + aarch64) and cnpy |

---

## Boot sequence

From [`robots/g1/main.cpp`](../deploy/robots/g1/main.cpp):

1. `param::helper()` locates the binary via `/proc/self/exe`, walks up to the project
   directory, and loads `config/config.yaml`.
2. `ChannelFactory::Init(0, --network)` starts DDS on the requested interface —
   `lo` for simulation, e.g. `enp5s0` for the real robot.
3. `init_fsm_state()` **refuses to start if another process already owns the lowcmd
   channel**, then blocks on `wait_for_connection()` until the robot is publishing.
4. `mode_machine = 5` (29-dof) is set and `check_mode_machine()` hard-exits on a
   robot-type mismatch.
5. `CtrlFSM(config["FSM"])` instantiates every state listed under `FSM._` by looking up
   `"State_" + type` in the factory map populated by the `REGISTER_FSM` macro
   ([`BaseState.h:45`](../deploy/include/FSM/BaseState.h#L45)).
6. `fsm->start()` enters `states[0]` (Passive) and spawns the 1 kHz FSM thread. `main`
   then sleeps forever; all work happens on the FSM and policy threads.

---

## The FSM graph

As configured in [`robots/g1/config/config.yaml`](../deploy/robots/g1/config/config.yaml):

```
             LT+up.on_pressed        RT+A.on_pressed
   Passive ──────────────────► FixStand ───────────────► Velocity
      ▲                            │                      │   ▲
      │        LT+B.on_pressed     │                      │   │ RT+A
      └────────────────────────────┴──────────────────────┘   │
      ▲                                              RB+A     │
      │                                                 ▼     │
      └───────────────── LT+B ───────────────── Mimic_Dance1_subject2
                                                        │
                                          time_end exceeded (→ Velocity)

  Implicit edge from EVERY state: lowstate->isTimeout() ──► Passive
```

Transitions are **strings parsed at construction** into predicates over the joystick
([`FSMState.h:35-43`](../deploy/include/FSM/FSMState.h#L35-L43)), using the DSL documented
in [`unitree_joystick_dsl.hpp`](../deploy/include/unitree_joystick_dsl.hpp). It supports
`+` (and), `|` (or), `!` (not), `.on_pressed` / `.on_released` edge triggers, and
long-press forms such as `LT(2s)`. Adding a new mode is a YAML edit plus a state class —
no change to the FSM driver.

For local `lo` runs these chords can be triggered from the keyboard instead of a physical
gamepad — see [Driving the FSM without a gamepad](#driving-the-fsm-without-a-gamepad).

Each state applies its own gains:

| State | Gains | Behaviour |
| --- | --- | --- |
| `Passive` | `kp = 0`, `kd = 3` | Damping only; commands current measured `q`. The safe state. |
| `FixStand` | Stiff gains from `config.yaml` | Linearly interpolates from the current pose to the stand pose over 2 s ([`LinearInterpolator.h`](../deploy/include/LinearInterpolator.h)) |
| `Velocity` / `Mimic_*` | `stiffness` / `damping` from `deploy.yaml` | Policy-authored joint targets |

Transition checks are evaluated in registration order and the first true one wins. The
DDS-timeout → `Passive` check is registered by the `FSMState` constructor, so it is
present in every state and is checked before any state-specific transition.

---

## The two-thread control loop

```
 FSM thread                              Policy thread
 (1 kHz, RecurrentThread)                (50 Hz, spawned in State_RLBase::enter)
 ─────────────────────────               ────────────────────────────────────────
 pre_run()   lowstate->update()   ┐
 run()       write action → lowcmd│      env->step():
 post_run()  lowcmd->publish()    │        robot->update()      ← IMU + motor state
 check transitions                │        obs = obs_mgr->compute()
 sleep 1 ms                       │        act = alg->act(obs)  ← onnxruntime
 ↺                                └──────  act_mgr->process_action(act)
                                          sleep_until(+20 ms)
                                          ↺
```

- The FSM thread period is fixed at 1 ms (`CtrlFSM::dt = 0.001`).
- The policy thread period is `step_dt` from `deploy.yaml` — 0.02 s (50 Hz) for G1.
- The FSM thread re-publishes the *latest* joint target every millisecond; actual 1 kHz
  joint tracking happens on the robot's motor controllers using the `kp`/`kd` shipped
  inside `lowcmd`.
- `State_RLBase::exit()` sets the run flag false and **joins** the policy thread, so
  leaving an RL state always stops inference before the next state touches `lowcmd`.

---

## One policy step, concretely

G1 velocity policy:

```
LowState (DDS)
  │ imu.gyroscope[3]           ──► data.root_ang_vel_b
  │ imu.quaternion[4]          ──► data.root_quat_w ──conj*(0,0,-1)──► projected_gravity_b
  │ motor_state[joint_ids_map] ──► data.joint_pos / joint_vel      unitree_articulation.h
  ▼
ObservationManager::compute_group("obs")     ← concatenated in deploy.yaml order
   base_ang_vel        3
   projected_gravity   3
   velocity_commands   3   ← joystick ly/-lx/-rx, clamped to commands.base_velocity.ranges
   gait_phase          2   ← sin/cos of a phase integrated at 1/0.6 s, zeroed when |cmd|<0.1
   joint_pos_rel      29   ← q - default_joint_pos
   joint_vel_rel      29
   last_action        29
   ─────────────────────
                      98  ══► OrtRunner  input "obs" [1,98] → output "actions" [1,29]
   ▼
ActionManager → JointPositionAction:  q_target[i] = a[i]*scale[i] + offset[i]
   ▼
State_RLBase::run():  lowcmd.motor_cmd[joint_ids_map[i]].q() = q_target[i]
```

The `98` and `29` above match the input/output shapes of the checked-in
`config/policy/velocity/v0/exported/policy.onnx`.

**Observation terms are a registry.** `REGISTER_OBSERVATION(name)` inserts a function into
a global map at static-init time
([`observation_manager.h:22-27`](../deploy/include/isaaclab/manager/observation_manager.h#L22-L27)).
Generic terms live in
[`isaaclab/envs/mdp/observations/observations.h`](../deploy/include/isaaclab/envs/mdp/observations/observations.h);
robot- or task-specific ones are defined in the robot's `.cpp` files. A term named in
`deploy.yaml` that was never registered throws at construction, not at runtime.

**Group names must match ONNX input names.** `OrtRunner::act` looks up every ONNX input
name in the computed observation map and throws if one is missing
([`algorithms.h:69-73`](../deploy/include/isaaclab/algorithms/algorithms.h#L69-L73)). When
the YAML lists observation terms directly (the common case), they are collected into a
single group literally named `"obs"`, selected by the `only_one_input` heuristic at
[`observation_manager.h:98`](../deploy/include/isaaclab/manager/observation_manager.h#L98).
A multi-input policy simply nests its terms under group names matching the ONNX inputs.

---

## The sim to real contract

[`deploy.yaml`](../deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml) *is* the
handoff, and it is **maintained by hand** — nothing in `scripts/` generates it. Training
only emits the ONNX file ([`src/tasks/velocity/rl/runner.py`](../src/tasks/velocity/rl/runner.py)).

| `deploy.yaml` field | Must match in training |
| --- | --- |
| `observations:` **order** | Field order of the `policy` observation group in [`velocity_env_cfg.py`](../src/tasks/velocity/velocity_env_cfg.py). Terms are matched by *function*, not key name: `command` → `velocity_commands`, `phase` → `gait_phase`, `actions` → `last_action` |
| `actions.*.scale` / `offset` | `JointPositionActionCfg` scale, and the robot's default joint positions |
| `default_joint_pos` | `HOME_KEYFRAME` in the robot constants; also the zero point for `joint_pos_rel` |
| `stiffness` / `damping` | Actuator gains from the robot constants — these become the motor `kp` / `kd` |
| `step_dt` | Environment `decimation × sim.dt` |
| `commands.base_velocity.ranges` | The velocity command sampling ranges |
| `joint_ids_map` | Permutation from **policy joint order → Unitree SDK motor index** |

`joint_ids_map` is the one that bites. It is the identity for G1 (29 joints in the same
order), but it is the only thing preventing a silently-scrambled policy on a robot whose
SDK ordering differs from its MJCF ordering.

**Policy version selection is a convention, not configuration.** `policy_dir:
config/policy/velocity` contains no `exported/` directory, so
[`param.h:95-113`](../deploy/include/param.h#L95-L113) sorts the subdirectories and picks
the **last one that contains `exported/`**. Dropping in a `v1/` alongside `v0/` switches
to it automatically.

---

## Mimic: motion tracking

[`State_Mimic.cpp`](../deploy/robots/g1/src/State_Mimic.cpp) is a separate state rather
than a subclass of `State_RLBase`, because it carries a reference trajectory:

- Loads the same `.npz` the training pipeline consumed (`body_pos_w`, `body_quat_w`,
  `joint_pos`, `joint_vel`) using vendored `cnpy`, at a fixed 50 fps, advancing `frame`
  from `episode_length * step_dt`.
- Adds two observation terms: `motion_command` (58 = reference joint positions +
  velocities) and `motion_anchor_ori_b` (6 = first two columns of a rotation matrix).
  Total observation width 154, actions still 29 — matching the checked-in mimic ONNX.
- On `enter()`, computes `init_quat` = robot yaw ⊗ reference yaw⁻¹, so the motion is
  **replayed in whatever direction the robot currently faces** rather than in the
  motion-capture world frame. The torso orientation used for that anchor is the IMU
  quaternion composed with the three waist joints
  ([`State_Mimic.cpp:18-24`](../deploy/robots/g1/src/State_Mimic.cpp#L18-L24)).
- Registers an extra transition: once `episode_length * step_dt` exceeds `time_end`, fall
  through to `end_state` (default `Velocity`).

Only `g1` and `g1_23dof` ship a Mimic state. `go2`, `h1_2`, `h2`, `a2` and `r1` are
velocity-only.

---

## Build and run

Prerequisites (system-wide):

- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)
- `libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev`
- `libglfw3-dev` — only for the simulator in [`simulate/`](../simulate). Missing it fails
  the build at `fatal error: GLFW/glfw3.h: No such file or directory`; the bundled
  `simulate/mujoco/` ships MuJoCo headers and `libmujoco.so` but no GLFW.

onnxruntime (x64 and aarch64) and cnpy are vendored under
[`deploy/thirdparty/`](../deploy/thirdparty), so the same source cross-compiles for the
robot's onboard aarch64 computer.

```bash
# 0. System dependencies
sudo apt-get install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                        libspdlog-dev libfmt-dev libglfw3-dev

# 1. Place the exported policy
cp policy.onnx policy.onnx.data \
   deploy/robots/g1/config/policy/velocity/v0/exported/

# 2. Build the controller  -> deploy/robots/g1/build/g1_ctrl
cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build -j$(nproc)

# 3. Build the simulator   -> simulate/build/unitree_mujoco (and jstest)
cmake -S simulate -B simulate/build
cmake --build simulate/build -j$(nproc)
```

Both commands are run from the repository root.

Then run the two processes in **separate terminals**, simulation first:

```bash
# Terminal 1 — simulator
./simulate/build/unitree_mujoco

# Terminal 2 — controller, over the loopback interface
cd deploy/robots/g1/build && ./g1_ctrl --network=lo

# On the real robot, swap the interface for the wired one
./g1_ctrl --network=enp5s0
```

Useful flags from [`param.h`](../deploy/include/param.h): `--network/-n`, `--log`
(rotating file log under `<proj>/log/`), `--version/-v`, `--help/-h`. The simulator takes
`--network/-n`, `--domain_id/-i`, `--robot/-r` and `--scene/-s`, which are applied after
the YAML and so override it.

The simulator locates [`simulate/config.yaml`](../simulate/config.yaml) relative to its own
executable ([`main.cc:674-675`](../simulate/src/main.cc#L674-L675)), so it can be launched
from any working directory; a relative `robot_scene` resolves against the repository root.
Which robot it loads is set by `robot` / `robot_scene` there — it must match the controller
binary you run, and `interface: "lo"` must match `--network=lo`.

On the physical robot: power on suspended, wait for `zero-torque`, press `L2 + R2` to
enter debug mode, then connect over Ethernet with the PC at `192.168.123.222/24`.
Operator sequence once `g1_ctrl` is running: `L2 + Up` → FixStand, `R2 + A` → Velocity,
`R1 + A` → Mimic, `L2 + B` → Passive at any time.

---

## Driving the FSM without a gamepad

The transitions above are gamepad chords, which makes a physical controller a hard
requirement for local `lo` runs. To remove it, the simulator can synthesize the gamepad
from terminal key presses: [`simulate/src/keyboard_joystick.h`](../simulate/src/keyboard_joystick.h)
implements `KeyboardJoystick : public unitree::common::UnitreeJoystick` and populates
`wireless_remote` exactly as the XBox/Switch drivers do.

The injection is entirely sim-side, so **nothing in `deploy/` changes** and every robot
works unchanged. From the controller's point of view a gamepad is plugged in.

Enable it in [`simulate/config.yaml`](../simulate/config.yaml):

```yaml
use_joystick: 1
joystick_type: "keyboard"     # instead of "xbox" / "switch"

keyboard_map:
  "1": "LT + up"  # Passive  -> FixStand
  "2": "RT + A"   # FixStand -> Velocity  (also Mimic -> Velocity)
  "3": "RB + A"   # Velocity -> Mimic
  "0": "LT + B"   # any      -> Passive (damping)
keyboard_press_duration: 0.3  # seconds a chord stays asserted per key press
keyboard_axis_step: 0.1       # velocity command increment per key press
keyboard_trigger_lead: 0.15   # seconds LT/RT lead the rest of the chord
```

Keys are typed in the **`unitree_mujoco` terminal**, which prints the active map at
startup. The four chords above cover every transition defined across all six robots.

| Key | Sends | Effect |
| --- | --- | --- |
| `1` | `LT + up` | Passive → FixStand |
| `2` | `RT + A` | FixStand → Velocity, Mimic → Velocity |
| `3` | `RB + A` | Velocity → Mimic |
| `0` | `LT + B` | → Passive (damping), from any state |
| `w` / `s` | `ly` ∓ | forward / backward (`lin_vel_x`) |
| `a` / `d` | `lx` ∓ | strafe left / right (`lin_vel_y`) |
| `q` / `e` | `rx` ∓ | turn left / right (`ang_vel_z`) |
| `space` | zeroes axes | stop |

Arrow keys alias `w`/`a`/`s`/`d`.

Four details make this behave like real hardware:

- **Chords, not keys.** A terminal reports one key at a time, so `LT + up` cannot be typed
  literally. Each chord is bound to a single key that asserts every button in it for
  `keyboard_press_duration`, which the controller's `extract()` sees as one clean rising
  edge — exactly what the `.on_pressed` DSL triggers need. Auto-repeat from holding the key
  pushes the release time out, so the chord stays held.
- **The triggers lead the rest of the chord.** `LT`/`RT` are `Axis`, not `Button`, so the
  controller low-passes them on `extract()` (`Axis::smooth = 0.03`) and they need ~23
  updates — `ln 0.5 / ln 0.97`, about 23 ms at the FSM's 1 kHz — to cross the 0.5 press
  threshold. A `.on_pressed` edge is true for exactly one update. Assert a chord all at
  once and the two never coincide: `up.on_pressed` fires on tick 0 and is gone long before
  `LT.pressed` becomes true, so `LT + up.on_pressed` is never satisfied and the transition
  silently never happens. `keyboard_trigger_lead` asserts the triggers first, the same way
  a human holds L2 before tapping Up. Chords of plain buttons (`RB + A`) are unaffected.
- **Sticky axes.** Velocity commands step and *stay* rather than being momentary, which is
  what a velocity policy expects. Values are clamped to ±1.
- **No filter lag on the sim side.** `smooth` is set to `1.0` on the synthesized axes; with
  the `Axis` default an asserted `LT` would take another ~23 updates just to appear in
  `combine()`, on top of the delay the controller already adds.

Button names accept both the DSL spelling used in the controller config (`LT`, `RB`) and
the labels the controller prints (`L2`, `R1`).

The simulator puts the terminal in raw mode (`ICANON`/`ECHO` off, `ISIG` kept so Ctrl-C
still works) and restores it on exit via `atexit` plus `SIGINT`/`SIGTERM` handlers, which
install themselves only if nothing else has claimed those signals.

Set `joystick_type` back to `"xbox"` to use a real gamepad. `jstest`, built alongside
`unitree_mujoco`, prints the packed Unitree button word for a pad on `joystick_device`,
which is a quick way to confirm a physical controller is being read at all.

---

## Browser-based visualization (viser mirror)

On machines without GPU-backed OpenGL (e.g. over Chrome Remote Desktop) the
simulator's GLFW window falls back to software rendering, which is slow and — via
the render/physics mutex — drags the whole simulation down. The fix is to not
render locally at all:

```bash
# Terminal 1 — simulator without a visible window (keyboard joystick still
# works: it reads the terminal, not the GLFW window). `env -u WAYLAND_DISPLAY`
# matters on Wayland sessions: GLFW 3.4 prefers WAYLAND_DISPLAY over the
# virtual X display, which would pop the window up on the real screen.
env -u WAYLAND_DISPLAY xvfb-run -a ./simulate/build/unitree_mujoco

# Terminal 2 — viser mirror; open http://localhost:8080 in your browser
uv run scripts/sim_viser_mirror.py

# Terminal 3 — controller, as usual
cd deploy/robots/g1/build && ./g1_ctrl --network=lo
```

The simulator streams `(time, qpos)` datagrams to `127.0.0.1:9870` at 100 Hz
(`state_tap_port` in [`simulate/config.yaml`](../simulate/config.yaml),
implemented in [`state_tap.h`](../simulate/src/state_tap.h)); the mirror loads
the same scene MJCF, runs forward kinematics, and serves the scene over
websocket — all rendering happens in the browser's WebGL. Killing the mirror
never affects the simulation or the controller.

Commands flow the other way on `state_tap_port + 1`: the mirror's sidebar has
buttons for every simulator command, each sent as a one-key datagram that the
`CommandTap` routes exactly like the native key paths —

| Sidebar control | Key sent | Native equivalent |
| --- | --- | --- |
| FSM chords (from `keyboard_map`) | `1`/`2`/`3`/`0`… | terminal keyboard joystick |
| Velocity move / turn / stop | `w a s d q e` + space | terminal keyboard joystick |
| Elastic band toggle / raise / lower | `9` / `7` / `8` | GLFW window keys |
| Reset simulation | backspace | GLFW window backspace |

so the GLFW window (invisible under `xvfb-run`) and the simulator terminal are
both fully optional. The FSM-chord and velocity buttons appear when
`joystick_type` is `"keyboard"`; band buttons when `enable_elastic_band: 1`.

The mirror deliberately does **not** subscribe to the DDS topics: the simulator
and controller link cyclonedds 0.10.2, which segfaults on the XTypes discovery
data emitted by the newer cyclonedds releases that Python 3.12 wheels exist
for. Keep Python processes off the DDS domain.

---

## Things to know before you trust it

1. **`bad_orientation` is disabled.**
   [`terminations.h:15`](../deploy/include/isaaclab/envs/mdp/terminations.h#L15)
   unconditionally returns `false`, with the real check commented out. Both
   `State_RLBase` and `State_Mimic` register it as a fall-to-`Passive` guard, so that
   particular safety net is currently inert. The DDS-timeout → `Passive` edge is live and
   is what actually protects you.

2. **The action handoff is unsynchronized.** The FSM thread reads
   `action_manager->processed_actions()` at 1 kHz while the policy thread writes it at
   50 Hz, with no lock. `Algorithms` has an `act_mtx_`, but
   `ActionManager::_processed_actions` is unprotected. In practice this is a torn read at
   worst, at most once per 20 ms window, on a vector of floats.

3. **`deploy.yaml` drift is silent.** Reordering or adding an observation term in the
   training config without updating the YAML can still produce a correctly-sized
   observation vector, in which case the policy simply behaves badly. The only automatic
   check is the ONNX input-shape mismatch, which fires just when the *totals* disagree.
