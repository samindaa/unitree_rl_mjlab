# g1_dex3 — G1 (29dof) + Dex3 hands, UMT tracking deploy

Deploys smp_v2 task **`Umt-Tracking-G1-No-State-Estimation`**
(`smp_v2/src/tasks/umt`). Same FSM/isaaclab-mirror stack as `../g1`; the
differences are the command layout and the action space:

| | `g1` Mimic | `g1_dex3` Umt |
| --- | --- | --- |
| Reference obs | `motion_command` (q_ref + qd_ref, 58) | `zest_ref` (55: z, roll, pitch, v_b, ω_b, g_b, **43** q_ref) |
| Anchor obs | `motion_anchor_ori_b` from waist joints | `motion_anchor_ori_b` from clip `body_quat_w[15]` (torso_link) |
| Action | `JointPositionAction` (default-pose offset) | `ReferenceJointPositionAction` (`q_cmd = q_ref + Σa`) |
| Clip npz | 29 joints / 1 body | 43 joints / 46 bodies (fingers follow the clip, see below) |
| `default_joint_pos` | HOME pose | mjlab `KNEES_BENT_KEYFRAME` |

Obs width 154, action width 29 — see the header of
[`include/State_UmtMimic.h`](include/State_UmtMimic.h) and
[`config/policy/umt/v0/params/deploy.yaml`](config/policy/umt/v0/params/deploy.yaml).

## Artifacts

```
config/policy/umt/v0/exported/policy.onnx   # actor export, static [1,154] -> [1,29]
config/policy/umt/v0/params/deploy.yaml     # checked in
config/policy/umt/v0/params/motion.npz      # one clip, from scripts/umt_bundle_to_deploy_npz.py
```

```
uv run python scripts/umt_bundle_to_deploy_npz.py umt_v1.zip --list
uv run python scripts/umt_bundle_to_deploy_npz.py umt_v1.zip <clip> deploy/robots/g1_dex3/config/policy/umt/v0/params/motion.npz
uv run python scripts/umt_bundle_to_deploy_npz.py --onnx deploy/robots/g1_dex3/config/policy/umt/v0/exported/policy.onnx
```

## Build / run

```
cd deploy/robots/g1_dex3 && mkdir -p build && cd build && cmake .. && make -j
./g1_dex3_ctrl --network <iface>
```

Joystick: `L2+Up` FixStand → `R2+A` Velocity → `R1+A` Umt → `R2+A` back to Velocity, `L2+B` Passive.

## Command runbook

All paths relative to the repo root (`unitree_rl_mjlab`) unless noted.

**Build** (once, or after code changes; the controller must be built in this
in-tree `build/` — `param.h` finds `config/config.yaml` from the binary's
parent directory name, and `cmake ..` must be re-run after adding a `.cpp`):

```bash
cd simulate/build && cmake .. && make -j8 unitree_mujoco && cd -
cd deploy/robots/g1_dex3/build && cmake .. && make -j8 && cd -
```

**Interactive run** (two terminals):

```bash
# terminal 1: simulator (publishes rt/odom_pelvis + rt/depth_camera, EGL depth render)
cd simulate/build && ./unitree_mujoco
# terminal 2: controller
cd deploy/robots/g1_dex3/build && ./g1_dex3_ctrl --network lo
```

Keys go to the **simulator** — either the terminal `unitree_mujoco` was
started from or the MuJoCo window (both feed the keyboard joystick,
`simulate/config.yaml keyboard_map`); keys typed in the controller's terminal
do nothing. The simulator terminal echoes each decoded chord (`[keyboard]
RT + A`) and the controller logs `FSM: Change state from ... to ...` — if
neither appears, the key did not reach the simulator. Sequence:
`1` FixStand -> `2` Velocity -> `8` x5 (lower the elastic band ~0.5 m) ->
`9` (release it) -> `4` HiphiStudent or `3` Umt -> `0` Passive. Release the
band only once `Velocity` is up: released in FixStand the robot drops 0.5 m
and falls. Gamepad: `L2+Up`, `R2+A`, `R1+B` (student) / `R1+A` (UMT), `L2+B`.

**Automated evaluation** (the baseline numbers below):

```bash
# deploy side: sim + controller + FSM sequence + probe summary (9 s in the student state)
uv run python scripts/sim_e2e_hiphi_student.py 9
# mjlab reference for the same clip (from smp_v2)
cd ~/third_party/smp_v2 && uv run scripts/rollout_hiphi_student.py \
  --checkpoint logs/rsl_rl/g1_hiphi_umt_multi_v2_distill/2026-09-13_01-24-51_pnp64_res_v0/model_9999.pt --steps 450
```

**Stream diagnostics**:

```bash
uv run python scripts/sim_viser_mirror.py     # browser viewer: robot + a live panel of the depth frames the controller receives
deploy/robots/g1_dex3/build/topic_probe -n lo -s 5 -d /tmp/probe.npz      # rates / ages while the sim runs
uv run python scripts/check_depth_stream.py --probe-npz /tmp/probe.npz
# pixel-level check: set depth_camera.dump_dir/dump_every in simulate/config.yaml, run the sim, then
MUJOCO_GL=egl uv run python scripts/check_depth_stream.py --dump-dir /tmp/depth_dump
```

**New checkpoints / clips**:

```bash
# student (from smp_v2); a --nose student needs the 154-obs base in hiphi_umt_base/ instead
cd ~/third_party/smp_v2 && uv run python scripts/export_student_onnx.py logs/rsl_rl/<run>/model_XXXX.pt --out ckpts/<name>.onnx
cp ckpts/<name>.onnx ~/third_party/unitree_rl_mjlab/deploy/robots/g1_dex3/config/policy/hiphi_student/v0/exported/policy.onnx
# UMT base / plain UMT policy (Mish!)
uv run python scripts/export_policy_onnx.py ckpts/<base>.pt --activation mish
# clip: umt bundle zip or a single pnp64 clip npz -> cnpy-safe deploy npz; then set motion_file in config/config.yaml
uv run python scripts/umt_bundle_to_deploy_npz.py /tmp/smp_motion_bundles/pnp64_dex3_v2-*/<clip>.npz \
  deploy/robots/g1_dex3/config/policy/hiphi_student/v0/params/<clip>.npz
```

## Dex3 hands

The hands are not part of `rt/lowcmd`. [`include/Dex3Hands.h`](include/Dex3Hands.h)
publishes a `HandCmd_` on `rt/dex3/left/cmd` and `rt/dex3/right/cmd` from a
background thread (100 Hz, kp 1.5 / kd 0.2 — the gains Unitree's own Dex3
teleop controller uses) in **every** FSM state, so the hands always hold a
pose:

| FSM state | Finger targets |
| --- | --- |
| Passive, FixStand, Velocity | rest pose `dex3.open_pose` — thumbs folded across the palm (the all-zero qpos0 puts the straight thumbs 2.4 cm inside the thighs at the stand pose) |
| Umt | the clip's 14 finger references, `MotionLoader_::hand_joint_pos()` (entity ids 22-28 / 36-42), no policy residual; back to the open pose on exit |

Motor order per hand is the Dex3 SDK order thumb_0, thumb_1, thumb_2,
middle_0, middle_1, index_0, index_1 — identical to the hand MJCF nesting and
to the clip's entity order, so no remapping; right-hand values are already in
the right hand's mirrored joint ranges. The body-only UMT clips carry zeros on
all finger joints, i.e. the open pose. Configure under `dex3:` in
[`config/config.yaml`](config/config.yaml) (`enable: false` when the hands are
not mounted).

### In the simulator

`simulate/config.yaml` selects `scene_g1_dex3.xml` (G1 + Dex3-1, generated by
`scripts/make_g1_dex3_scene.py` from the vendored
`src/assets/robots/unitree_g1/xmls/dex3_1/` MJCFs — the same attach smp_v2
uses, so the sim entity is kinematically the training entity). The bridge
serves `rt/dex3/<side>/{cmd,state}` like the real hands. Unlike the body, the
finger PD is evaluated by MuJoCo `<position>` servos whose gains follow the
command's kp/kd, and the scene integrates with `implicitfast` (mjlab's
integrator): the fingers' tiny inertia makes kd 0.2 unstable both as a
bridge-side PD held across physics steps and as explicit Euler servo damping.
Hand/body contacts are disabled as in training (collision bitmasks); hand/hand
and hand/floor contacts stay. Fingers go limp 1 s after the last command.
Unitree's `g1_dex3_example` from `unitree_sdk2` drives the simulated hands
unchanged, which is the quickest bridge check.

## Streamed inputs: state estimate and depth camera

Two inputs the current UMT policies do not use but the estimator-aware and
depth-distilled policies will. Contract: [`deploy/include/msgs/stream_msgs.h`](../../include/msgs/stream_msgs.h);
design and validation: [`doc/plan_state_estimator_depth_streaming.md`](../../../doc/plan_state_estimator_depth_streaming.md).

| topic | type | content | sim publisher | robot publisher |
| --- | --- | --- | --- | --- |
| `rt/odom_pelvis` | `nav_msgs Odometry_` | pelvis IMU site pose in world; twist in the site frame (= mjlab `imu_lin_vel` / `imu_ang_vel`) | `unitree_mujoco` bridge, 500 Hz | to do (`odom_adapter`) |
| `rt/depth_camera` | `sensor_msgs PointCloud2_` (organized, `UINT16 depth_mm`) | 64×36 planar z-depth, 0 = no return, row 0 = top | `unitree_mujoco` EGL render of `depth_camera`, 30 Hz | to do (`depth_camera_node`, D435i) |

Controller side ([`sources/odom_source.h`](../../include/sources/odom_source.h),
[`sources/depth_source.h`](../../include/sources/depth_source.h)), configured
under `sources:` in [`config/config.yaml`](config/config.yaml):

- `ArticulationData.root_pos_w / root_lin_vel_b / has_odom / odom_age_ms` from
  the odom source; observation terms `base_lin_vel` (3), `motion_anchor_pos_b`
  (3, UMT: clip anchor relative to the torso anchor, xy-aligned at `enter()`,
  `align_z:` per state) and `camera_depth` (H·W, `params: {sensor_name,
  cutoff_distance, min_depth}` like the training term; multi-input
  `deploy.yaml` groups map to the ONNX input names).
- Probe / dump: `build/topic_probe -n lo -s 5 -d probe.npz`; validate with
  `uv run python scripts/check_depth_stream.py --probe-npz probe.npz` and,
  for the simulator's own frames (`depth_camera.dump_dir` in
  `simulate/config.yaml`), `--dump-dir`.

Build the controller in this directory's `build/` (or `bin/`): `param.h`
resolves `config/config.yaml` from the binary's parent directory name.

## Hiphi depth student (`State_HiphiStudent`)

Deploy mirror of smp_v2 `Hiphi-Tracking-Multi-V2-Distill-G1` (checkpoint
`g1_hiphi_umt_multi_v2_distill/2026-09-13_01-24-51_pnp64_res_v0`): a depth
student riding on the frozen 160-obs UMT base, both as ONNX
([`config/policy/hiphi_umt_base/v0`](config/policy/hiphi_umt_base/v0),
[`config/policy/hiphi_student/v0`](config/policy/hiphi_student/v0)), composed
per step exactly as `tasks/hiphi_tracking_umt/actions.py`:

| | value |
| --- | --- |
| UMT base obs (160) | zest_ref with fingers masked to the default pose, motion_anchor_pos_b, motion_anchor_ori_b, base_lin_vel, base_ang_vel, projected_gravity, joint_pos/vel_rel (29), own last action |
| student obs | `student_state` 196 = zest_ref 55 + anchor_ori 6 + ang_vel 3 + gravity 3 + joint_pos/vel over **all 43 joints** (entity order) + last action 43; `student_camera` 1×36×64 |
| student action (43) | `[body 29 \| hand 14]`, term order — not entity order |
| body target | `clamp(q_ref + Σ·π_umt + 0.06·a[0:29], limits)` |
| hand target | `clamp(q_ref_hand + 0.15·a[29:43], limits)` → Dex3Hands |

Needs both streams (`sources:`), the Dex3 hand state (fingers in the proprio)
and a clip converted with `scripts/umt_bundle_to_deploy_npz.py <clip.npz>
<out.npz>` (pnp64 clips carry object arrays cnpy cannot read). Enter from
`Velocity` with `R1 + B`. Re-export the student with
`smp_v2/scripts/export_student_onnx.py` (two named inputs).

Sim-to-sim evaluation (headless, band lowered then released in `Velocity`):
`uv run python scripts/sim_e2e_hiphi_student.py 9`; the mjlab reference for
the same clip comes from `smp_v2/scripts/rollout_hiphi_student.py`. Baseline
2026-09-14, clip `pnp64_9204_13_1520609083`, 9 s from the clip start, the
deploy side with robot + hands only (no table / objects):

| | mjlab play (table + objects) | deploy in unitree_mujoco |
| --- | --- | --- |
| body \|q − q_ref\| mean / last 1 s | 0.075 / 0.070 rad | 0.080 / 0.067 rad |
| hand \|q − q_ref\| mean | 0.449 rad | 0.461 rad |
| Σ·π_umt mean / max | 0.157 / 0.93 rad | 0.175 / 1.93 rad |
| body residual mean / max | 0.110 / 0.272 rad | 0.102 / 0.273 rad |
| hand residual mean / max | 0.472 / 0.717 rad | 0.466 / 0.715 rad |
| fell | no | no |

The residual saturates on both sides (±0.27 rad body, ±0.72 rad fingers —
the student squeezes at full authority with or without an object). The one
deploy-only feature is a transient on the left wrist yaw (joint 21, 1.2 rad
peak error, UMT offset 1.9 rad) — to investigate with objects in the scene.
