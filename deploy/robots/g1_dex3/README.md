# g1_dex3 — G1 (29dof) + Dex3 hands, UMT tracking deploy

Deploys smp_v2 task **`Umt-Tracking-G1-No-State-Estimation`**
(`smp_v2/src/tasks/umt`). Same FSM/isaaclab-mirror stack as `../g1`; the
differences are the command layout and the action space:

| | `g1` Mimic | `g1_dex3` Umt |
| --- | --- | --- |
| Reference obs | `motion_command` (q_ref + qd_ref, 58) | `zest_ref` (55: z, roll, pitch, v_b, ω_b, g_b, **43** q_ref) |
| Anchor obs | `motion_anchor_ori_b` from waist joints | `motion_anchor_ori_b` from clip `body_quat_w[15]` (torso_link) |
| Action | `JointPositionAction` (default-pose offset) | `ReferenceJointPositionAction` (`q_cmd = q_ref + Σa`) |
| Clip npz | 29 joints / 1 body | 43 joints / 46 bodies (fingers frozen; hands not driven yet) |
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

## Dex3 (later)

`MotionLoader_::hand_joint_pos()` already exposes the 14 finger references
(entity ids 22-28 / 36-42); publish them on the Dex3 hand-command topics from
`State_UmtMimic::run()` once the hands are mounted.
