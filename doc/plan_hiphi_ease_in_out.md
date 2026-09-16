# Plan: eased entry into, and held exit from, the hiphi motion

Status: **implemented and validated in simulation (2026-09-16)** — `HiphiStack` + `State_HiphiEaseIn` / `State_HiphiStudent` / `State_HiphiEaseOut`; results in the g1_dex3 README ("Eased entry / held exit"). Step 0 (offline padded-clip cross-check) was skipped; the online version validated directly.

## Problem

`Velocity -> HiphiStudent` applies the clip's first frame in one 20 ms step:
on `pnp64_9204_13` the first-frame body pose differs from the Velocity stance
by up to **1.2 rad** (mean 0.27 rad over the 29 body joints), the fingers go
from the rest pose to a half-closed grasp, and the clip's own first frame
already moves (0.33 rad/s). The composed target `q_ref + Σ·π_umt + res` jumps
by that amount, the stiff PD follows it, and the robot lurches. At the end of
the clip the state times out into `Velocity`, which is the same jump in
reverse — and it lets go of whatever the hands hold.

Wanted: a graceful approach from wherever the robot is into frame 0, the
motion, then a smooth deceleration into the final frame held indefinitely
(no return to locomotion until the operator says so).

## Approach: ease through the *reference*, keep the balance controller in the loop

Do what `g1_spinkick_example/pkl_to_csv.py` does offline — prepend
`ease_in_cubic` transition frames from a standing pose to frame 0, append
`ease_out_cubic` frames into a hold pose, then hold — but **online**, as
synthetic reference frames fed to the same tracking stack, starting from the
robot's **measured** pose rather than a canned standing pose:

- The UMT base is a reference-tracking balance controller and was trained
  with RSI (episodes start with the robot *on* the reference). A reference
  that starts exactly at the current pose and bends smoothly into the clip
  is the deployment analogue of RSI; a FixStand-style raw PD blend would
  drop the balance controller for a second while standing on the legs.
- One code path for sim and robot; no clip files edited by hand.

Easing exactly as the reference script (its recipe transferred to the real
G1):

```
ease_in_cubic(t)  = t^3            # start transition: standing pose -> frame 0
ease_out_cubic(t) = 1 - (1 - t)^3  # end transition:   last frame -> hold pose
joints / positions: lerp(a, b, e)     rotations: slerp(a, b, e)
```

with `transition_duration` 1.0 s (their default) for both, configurable.
Velocities of the synthetic frames come from finite differences at 50 Hz —
the same way the clip's own velocities were produced (csv_to_npz), so the
`zest_ref` velocity terms stay consistent.

## FSM

```
Velocity --RB+B--> HiphiEaseIn --(blend done)--> HiphiStudent --(clip end)--> HiphiEaseOut (hold)
                       |                              |                              |
                  LT+B / error                   LT+B / error                    LT+B -> Passive
                       v                              v                          RT+A -> Velocity (operator, explicit)
                    Passive                        Passive
```

Three thin states over one shared **`HiphiStack`** (the UMT env, the student
env, the clip loader, the alignment, the composition and the 50 Hz policy
thread that `State_HiphiStudent` owns today). The stack lives across the
three states so the UMT's own last-action buffer, the alignment and the hand
targets are continuous through the hand-overs; each state only chooses the
stack's *reference source* and whether the *student residual* is applied:

| state | reference frames | student residual | fingers | leaves when |
| --- | --- | --- | --- | --- |
| `HiphiEaseIn` | synthetic: measured pose -> clip frame 0, `ease_in_cubic`, `T_in` | off (UMT only) | rest pose -> clip frame-0 fingers, same easing | blend complete (`t >= T_in`) -> `HiphiStudent`; abort -> `Passive` |
| `HiphiStudent` | clip (as today) | on | clip | `t >= time_end` -> `HiphiEaseOut` (not `Velocity`) |
| `HiphiEaseOut` | synthetic: last frame -> hold pose, `ease_out_cubic`, `T_out`, then the hold pose forever with zero velocities | off | hold at the last finger targets (keeps the grasp) | only the operator: `LT+B` -> Passive, `RT+A` -> Velocity |

Details that matter:

- **Ease-in start pose** = measured body joints, rest-pose fingers, and the
  robot's current pelvis / torso pose from the odom stream (site -> pelvis
  origin -> torso FK, already in `UmtAnchor.h`). The clip is aligned first
  (yaw + xy, as today), so frame 0's mapped anchor differs from the robot's
  only in z / roll / pitch; those are blended too. Without a state estimate
  (no-SE stack) only joints and anchor orientation blend, positions come
  from frame 0.
- **`ease_in_cubic` arrives at frame 0 at full speed** (t³ has its largest
  slope at the end). That is the reference recipe; keep `ease: in_cubic |
  in_out_cubic` selectable and try both in sim — `in_out_cubic` arrives
  with zero velocity, which the UMT base may prefer when the clip's frame 0
  is itself nearly static (0.05 m/s anchor speed here).
- **Hold pose** = the clip's last frame (`hold_pose: last`, the user's
  request); `hold_pose: default` gives the spin-kick script's "safe standing
  pose" exit as an option. The pnp64 clip ends almost static (0.11 rad/s,
  anchor 0.01 m/s), so `T_out` mostly zeroes the velocity terms; the general
  form covers clips that end mid-motion.
- **Hand-overs reset nothing but what training reset**: entering
  `HiphiStudent` from `HiphiEaseIn` resets the student env only (its last
  action, like a fresh episode on the reference); the UMT env keeps running.
  Entering `HiphiEaseOut` keeps both running and just stops applying the
  residual (`res_scale` -> 0 blended over the first 0.2 s to avoid a step).
- **Abort conditions** in the eased states (registered checks): stale
  streams (as now), body tracking error > 0.5 rad on any joint or anchor
  error > 0.3 m -> `Passive`; `HiphiEaseOut` additionally never times out.
- `Velocity`'s `RB+B` now targets `HiphiEaseIn`; the direct
  `Velocity -> HiphiStudent` edge goes away (keyboard `4` unchanged).

## Implementation steps

| # | item | files | size |
| --- | --- | --- | --- |
| 0 | **Offline cross-check first**: `scripts/pad_deploy_clip.py` — same padding as `pkl_to_csv.py` on a deploy npz (ease-in from a chosen standing pose, ease-out to hold, hold padding, finite-difference velocities). Play the padded clip in mjlab (`rollout_hiphi_student.py`) and through today's `HiphiStudent` in sim: shows the easing recipe works with this UMT base before any FSM code | `scripts/` | S |
| 1 | `HiphiStack` — lift the two envs / loader / alignment / compose / probe / policy thread out of `State_HiphiStudent` into a shared object with a swappable `MotionSource` (clip loader or synthetic) and a `residual_gain` (0..1) | `deploy/robots/g1_dex3/include/HiphiStack.h`, `src/HiphiStack.cpp` | M |
| 2 | `SyntheticMotion` — builds `MotionLoader_`-compatible frames from (start pose, target frame, easing, duration) incl. slerp and finite-difference velocities; `MotionLoader_` gets an in-memory constructor | `include/SyntheticMotion.h`, `State_UmtMimic.h` | M |
| 3 | `State_HiphiEaseIn`, `State_HiphiEaseOut`; `State_HiphiStudent` becomes a thin state over the stack; config blocks (`T_in`, `T_out`, `ease`, `hold_pose`, abort thresholds); `end_state: HiphiEaseOut` | `include/State_Hiphi*.h`, `src/`, `config/config.yaml`, `main.cpp` | M |
| 4 | Validation: `sim_e2e_hiphi_student.py` drives `Velocity -> EaseIn -> Student -> EaseOut`, reports the **max per-step target jump** (today ≈ 1.2 rad/step at entry; goal < 0.05 rad/step), joint/anchor error through both transitions, pelvis z, hold stability over 10 s; mjlab comparison via the padded clip from step 0 | `scripts/` | S–M |
| 5 | Docs: README runbook + FSM table, skill notes | | S |

Steps 0 and 2 are independent; 1 before 3; 4 after 3.

## Open questions (defaults proposed)

- `T_in` / `T_out`: 1.0 s / 1.0 s (reference script). Sim sweep 0.5–2 s.
- Ease-in curve: `in_cubic` (recipe) vs `in_out_cubic` — decide from the
  sim sweep (per-step jump and UMT offset magnitude at the hand-over).
- Should `HiphiEaseOut` ever apply the student residual (e.g. to keep
  squeezing an object)? Proposed no: the fingers hold their last *targets*
  (which already include the residual squeeze at the last step), and the
  UMT base holds the body.
- Exit from the hold into locomotion: for now the explicit `RT+A` jump into
  `Velocity`; a `HiphiEaseOut -> Velocity` eased exit (blend to the Velocity
  stance, then hand over) is the same machinery and can follow.
