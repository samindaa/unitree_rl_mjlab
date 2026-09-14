# Unitree Dex3-1 hands (MJCF)

Standalone left/right Dex3-1 hand models, layout mirroring `assets/sharpa_wave/`.
7 DoF per hand (thumb 3, index 2, middle 2), 0.6965 kg per hand, palm-rooted
(`<side>_hand_palm_link`) for MjSpec `attach_body` onto the G1 wrist yaw links.

## Provenance

- Kinematics, inertials, joint limits: `unitree_ros`
  `robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf` (canonical),
  converted via MuJoCo's URDF importer. Cross-checked against
  `avp_teleoperate/assets/g1/g1_body29_hand14.xml` (finger body offsets match
  exactly).
- Visual meshes: unitree_ros STLs (byte-identical to avp_teleoperate copies).
- Collision: reduced convex hulls (`*_convex.STL`) from
  `avp_teleoperate/assets/unitree_hand/meshes/*.convex.stl` (converted
  ASCII→binary STL); thumb_1 uses the URDF's primitive box. Conventions match
  sharpa_wave: visual geoms group 1 non-colliding, collision geoms group 3.
- Mount (for attaching to G1, from Unitree's official assembly in
  `g1_body29_hand14.xml`): palm at pos `(0.0415, +0.003, 0)` left /
  `(0.0415, -0.003, 0)` right in the `*_wrist_yaw_link` frame, identity quat.

## Actuator gains and provisional values

Position servos with kp 1.5, kv 0.2 (all joints) — the PD gains Unitree's
own real-hardware teleop controller sends to every Dex3 motor
(avp_teleoperate `robot_hand_unitree.py`, `Dex3_1_Controller`). forcerange =
URDF effort limits (±2.45 / ±1.4 Nm). No official MuJoCo gains exist
otherwise (Unitree's IsaacLab cfg uses placeholders 100/10/0.1, and e.g.
arXiv:2604.17258 uses arm-scale kp 60 that would saturate the 1.4 N·m
motors at 1.3° error).

Joint armature 0.0001, damping 0.01, frictionloss 0.05 follow Unitree's
"corrected MuJoCo defaults" for the F-1515-214 finger motor
(`docs/Unitree G1 Simulation & Electrical Parameters.html`, class
`motor_dex3_finger`; audit in `docs/g1_dex3_dynamics_audit.html`). They are
Unitree-recommended, not measured. Before 2026-08-27 the XMLs carried
Sharpa-style placeholders (armature 0.003, damping 0.001) — 30x more rotor
inertia, which put the kp 1.5 / kv 0.2 servo at ~3.6 Hz instead of ~19 Hz;
Dex3 checkpoints trained before that date assume the old finger dynamics.

Joint-limit note: sources disagree slightly on thumb_1
(unitree_ros: [-0.611, 1.047]; avp URDF: [-0.724, 0.920]); unitree_ros wins.
