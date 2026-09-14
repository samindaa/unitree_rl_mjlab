#!/usr/bin/env python3
"""Generate the unitree_mujoco scene for the G1 (29dof) with Dex3-1 hands.

Takes the flat 29-motor simulator scene (``scene_g1.xml``) and performs the
same MjSpec surgery smp_v2 uses for its G1+Dex3 entity
(``smp_v2/src/retarget/dex3.py::attach_dex3_hands``):

  * delete the fixed rubber-hand geoms (and their mesh assets),
  * attach the vendored Dex3-1 palm to each ``*_wrist_yaw_link`` at Unitree's
    official mount (identity rotation, x = 0.0415, 3 mm medial offset),

then adapts the result to the simulator's DDS bridge conventions
(``simulate/src/unitree_sdk2_bridge.h``):

  * the hand XMLs' ``<position>`` servos (kp 1.5, kv 0.2, ctrlrange = joint
    range) are KEPT, renamed ``<side>_hand_<finger>`` (the ``_ctrl`` suffix
    dropped). The bridge writes each Dex3 ``HandCmd`` q into ``ctrl`` and its
    kp/kd into the servo's gain/bias parameters, so MuJoCo evaluates the
    finger PD at every physics step. A bridge-side PD on ``<motor>``
    actuators (the body's scheme) is NOT usable for the fingers: with
    ~3e-4 kg.m^2 of reflected inertia, kd 0.2 held across the multi-step
    bursts the simulator's physics loop takes makes the explicit damping
    term unstable (a +-10 rad/s limit cycle — verified).
  * the integrator is set to ``implicitfast`` (mjlab's), because the servo
    damping on the light distal finger links is unstable under explicit
    Euler at the 2 ms step,
  * hand <-> robot-body contacts are disabled through collision bitmasks
    (hand geoms 2/2, floor conaffinity 3), mirroring the training entity's
    hand/body exclusion; hand <-> hand and hand <-> floor contacts stay,
  * the ``jointpos`` / ``jointvel`` / ``jointactuatorfrc`` sensor blocks are
    rebuilt to cover all 43 actuators in actuator order (the bridge indexes
    ``sensordata`` as three contiguous blocks of ``nu``), body motors first,
    then left hand, then right hand; the IMU / frame sensors follow.

Actuator / joint order of the hands is the Dex3 SDK motor order (which is
also the hand XML's nesting order and the mjlab entity order used by the
deploy clips): thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0,
index_1.

The output is a flat, self-contained MJCF (like ``scene_g1.xml`` itself) so
it loads in the C++ simulator, ``scripts/sim_viser_mirror.py`` and any
MuJoCo >= 3.2 without further tooling. Re-run after editing ``scene_g1.xml``
or the vendored hand XMLs:

    uv run python scripts/make_g1_dex3_scene.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_DIR = REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls"
DEX3_DIR = XML_DIR / "dex3_1"

RUBBER_HAND_MESHES = {"left_rubber_hand", "right_rubber_hand"}

# Palm pose in the wrist yaw link frame (identity rotation), from Unitree's
# official G1 assembly (avp_teleoperate g1_body29_hand14.xml); identical to
# smp_v2 retarget/dex3.py MOUNT_POS.
MOUNT_POS = {
    "left": (0.0415, 0.003, 0.0),
    "right": (0.0415, -0.003, 0.0),
}

DEX3_MOTOR_ORDER = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)


def attach_dex3_hands(spec: mujoco.MjSpec) -> None:
    """smp_v2 ``attach_dex3_hands``: drop the rubber hands, attach the palms."""
    for geom in list(spec.geoms):
        if geom.meshname in RUBBER_HAND_MESHES:
            spec.delete(geom)
    for mesh in list(spec.meshes):
        if mesh.name in RUBBER_HAND_MESHES:
            spec.delete(mesh)
    for side in ("left", "right"):
        hand = mujoco.MjSpec.from_file(str(DEX3_DIR / f"{side}_dex3_1" / f"{side}_dex3_1.xml"))
        wrist = spec.body(f"{side}_wrist_yaw_link")
        frame = wrist.add_frame(pos=np.array(MOUNT_POS[side]))
        frame.attach_body(hand.body(f"{side}_hand_palm_link"), "", "")


HAND_SERVO_KP = 1.5
HAND_SERVO_KV = 0.2


def rename_hand_actuators(spec: mujoco.MjSpec) -> None:
    """Drop the ``_ctrl`` suffix from the attached hands' position servos and
    check they are what the bridge expects (position servo, kp 1.5, kv 0.2,
    one per finger joint, in SDK motor order after the 29 body motors)."""
    hand_acts = [a for a in spec.actuators if "_hand_" in a.name]
    expected = [f"{s}_hand_{f}_ctrl" for s in ("left", "right") for f in DEX3_MOTOR_ORDER]
    assert [a.name for a in hand_acts] == expected, [a.name for a in hand_acts]
    for act in hand_acts:
        assert act.gaintype == mujoco.mjtGain.mjGAIN_FIXED and act.biastype == mujoco.mjtBias.mjBIAS_AFFINE
        assert np.isclose(act.gainprm[0], HAND_SERVO_KP) and np.isclose(act.biasprm[1], -HAND_SERVO_KP)
        assert np.isclose(act.biasprm[2], -HAND_SERVO_KV), act.biasprm[:3]
        act.name = act.name[: -len("_ctrl")]


# Collision bitmasks. G1 body geoms keep MuJoCo's default contype=1 /
# conaffinity=1; hand collision geoms get 2/2 and the floor conaffinity 3, so
# hand<->hand, finger<->finger and hand<->floor contacts stay alive while
# hand<->body contacts are disabled (a pair collides when
# contype_a & conaffinity_b or contype_b & conaffinity_a is non-zero). Any
# object added to the scene that the hands should touch needs conaffinity 3.
HAND_CONTYPE = 2
HAND_CONAFFINITY = 2
FLOOR_CONAFFINITY = 3


def exclude_hand_body_collisions(spec: mujoco.MjSpec) -> None:
    """Disable hand <-> robot-body contacts, as the training entity does.

    smp_v2 (``tasks/gdm_tracking/assets.py::_exclude_hand_body_collisions``)
    excludes every hand-body / robot-body pair: at the knees-bent default
    pose the thumbs sit inside the thigh meshes and the resulting contact
    forces deflect the fingers (and kick the robot). The policy never saw
    those contacts, so the simulator must not produce them either.
    """
    n = 0
    for geom in spec.geoms:
        if geom.name.startswith(("left_hand_", "right_hand_")) and geom.name.endswith("_collision"):
            geom.contype = HAND_CONTYPE
            geom.conaffinity = HAND_CONAFFINITY
            n += 1
    assert n == 16, n  # palm + 7 finger links per hand
    floor = spec.geom("floor")
    floor.conaffinity = FLOOR_CONAFFINITY


def rebuild_joint_sensors(spec: mujoco.MjSpec) -> None:
    """Rebuild the per-motor sensor blocks in actuator order.

    Layout required by the bridge: jointpos x nu, jointvel x nu,
    jointactuatorfrc x nu, then everything else (IMU, frame sensors) in its
    original order.
    """
    motor_types = {
        mujoco.mjtSensor.mjSENS_JOINTPOS,
        mujoco.mjtSensor.mjSENS_JOINTVEL,
        mujoco.mjtSensor.mjSENS_JOINTACTFRC,
    }
    others = []
    for s in list(spec.sensors):
        if s.type in motor_types:
            spec.delete(s)
        else:
            others.append(
                dict(name=s.name, type=s.type, objtype=s.objtype, objname=s.objname,
                     reftype=s.reftype, refname=s.refname, cutoff=s.cutoff,
                     noise=s.noise, datatype=s.datatype, needstage=s.needstage)
            )
            spec.delete(s)

    joints = [a.target for a in spec.actuators]
    for suffix, stype in (
        ("_pos", mujoco.mjtSensor.mjSENS_JOINTPOS),
        ("_vel", mujoco.mjtSensor.mjSENS_JOINTVEL),
        ("_torque", mujoco.mjtSensor.mjSENS_JOINTACTFRC),
    ):
        for act, joint in zip(spec.actuators, joints):
            spec.add_sensor(
                name=f"{act.name}{suffix}",
                type=stype,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                objname=joint,
            )
    for o in others:
        spec.add_sensor(**o)


# Head depth camera, verbatim from the training env
# (smp_v2 tasks/hiphi_tracking_multi_distill/env_cfg.py: HEAD_CAMERA_POS,
# HEAD_CAMERA_PITCH_RAD, DEPTH_CAMERA_FOVY, _depth_camera_quat). Nominal D435
# mount from Unitree's g1_29dof_rev_1_0.urdf d435_joint, parent torso_link.
DEPTH_CAMERA_NAME = "depth_camera"
HEAD_CAMERA_POS = (0.0576235, 0.01753, 0.42987)
HEAD_CAMERA_PITCH_RAD = 0.8307767239493009  # ~47.6 deg down
DEPTH_CAMERA_FOVY = 58.0


def depth_camera_quat(pitch_rad: float = HEAD_CAMERA_PITCH_RAD) -> np.ndarray:
    """MuJoCo camera (looks along -Z, +X right, +Y up) for the URDF's ROS-style
    mount (+X view, +Z up, pitched about +Y): view (cos p, 0, -sin p) in the
    torso frame, camera-up ~ torso +z, image-right = torso -y."""
    p = pitch_rad
    mat = np.array(
        [
            [0.0, np.sin(p), -np.cos(p)],
            [-1.0, 0.0, 0.0],
            [0.0, np.cos(p), np.sin(p)],
        ]
    )
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def add_depth_camera(spec: mujoco.MjSpec) -> None:
    spec.body("torso_link").add_camera(
        name=DEPTH_CAMERA_NAME,
        pos=np.array(HEAD_CAMERA_POS),
        quat=depth_camera_quat(),
        fovy=DEPTH_CAMERA_FOVY,
    )


def relocate_mesh_files(spec: mujoco.MjSpec) -> None:
    """Make the G1 mesh paths relative to the xmls/ directory (no meshdir).

    The attached hand meshes are left alone here: MjSpec still resolves them
    through the hand model's own ``meshes/`` dir when compiling, but
    ``to_xml`` emits only their bare file names. ``fix_hand_mesh_paths``
    rewrites those in the exported text.
    """
    spec.meshdir = ""
    for mesh in spec.meshes:
        if not mesh.name.startswith(("left_hand_", "right_hand_")):
            mesh.file = f"assets/{Path(mesh.file).name}"


_HAND_MESH_RE = re.compile(r'(<mesh name="(left|right)_hand_[^"]*"[^>]*file=")([^"/]+)(")')


def fix_hand_mesh_paths(xml: str) -> str:
    return _HAND_MESH_RE.sub(
        lambda m: f"{m.group(1)}dex3_1/{m.group(2)}_dex3_1/meshes/{m.group(3)}{m.group(4)}", xml
    )


# attach() imports each hand's implicit main default class as an UNNAMED
# nested <default>, which MuJoCo refuses to re-parse ("empty class name") and
# which the Python MjSpec API cannot reach. Nothing references them (every
# hand joint/geom spells out its attributes), so drop them from the text.
_UNNAMED_NESTED_DEFAULT_RE = re.compile(  # to_xml indents nested classes by 4
    r"\n    <default>\n(?:      <[^\n]*\n)*    </default>\n"
)


def strip_unnamed_nested_defaults(xml: str) -> str:
    xml, n = _UNNAMED_NESTED_DEFAULT_RE.subn("", xml)
    assert n == 2, f"expected 2 imported hand default classes, found {n}"
    assert 'class="' not in xml, "unexpected class reference in generated scene"
    return xml


def build(src: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(src))
    spec.modelname = "scene_g1_dex3"
    # The hand servos' kv (0.2) on the distal finger links (~1e-4 kg.m^2 incl.
    # armature) exceeds the explicit-damping stability bound at the 2 ms step
    # (kv*h/I ~ 4 > 2): under Euler those joints chatter at the force limit and
    # never reach the target. implicitfast integrates actuator velocity terms
    # implicitly (it is also what mjlab trains with). The body's bridge-side
    # PD torques are velocity-independent <motor> forces, unaffected.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    attach_dex3_hands(spec)
    rename_hand_actuators(spec)
    exclude_hand_body_collisions(spec)
    rebuild_joint_sensors(spec)
    add_depth_camera(spec)
    relocate_mesh_files(spec)
    return spec


def check(spec: mujoco.MjSpec, xml_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_nu = 29
    assert model.nu == body_nu + 14, model.nu
    assert model.nq == 7 + body_nu + 14, model.nq
    assert model.nsensor >= 3 * model.nu
    for i in range(model.nu):
        act = model.actuator(i).name
        jnt = model.joint(model.actuator_trnid[i, 0]).name
        for k, suffix in enumerate(("_pos", "_vel", "_torque")):
            s = model.sensor(k * model.nu + i)
            assert s.name == act + suffix, (s.name, act + suffix)
            assert model.joint(model.sensor_objid[s.id]).name == jnt
            assert model.sensor_adr[s.id] == k * model.nu + i
    hand_names = [model.actuator(i).name for i in range(body_nu, model.nu)]
    expected = [f"{s}_hand_{f}" for s in ("left", "right") for f in DEX3_MOTOR_ORDER]
    assert hand_names == expected, hand_names
    for i in range(body_nu, model.nu):
        assert model.actuator_gainprm[i, 0] == HAND_SERVO_KP
        assert model.actuator_biasprm[i, 1] == -HAND_SERVO_KP and model.actuator_biasprm[i, 2] == -HAND_SERVO_KV
        jnt = model.actuator_trnid[i, 0]
        assert model.actuator_ctrllimited[i] and np.allclose(model.actuator_ctrlrange[i], model.jnt_range[jnt])
    # Joints are tree-ordered (each hand's 7 joints follow its wrist, as in
    # the mjlab entity the deploy clips are expressed in); within a hand the
    # qpos order must be the SDK motor order.
    for side in ("left", "right"):
        adrs = [model.joint(f"{side}_hand_{f}_joint").qposadr[0] for f in DEX3_MOTOR_ORDER]
        assert adrs == list(range(adrs[0], adrs[0] + 7)), (side, adrs)
        assert model.joint(model.jnt_qposadr.tolist().index(adrs[0]) - 1).name == f"{side}_wrist_yaw_joint"
    for side, sign in (("left", 1), ("right", -1)):
        # thumb_1 range is mirrored between the sides (README joint-limit note)
        lo, hi = model.joint(f"{side}_hand_thumb_1_joint").range
        assert (hi > 1.0) if sign > 0 else (lo < -1.0)
    # collision masks: hand geoms collide with each other and the floor only
    hand_geoms = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_collision")
                  and model.geom(g).name.startswith(("left_hand_", "right_hand_"))]
    assert len(hand_geoms) == 16
    assert all(model.geom_contype[g] == HAND_CONTYPE and model.geom_conaffinity[g] == HAND_CONAFFINITY for g in hand_geoms)
    floor = model.geom("floor").id
    thigh = model.geom(model.body("left_hip_roll_link").geomadr[0] + 1).id  # the colliding mesh
    def collides(a, b):
        return bool(model.geom_contype[a] & model.geom_conaffinity[b] or model.geom_contype[b] & model.geom_conaffinity[a])
    assert collides(hand_geoms[0], floor) and collides(hand_geoms[0], hand_geoms[9])
    assert not collides(hand_geoms[0], thigh) and collides(thigh, floor)
    # depth camera: on torso_link, training pose / fovy
    cam = model.camera(DEPTH_CAMERA_NAME)
    assert model.body(model.cam_bodyid[cam.id]).name == "torso_link"
    assert np.allclose(model.cam_pos[cam.id], HEAD_CAMERA_POS)
    assert np.allclose(model.cam_fovy[cam.id], DEPTH_CAMERA_FOVY)
    q = depth_camera_quat()
    assert np.allclose(model.cam_quat[cam.id], q) or np.allclose(model.cam_quat[cam.id], -q)
    print(f"{xml_path.name}: nq={model.nq} nu={model.nu} nsensor={model.nsensor} "
          f"nbody={model.nbody} ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--src", type=Path, default=XML_DIR / "scene_g1.xml")
    p.add_argument("--out", type=Path, default=XML_DIR / "scene_g1_dex3.xml")
    args = p.parse_args()

    spec = build(args.src)
    xml = strip_unnamed_nested_defaults(fix_hand_mesh_paths(spec.to_xml()))
    xml = re.sub(r"<mujoco model=\"[^\"]*\">",
                 f'<!-- GENERATED by scripts/{Path(__file__).name} from {args.src.name}; do not edit by hand. -->\n\\g<0>',
                 xml, count=1)
    args.out.write_text(xml)
    check(spec, args.out)


if __name__ == "__main__":
    main()
