#!/usr/bin/env python3
"""Extract one clip from a smp-umt-bundle-v1 zip into a deploy-ready npz.

The C++ deploy loader (deploy/robots/g1_dex3/include/State_UmtMimic.h) reads
the npz with the vendored cnpy, which cannot handle deflate+zip64 entries or
``<U`` string arrays (see DEPLOY_POLICY_FINDINGS.md). This writes an
UNCOMPRESSED npz containing only the numeric arrays, all float32:

    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

It also prints the root/anchor body indices and body-joint ids from the
bundle manifest so they can be checked against ``config/config.yaml``.

Usage:
    umt_bundle_to_deploy_npz.py BUNDLE.zip CLIP_NAME OUT.npz
    umt_bundle_to_deploy_npz.py BUNDLE.zip --list
    umt_bundle_to_deploy_npz.py --onnx policy.onnx   # make batch dim static (in place)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile

import numpy as np

NUMERIC_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
BODY_JOINT_REGEX = r"(?!(?:left|right)_hand_).*_joint"  # tasks.gdm_tracking.env_cfg


def fix_onnx_static_batch(path: str) -> None:
    import onnx  # optional dependency

    m = onnx.load(path)
    changed = False
    for v in list(m.graph.input) + list(m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        if not d.HasField("dim_value"):
            d.ClearField("dim_param")
            d.dim_value = 1
            changed = True
    onnx.checker.check_model(m)
    if changed:
        onnx.save(m, path)
        print(f"[onnx] {path}: batch dim set to 1")
    else:
        print(f"[onnx] {path}: already static")
    for v in list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        print(f"[onnx]   {v.name}: {dims}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", help="smp-umt-bundle-v1 zip")
    ap.add_argument("clip", nargs="?", help="clip name from manifest.json")
    ap.add_argument("out", nargs="?", help="output .npz (uncompressed)")
    ap.add_argument("--list", action="store_true", help="list clips in the bundle and exit")
    ap.add_argument("--onnx", help="patch this policy.onnx to a static batch dim of 1 (in place)")
    args = ap.parse_args()

    if args.onnx:
        fix_onnx_static_batch(args.onnx)
        if not args.bundle:
            return 0

    if not args.bundle:
        ap.error("bundle is required unless only --onnx is given")

    with zipfile.ZipFile(args.bundle) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("schema") != "smp-umt-bundle-v1":
            print(f"error: unexpected schema {manifest.get('schema')!r}", file=sys.stderr)
            return 1

        if args.list or not args.clip:
            for c in manifest["clips"]:
                print(f"{c['name']:40s} {c['split']:6s} {c['frames']:6d} frames  {c['seconds']:.2f}s")
            return 0

        names = [c["name"] for c in manifest["clips"]]
        if args.clip not in names:
            print(f"error: clip {args.clip!r} not in bundle (use --list)", file=sys.stderr)
            return 1
        if not args.out:
            ap.error("out is required")

        with np.load(io.BytesIO(zf.read(f"clips/{args.clip}.npz"))) as data:
            arrays = {}
            for k in NUMERIC_KEYS:
                if k not in data.files:
                    if k == "fps":
                        arrays[k] = np.array([50.0], dtype=np.float32)
                        continue
                    print(f"error: clip is missing {k!r}", file=sys.stderr)
                    return 1
                arrays[k] = np.ascontiguousarray(data[k], dtype=np.float32)

    np.savez(args.out, **arrays)  # NOT savez_compressed: cnpy needs stored entries
    print(f"wrote {args.out}")
    for k, v in arrays.items():
        print(f"  {k:15s} {v.shape} {v.dtype}")

    joint_names = manifest["joint_names"]
    body_names = manifest["body_names"]
    body_joint_ids = [i for i, n in enumerate(joint_names) if re.fullmatch(BODY_JOINT_REGEX, n)]
    hand_joint_ids = [i for i in range(len(joint_names)) if i not in body_joint_ids]
    print("\nlayout (check against deploy/robots/g1_dex3/config/config.yaml):")
    print(f"  root_body_index:   {body_names.index('pelvis')}  # pelvis")
    print(f"  anchor_body_index: {body_names.index('torso_link')}  # torso_link")
    print(f"  body_joint_ids:    {body_joint_ids}")
    print(f"  hand_joint_ids:    {hand_joint_ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
