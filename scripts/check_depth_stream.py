#!/usr/bin/env python3
"""Validate the simulator's streamed inputs (deploy/include/msgs/stream_msgs.h).

Two independent checks:

1. ``--dump-dir DIR``: frames the simulator wrote with ``depth_camera.dump_dir``
   (16-bit PNG in mm + ``.txt`` sidecar with sim time and qpos). Each frame is
   re-rendered from the sidecar's qpos with MuJoCo's Python renderer through
   the scene's ``depth_camera`` and compared pixel by pixel. Checks the
   metric conversion, the row order (row 0 = top), the background convention
   (0, not the far plane) and that the streamed image really is the scene
   camera's view.

2. ``--probe-npz FILE``: a dump from ``topic_probe -d FILE``. Reports odom /
   depth rates and jitter from the receive timestamps, source-vs-receive
   latency, and depth statistics (valid fraction, range).

Python cannot join the DDS domain (see doc), which is why both inputs come
through files.

    uv run python scripts/check_depth_stream.py --dump-dir /tmp/depth_dump
    uv run python scripts/check_depth_stream.py --probe-npz /tmp/probe.npz
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_dex3.xml"
CAMERA = "depth_camera"


def read_png16(path: Path) -> np.ndarray:
    """Minimal 16-bit greyscale PNG reader (all five scanline filters; no Pillow)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        if kind == b"IHDR":
            width, height = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            assert body[8] == 16 and body[9] == 0, "expected 16-bit greyscale"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride, bpp = width * 2, 2
    out = np.zeros((height, width), dtype=np.uint16)
    prev = np.zeros(stride, dtype=np.int32)
    for r in range(height):
        f = raw[r * (stride + 1)]
        line = np.frombuffer(raw[r * (stride + 1) + 1 : (r + 1) * (stride + 1)], dtype=np.uint8).astype(np.int32)
        if f == 2:  # Up
            line = (line + prev) & 0xFF
        else:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                if f == 1:  # Sub
                    line[i] = (line[i] + a) & 0xFF
                elif f == 3:  # Average
                    line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
                elif f == 4:  # Paeth
                    p_ = a + b - c
                    pa, pb, pc = abs(p_ - a), abs(p_ - b), abs(p_ - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    line[i] = (line[i] + pred) & 0xFF
        out[r] = line.astype(np.uint8).view(">u2")
        prev = line
    return out


SCENE_OPTION = mujoco.MjvOption()
for _g in range(mujoco.mjNGROUP):
    SCENE_OPTION.geomgroup[_g] = 1 if _g <= 2 else 0  # visual groups only, as streamed


def render_depth(model, data, renderer, qpos: np.ndarray) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=CAMERA, scene_option=SCENE_OPTION)
    z = renderer.render().astype(np.float64)  # metric z along the optical axis
    zfar = model.vis.map.zfar * model.stat.extent
    z[z >= 0.99 * zfar] = 0.0  # stream convention: no return = 0
    return z


def check_dump_dir(dump_dir: Path) -> int:
    pngs = sorted(dump_dir.glob("depth_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not pngs:
        print(f"no depth_*.png in {dump_dir}")
        return 1
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # No MSAA: the stream (and the training ray tracer) sample pixel centres;
    # MSAA-resolved depth is biased by +2-6 cm at grazing angles (measured).
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    first = read_png16(pngs[0])
    h, w = first.shape
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.enable_depth_rendering()
    # No shadows/reflections in the stream either.
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

    worst = 0.0
    fails = 0
    for png in pngs:
        side = png.with_suffix(".txt").read_text().split()
        assert side[0] == "time" and side[2] == "qpos", png
        qpos = np.array(side[3:], dtype=np.float64)
        assert qpos.size == model.nq, (qpos.size, model.nq)
        streamed = read_png16(png).astype(np.float64) * 1e-3
        ref = render_depth(model, data, renderer, qpos)

        bg_s, bg_r = streamed == 0, ref == 0
        both = ~bg_s & ~bg_r
        err = np.abs(streamed - ref)[both]
        bg_mismatch = np.mean(bg_s != bg_r)
        flipped = np.abs(streamed[::-1] - ref)[both[::-1] & both].mean() if both.any() else np.nan
        same = err.mean() if err.size else np.nan
        ok = err.size > 0 and np.median(err) < 0.01 and bg_mismatch < 0.05 and (np.isnan(flipped) or same < flipped)
        worst = max(worst, float(np.median(err)) if err.size else worst)
        fails += not ok
        print(f"{png.name}: t={float(side[1]):.3f}s valid {both.mean()*100:5.1f}%  |dz| median {np.median(err) if err.size else np.nan:.4f} "
              f"p95 {np.percentile(err, 95) if err.size else np.nan:.4f} max {err.max() if err.size else np.nan:.3f} m  "
              f"bg mismatch {bg_mismatch*100:.1f}%  (row-flipped mean err {flipped:.3f} vs {same:.3f})  {'OK' if ok else 'FAIL'}")
    print(f"{len(pngs)} frames, {fails} failed, worst median |dz| {worst:.4f} m")
    return 1 if fails else 0


def check_probe_npz(path: Path) -> int:
    d = np.load(path)
    rc = 0
    if "odom_t" in d:
        t, rx = d["odom_t"], d["odom_rx_wall"]
        dt = np.diff(rx)
        print(f"odom: {t.size} samples over {rx[-1]-rx[0]:.2f}s -> {1/np.mean(dt):.1f} Hz "
              f"(jitter p95 {np.percentile(dt,95)*1e3:.2f} ms, max gap {dt.max()*1e3:.1f} ms)")
        print(f"      sim dt mean {np.mean(np.diff(t))*1e3:.2f} ms; pos range x[{d['odom_pos'][:,0].min():+.3f},{d['odom_pos'][:,0].max():+.3f}] "
              f"z[{d['odom_pos'][:,2].min():.3f},{d['odom_pos'][:,2].max():.3f}]; |quat|-1 max {np.abs(np.linalg.norm(d['odom_quat_wxyz'],axis=1)-1).max():.1e}")
        v, w = d["odom_lin_vel_b"], d["odom_ang_vel_b"]
        print(f"      |v_b| max {np.linalg.norm(v,axis=1).max():.3f} m/s   |w_b| max {np.linalg.norm(w,axis=1).max():.3f} rad/s")
    else:
        print("odom: nothing received")
        rc = 1
    if "depth_t" in d:
        t, rx, fr = d["depth_t"], d["depth_rx_wall"], d["depth_mm"]
        dt = np.diff(rx)
        valid = fr > 0
        print(f"depth: {fr.shape[0]} frames {fr.shape[2]}x{fr.shape[1]} -> {1/np.mean(dt):.1f} Hz "
              f"(jitter p95 {np.percentile(dt,95)*1e3:.2f} ms); valid {valid.mean()*100:.1f}%; "
              f"range {fr[valid].min() if valid.any() else 0}..{fr[valid].max() if valid.any() else 0} mm; "
              f"sim-time span {t[-1]-t[0]:.2f}s vs wall {rx[-1]-rx[0]:.2f}s")
        col = fr[:, :, fr.shape[2] // 2]
        print(f"       centre column mean by row (top->bottom, mm): {np.round(col.mean(axis=0)[:: max(1, fr.shape[1]//6)]).astype(int).tolist()}")
    else:
        print("depth: nothing received")
        rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", type=Path)
    ap.add_argument("--probe-npz", type=Path)
    args = ap.parse_args()
    if not args.dump_dir and not args.probe_npz:
        ap.error("give --dump-dir and/or --probe-npz")
    rc = 0
    if args.dump_dir:
        rc |= check_dump_dir(args.dump_dir)
    if args.probe_npz:
        rc |= check_probe_npz(args.probe_npz)
    return rc


if __name__ == "__main__":
    sys.exit(main())
