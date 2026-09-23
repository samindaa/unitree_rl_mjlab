#!/usr/bin/env python3
"""Headless sim-to-sim evaluation of the hiphi student stack (State_HiphiStudent).

Runs unitree_mujoco and g1_dex3_ctrl from a non-TTY shell (see the recipe in
doc/deploy_architecture.md), drives the FSM over the simulator's UDP command
tap (FixStand -> Velocity -> lower the elastic band -> release -> HiphiStudent
for N seconds -> Passive), then summarises the controller log and the probe
npz the state dumps on exit (/tmp/hiphi_probe_<ts>.npz): body / hand tracking
error against the clip, UMT and residual magnitudes, per-joint worst offenders,
hand-state liveness and the pelvis height before / after.

Requires: simulate/build/unitree_mujoco and deploy/robots/g1_dex3/build/
g1_dex3_ctrl built, DISPLAY (:0) for the simulator window, keyboard joystick
in simulate/config.yaml with "1","2","4","0" mapped (default) and the "9"/"8"
elastic-band keys. Python never joins DDS (cyclonedds version clash).

    uv run python scripts/sim_e2e_hiphi_student.py [seconds]
"""
import glob, os, re, socket, struct, subprocess, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S = os.environ.get("E2E_LOG_DIR", "/tmp")
env = dict(os.environ, DISPLAY=":0")
def start(cmd, cwd, log):
    return subprocess.Popen(["setsid", "script", "-q", "-f", "-c", cmd, log], cwd=cwd, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def key(k):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(struct.pack("<I", 0x4D4A4B59) + k.encode(), ("127.0.0.1", 9871)); s.close()
subprocess.run(["pkill", "-x", "unitree_mujoco"]); subprocess.run(["pkill", "-x", "g1_dex3_ctrl"]); time.sleep(0.5)
before = set(glob.glob("/tmp/hiphi_probe_*.npz"))
start("./unitree_mujoco", f"{REPO}/simulate/build", f"{S}/e2e2_sim.log"); time.sleep(5)
start("./g1_dex3_ctrl --network lo", f"{REPO}/deploy/robots/g1_dex3/build", f"{S}/e2e2_ctrl.log")
for _ in range(60):
    time.sleep(0.5)
    log = open(f"{S}/e2e2_ctrl.log", errors="ignore").read()
    if "Connected to robot" in log or "critical" in log: break
time.sleep(1.0)
tap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); tap.bind(("127.0.0.1", 9870))
def pelvis_z():
    tap.setblocking(False)
    try:
        while True: tap.recv(65536)
    except BlockingIOError: pass
    tap.setblocking(True); pkt = tap.recv(65536)
    return struct.unpack_from("<d", pkt, 16 + 2 * 8)[0]
CLIP = os.environ.get("E2E_CLIP", "pnp64_9204_13_1520609083")   # motion-library stem/index to play (config.yaml motions:)
for attempt in range(3):
    if attempt:
        print("  harness: robot went down during the band release, restarting")
        subprocess.run(["pkill", "-x", "g1_dex3_ctrl"]); subprocess.run(["pkill", "-x", "unitree_mujoco"]); time.sleep(0.5)
        start("./unitree_mujoco", f"{REPO}/simulate/build", f"{S}/e2e2_sim.log"); time.sleep(5)
        start("./g1_dex3_ctrl --network lo", f"{REPO}/deploy/robots/g1_dex3/build", f"{S}/e2e2_ctrl.log")
        for _ in range(60):
            time.sleep(0.5)
            if "Connected to robot" in open(f"{S}/e2e2_ctrl.log", errors="ignore").read(): break
        time.sleep(1.0)
    key("1"); time.sleep(3.0); key("2"); time.sleep(2.0)
    for _ in range(5): key("8"); time.sleep(0.4)   # lower the elastic band 0.5 m
    time.sleep(1.0); print(f"pelvis z before release {pelvis_z():.3f}"); key("9"); time.sleep(2.5); z = pelvis_z(); print(f"pelvis z in Velocity after release {z:.3f}")
    if z > 0.6: break
lib = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); lib.settimeout(1.0)
try:
    lib.sendto(f"clip {CLIP}".encode(), ("127.0.0.1", 9873)); print("motion library:", lib.recv(4096).decode().splitlines()[0])
except OSError:
    print("motion library: no command port (using the state's motion_file)")
key("4"); time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 16.0); print(f"pelvis z at end of student phase {pelvis_z():.3f}"); key("0"); time.sleep(1.5)
subprocess.run(["pkill", "-x", "g1_dex3_ctrl"]); subprocess.run(["pkill", "-x", "unitree_mujoco"]); time.sleep(0.5)
log = open(f"{S}/e2e2_ctrl.log", errors="ignore").read().replace("\r", "")
for line in log.splitlines():
    if re.search(r"Source |Hiphi|align:|Loaded hiphi|critical|error|Exception|terminate|stale|what\(\)|FSM: Change", line): print("CTRL:", line.strip()[:300])
new = sorted(set(glob.glob("/tmp/hiphi_probe_*.npz")) - before)
if not new: print("no probe dump"); sys.exit(1)
d = np.load(new[-1]); t = d["t"]; ref = d["ref_entity"]; cmd = d["q_cmd"]; q = d["q_meas"]; umt = d["umt_offset"]; res = d["residual"]
phase = d["phase"] if "phase" in d else np.full(t.shape, 2.0)
BODY = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,29,30,31,32,33,34,35]; HAND = [22,23,24,25,26,27,28,36,37,38,39,40,41,42]
ref_b, ref_h = ref[:, BODY], ref[:, HAND]
eb = np.abs(q[:, :29] - ref_b); eh = np.abs(q[:, 29:] - ref_h)
jump = np.abs(np.diff(cmd, axis=0)).max(axis=1)  # max per-step target change over all 43 targets
names = {1: "EaseIn", 2: "Clip", 3: "EaseOut"}
print(f"probe {new[-1]}: {t.size} steps; phases: " + ", ".join(f"{names.get(int(p), p)} {int((phase==p).sum())} steps" for p in np.unique(phase)))
print(f"  max per-step target jump overall {jump.max():.3f} rad/step (step {jump.argmax()}, phase {names.get(int(phase[jump.argmax()+1]), '?')}); "
      + ", ".join(f"{names.get(int(p), p)} {jump[phase[1:]==p].max():.3f}" for p in np.unique(phase) if (phase[1:]==p).any()))
for p in np.unique(phase):
    m = phase == p
    print(f"  {names.get(int(p), p):7s}: body |q-q_ref| mean {eb[m].mean():.4f} max {eb[m].max():.3f} rad; hand mean {eh[m].mean():.3f}; |umt| mean {np.abs(umt[m]).mean():.3f} max {np.abs(umt[m]).max():.3f}; |res| body mean {np.abs(res[m][:, :29]).mean():.3f}")
hold = phase == 3
if hold.any():
    last = np.where(hold)[0][-50:]
    print(f"  hold (last 1 s): body |q-q_ref| mean {eb[last].mean():.4f}, hand cmd std {cmd[last][:, 29:].std(axis=0).max():.4f} (frozen -> 0)")
print(f"  hand q_meas std over time {q[:, 29:].std(axis=0).mean():.4f} (0 = hand state not arriving); NaN in cmd: {np.isnan(cmd).any()}")
