"""Browser-based viser mirror for the C++ `simulate/` (unitree_mujoco) simulator.

The simulator's GLFW window renders with software OpenGL on machines without a
GPU-backed display (e.g. over Chrome Remote Desktop), which is slow and drags
the physics thread through the shared mutex. This script is a *passive* viewer:
the simulator streams (time, qpos) datagrams to 127.0.0.1:<state_tap_port>
(see `state_tap_port` in simulate/config.yaml and simulate/src/state_tap.h),
and the depth camera streamer sends its published frames to port + 2, shown
in a "Depth camera (policy input)" panel exactly as the controller sees them,
and this process renders them with viser — all pixels are drawn by the
browser's WebGL, so nothing OpenGL runs on this machine.

Why not subscribe to the DDS topics directly: the simulator links cyclonedds
0.10.2, which segfaults on the XTypes discovery data emitted by the newer
cyclonedds releases that Python 3.12 wheels exist for. The UDP tap keeps the
visualizer entirely off the DDS domain.

Usage — three terminals:
  # 1. simulator, headless-ish (xvfb-run) or with its normal window
  #    (-u WAYLAND_DISPLAY: else GLFW 3.4 opens on the real Wayland screen)
  env -u WAYLAND_DISPLAY xvfb-run -a ./simulate/build/unitree_mujoco

  # 2. this mirror (open the printed URL in your local browser)
  uv run scripts/sim_viser_mirror.py

  # 3. controller
  cd deploy/robots/g1/build && ./g1_ctrl --network=lo

The robot/scene defaults to whatever `simulate/config.yaml` selects, so the
mirror always loads the same MJCF as the simulator.
"""

import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread

import mujoco
import numpy as np
import tyro
import viser
import yaml
from mjviser import ViserMujocoScene

REPO_ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x4D4A5150  # "MJQP", see simulate/src/state_tap.h
HEADER = struct.Struct("<IId")  # magic, nq, time
COMMAND_MAGIC = struct.pack("<I", 0x4D4A4B59)  # "MJKY", command tap
DEPTH_MAGIC = 0x4D4A4450  # "MJDP", depth tap (state tap port + 2), see simulate/src/depth_camera.h
DEPTH_HEADER = struct.Struct("<IIId")  # magic, width, height, time
DEPTH_CUTOFF_M = 3.0  # training normalisation (camera_depth cutoff_distance)


@dataclass(frozen=True)
class MirrorConfig:
  scene: Path | None = None
  """Robot scene MJCF. Defaults to `robot_scene` from simulate/config.yaml."""
  tap_port: int | None = None
  """State-tap UDP port. Defaults to `state_tap_port` from simulate/config.yaml."""
  port: int = 8080
  """Port for the viser web server."""
  fps: float = 30.0
  """Render update rate pushed to the browser."""
  controller_host: str = "127.0.0.1"
  """Host running g1_dex3_ctrl (its motion-library command port)."""
  controller_port: int = 9873
  """`motions.command_port` in the controller's config.yaml; 0 disables the clip panel."""


class TapReceiver:
  """Drains state-tap datagrams on a background thread, keeping the latest."""

  def __init__(self, port: int, nq: int):
    self.lock = Lock()
    self.qpos = None
    self.sim_time = 0.0
    self.count = 0
    self._nq = nq
    self._payload = struct.Struct(f"<{nq}d")
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.bind(("127.0.0.1", port))
    self._sock.settimeout(0.5)
    Thread(target=self._run, daemon=True).start()

  def _run(self) -> None:
    expected = HEADER.size + self._payload.size
    while True:
      try:
        packet = self._sock.recv(65536)
      except TimeoutError:
        continue
      if len(packet) != expected:
        continue
      magic, nq, sim_time = HEADER.unpack_from(packet)
      if magic != MAGIC or nq != self._nq:
        continue
      qpos = self._payload.unpack_from(packet, HEADER.size)
      with self.lock:
        self.qpos = qpos
        self.sim_time = sim_time
        self.count += 1


class DepthTapReceiver:
  """Keeps the latest depth frame (uint16 mm, row 0 = top) from the depth tap."""

  def __init__(self, port: int):
    self.lock = Lock()
    self.frame = None
    self.sim_time = 0.0
    self.count = 0
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.bind(("127.0.0.1", port))
    self._sock.settimeout(0.5)
    Thread(target=self._run, daemon=True).start()

  def _run(self) -> None:
    while True:
      try:
        packet = self._sock.recv(65536)
      except TimeoutError:
        continue
      if len(packet) < DEPTH_HEADER.size:
        continue
      magic, width, height, sim_time = DEPTH_HEADER.unpack_from(packet)
      if magic != DEPTH_MAGIC or len(packet) != DEPTH_HEADER.size + 2 * width * height:
        continue
      frame = np.frombuffer(packet, dtype="<u2", count=width * height, offset=DEPTH_HEADER.size)
      with self.lock:
        self.frame = frame.reshape(height, width).copy()
        self.sim_time = sim_time
        self.count += 1


def depth_to_rgb(frame_mm: np.ndarray, upscale: int = 6) -> np.ndarray:
  """Policy-input view: clamp(d, 0.01, cutoff)/cutoff as grey (near = bright),
  no-return pixels (0 mm) in red; nearest-neighbour upscaled for the panel."""
  d = np.clip(frame_mm.astype(np.float32) * 1e-3, 0.01, DEPTH_CUTOFF_M) / DEPTH_CUTOFF_M
  grey = ((1.0 - d) * 255).astype(np.uint8)
  rgb = np.stack([grey, grey, grey], axis=-1)
  rgb[frame_mm == 0] = (200, 30, 30)
  return np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)


class MotionClipClient:
  """Talks to the controller's MotionLibrary command port (MotionLibrary.h):
  sends `clip <stem>` / `next` / `prev` / `list`; every command is answered
  with "current <index> <stem>\n<stem>...". Polled on a background thread."""

  def __init__(self, host: str, port: int):
    self.lock = Lock()
    self.names: list[str] = []
    self.current = -1
    self.alive = False
    self._addr = (host, port)
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.settimeout(0.5)
    Thread(target=self._poll, daemon=True).start()

  def send(self, cmd: str) -> None:
    try:
      self._sock.sendto(cmd.encode(), self._addr)
      self._parse(self._sock.recv(65536))
    except OSError:
      with self.lock:
        self.alive = False

  def _parse(self, packet: bytes) -> None:
    lines = packet.decode(errors="replace").splitlines()
    if not lines or not lines[0].startswith("current "):
      return
    head = lines[0].split(" ", 2)
    with self.lock:
      self.current = int(head[1])
      self.names = [n for n in lines[1:] if n]
      self.alive = True

  def _poll(self) -> None:
    while True:
      self.send("list")
      time.sleep(2.0)


class CommandSender:
  """Sends single-key commands to the simulator's command tap (port + 1).

  Keys mean exactly what they mean in the simulator: keyboard-joystick keys as
  documented in simulate/config.yaml, '9'/'7'/'8' for the elastic band, and
  backspace for simulation reset.
  """

  def __init__(self, port: int):
    self._addr = ("127.0.0.1", port)
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  def send(self, key: str) -> None:
    self._sock.sendto(COMMAND_MAGIC + key.encode("latin-1"), self._addr)


def load_sim_config(cfg: MirrorConfig) -> tuple[Path, int, dict]:
  sim_cfg = yaml.safe_load((REPO_ROOT / "simulate" / "config.yaml").read_text())
  scene = cfg.scene if cfg.scene is not None else Path(sim_cfg["robot_scene"])
  if not scene.is_absolute():
    scene = REPO_ROOT / scene
  tap_port = cfg.tap_port
  if tap_port is None:
    tap_port = int(sim_cfg.get("state_tap_port", 0))
  if tap_port <= 0:
    raise SystemExit(
      "state tap is disabled: set `state_tap_port` in simulate/config.yaml "
      "or pass --tap-port"
    )
  return scene, tap_port, sim_cfg


def build_command_gui(
  server: viser.ViserServer, commands: CommandSender, sim_cfg: dict
) -> None:
  """Buttons mirroring every terminal-keyboard and GLFW-window command."""
  keyboard_map = sim_cfg.get("keyboard_map") or {}
  if int(sim_cfg.get("use_joystick", 0)) == 1 and sim_cfg.get("joystick_type") == "keyboard":
    with server.gui.add_folder("FSM chords"):
      for key, chord in keyboard_map.items():
        button = server.gui.add_button(f"[{key}]  {chord}")
        button.on_click(lambda _, k=str(key): commands.send(k))

    with server.gui.add_folder("Velocity commands"):
      move = server.gui.add_button_group("move", ("◀ a", "▲ w", "▼ s", "▶ d"))
      move.on_click(lambda _: commands.send(move.value[-1]))
      turn = server.gui.add_button_group("turn", ("↺ q", "stop", "↻ e"))
      turn.on_click(
        lambda _: commands.send(" " if turn.value == "stop" else turn.value[-1])
      )

  if int(sim_cfg.get("enable_elastic_band", 0)) == 1:
    with server.gui.add_folder("Elastic band"):
      for label, key in (
        ("Toggle band  [9]", "9"),
        ("Raise  [7]", "7"),
        ("Lower  [8]", "8"),
      ):
        button = server.gui.add_button(label)
        button.on_click(lambda _, k=key: commands.send(k))

  reset_button = server.gui.add_button("Reset simulation", color="red")
  reset_button.on_click(lambda _: commands.send("\b"))


def main(cfg: MirrorConfig) -> None:
  scene_path, tap_port, sim_cfg = load_sim_config(cfg)
  print(f"Loading scene: {scene_path}")
  model = mujoco.MjModel.from_xml_path(str(scene_path))
  data = mujoco.MjData(model)

  receiver = TapReceiver(tap_port, model.nq)
  commands = CommandSender(tap_port + 1)
  depth = DepthTapReceiver(tap_port + 2)
  print(f"Listening for simulator state on udp://127.0.0.1:{tap_port}, depth on :{tap_port + 2}")

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.create_scene_gui()
  with server.gui.add_folder("Simulator"):
    status_md = server.gui.add_markdown("waiting for state tap ...")
  with server.gui.add_folder("Depth camera (policy input)"):
    depth_img = server.gui.add_image(np.zeros((36 * 6, 64 * 6, 3), dtype=np.uint8), label="rt/depth_camera")
    depth_md = server.gui.add_markdown("waiting for depth tap ...")
  build_command_gui(server, commands, sim_cfg)

  clips = MotionClipClient(cfg.controller_host, cfg.controller_port) if cfg.controller_port > 0 else None
  clip_dropdown = clip_md = None
  if clips is not None:
    with server.gui.add_folder("Motion clip (controller)"):
      clip_md = server.gui.add_markdown("waiting for g1_dex3_ctrl ...")
      clip_dropdown = server.gui.add_dropdown("clip", options=("-",), initial_value="-")
      clip_dropdown.on_update(lambda _: clips.send(f"clip {clip_dropdown.value}") if clip_dropdown.value != "-" else None)
      step = server.gui.add_button_group("step", ("◀ prev", "next ▶"))
      step.on_click(lambda _: clips.send("prev" if step.value.endswith("prev") else "next"))
  shown_names: list[str] = []
  shown_current = -1

  # Show the default pose until the first packet arrives.
  mujoco.mj_forward(model, data)
  scene.update_from_mjdata(data)
  print(f"viser mirror running: http://localhost:{cfg.port}")

  period = 1.0 / cfg.fps
  last_count = 0
  last_depth_count = 0
  shown_depth_count = 0
  last_stats_time = time.monotonic()
  while True:
    tic = time.monotonic()
    with receiver.lock:
      qpos = receiver.qpos
      sim_time = receiver.sim_time
      count = receiver.count
    with depth.lock:
      frame = depth.frame
      depth_count = depth.count
      depth_time = depth.sim_time
    if frame is not None and depth_count != shown_depth_count:
      depth_img.image = depth_to_rgb(frame)
      shown_depth_count = depth_count

    if qpos is not None:
      data.qpos[:] = qpos
      mujoco.mj_forward(model, data)
      scene.update_from_mjdata(data)

    if clips is not None:
      with clips.lock:
        names, current, alive = list(clips.names), clips.current, clips.alive
      if alive and names and (names != shown_names or current != shown_current):
        if names != shown_names:
          clip_dropdown.options = tuple(names)
        if 0 <= current < len(names) and clip_dropdown.value != names[current]:
          clip_dropdown.value = names[current]
        clip_md.content = f"current: **[{current}] {names[current] if 0 <= current < len(names) else '?'}** — applies on the next entry into Umt / HiphiEaseIn"
        shown_names, shown_current = names, current
      elif not alive and shown_current != -2:
        clip_md.content = f"waiting for g1_dex3_ctrl on udp://{cfg.controller_host}:{cfg.controller_port} ..."
        shown_current = -2

    now = time.monotonic()
    if now - last_stats_time >= 1.0:
      tap_hz = (count - last_count) / (now - last_stats_time)
      if count == 0:
        status_md.content = "waiting for state tap ... (is unitree_mujoco running?)"
      elif tap_hz == 0:
        status_md.content = "**stale** — simulator stopped sending"
      else:
        status_md.content = (
          f"state tap: **{tap_hz:.0f} Hz** \nsim time: **{sim_time:.1f} s**"
        )
      depth_hz = (depth_count - last_depth_count) / (now - last_stats_time)
      if depth_count == 0:
        depth_md.content = "waiting for depth tap ... (depth_camera.enable in simulate/config.yaml?)"
      else:
        valid = frame[frame > 0]
        rng = f"{valid.min() / 1000:.2f}..{valid.max() / 1000:.2f} m" if valid.size else "no returns"
        depth_md.content = (
          f"**{depth_hz:.0f} Hz**, t {depth_time:.1f} s, {frame.shape[1]}x{frame.shape[0]}, "
          f"{100 * valid.size / frame.size:.0f}% valid, {rng}  \n"
          f"grey = clamp(d, 0.01, {DEPTH_CUTOFF_M:.0f} m) / {DEPTH_CUTOFF_M:.0f} m (near = bright), red = no return"
        )
      last_depth_count = depth_count
      last_count = count
      last_stats_time = now

    time.sleep(max(0.0, period - (time.monotonic() - tic)))


if __name__ == "__main__":
  main(tyro.cli(MirrorConfig))
