"""Browser-based viser mirror for the C++ `simulate/` (unitree_mujoco) simulator.

The simulator's GLFW window renders with software OpenGL on machines without a
GPU-backed display (e.g. over Chrome Remote Desktop), which is slow and drags
the physics thread through the shared mutex. This script is a *passive* viewer:
the simulator streams (time, qpos) datagrams to 127.0.0.1:<state_tap_port>
(see `state_tap_port` in simulate/config.yaml and simulate/src/state_tap.h),
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
import tyro
import viser
import yaml
from mjviser import ViserMujocoScene

REPO_ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x4D4A5150  # "MJQP", see simulate/src/state_tap.h
HEADER = struct.Struct("<IId")  # magic, nq, time
COMMAND_MAGIC = struct.pack("<I", 0x4D4A4B59)  # "MJKY", command tap


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
  print(f"Listening for simulator state on udp://127.0.0.1:{tap_port}")

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.create_scene_gui()
  with server.gui.add_folder("Simulator"):
    status_md = server.gui.add_markdown("waiting for state tap ...")
  build_command_gui(server, commands, sim_cfg)

  # Show the default pose until the first packet arrives.
  mujoco.mj_forward(model, data)
  scene.update_from_mjdata(data)
  print(f"viser mirror running: http://localhost:{cfg.port}")

  period = 1.0 / cfg.fps
  last_count = 0
  last_stats_time = time.monotonic()
  while True:
    tic = time.monotonic()
    with receiver.lock:
      qpos = receiver.qpos
      sim_time = receiver.sim_time
      count = receiver.count

    if qpos is not None:
      data.qpos[:] = qpos
      mujoco.mj_forward(model, data)
      scene.update_from_mjdata(data)

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
      last_count = count
      last_stats_time = now

    time.sleep(max(0.0, period - (time.monotonic() - tic)))


if __name__ == "__main__":
  main(tyro.cli(MirrorConfig))
