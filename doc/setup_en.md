# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 22.04 
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 550 or later  

---

## 1. Installing uv

The project uses [uv](https://docs.astral.sh/uv/) to manage its Python version,
virtual environment and dependencies. If uv is already installed, skip this step.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell (or `source ~/.bashrc`) so that `uv` is on your `PATH`.

---

## 2. Installing

### 2.1 Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 System Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

### 2.3 Python Dependencies

All Python dependencies are specified in the `pyproject.toml` file and pinned in
`uv.lock`. Navigate to the project root directory and create the environment with:

```bash
cd unitree_rl_mjlab
uv sync --extra cu128
```

This creates a `.venv/` directory containing Python and every dependency,
including `mjlab==1.5.3` and a CUDA 12.8 build of PyTorch. On a machine without
an Nvidia GPU, use `uv sync --extra cpu` instead.

### 2.4 Running

Prefix commands with `uv run`, which uses the project environment without
requiring activation:

```bash
uv run python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Alternatively, activate the environment once and call `python` directly:

```bash
source .venv/bin/activate
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.

