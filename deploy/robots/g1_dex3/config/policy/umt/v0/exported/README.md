# Drop `policy.onnx` here

Export of smp_v2 task `Umt-Tracking-G1-No-State-Estimation` (actor only).

Current file: `smp_v2/ckpts/286665166-hiphi_umt_no_se_model_18500.onnx`
(run 286665166, iter 18500, actor 2048-2048-1024-1024-512-512, **Mish**,
soft-clip 5.0, obs normalizer baked in). Placed 2026-09-04; graph verified
(Mish ops, static shapes), not yet run in unitree_mujoco.

Previous:
- `smp_v2/ckpts/286448593-hiphi_umt_no_se_model_16500.onnx` (run 286448593,
  iter 16500) — placed 2026-09-04.
- `smp_v2/ckpts/283861004-hiphi_umt_no_se_model_12500.onnx` (run 283861004,
  iter 12500) — verified in unitree_mujoco 2026-08-28.

Requirements (see `DEPLOY_POLICY_FINDINGS.md` at the repo root):

- Export with the activation the run was trained with. `export_policy_onnx.py`
  defaults to `--activation relu`; this run needs `--activation mish`. A wrong
  activation loads and exports without error but the robot falls.

- Input `obs` with a **static** shape `[1, 154]`, output `actions` `[1, 29]`.
  A dynamic batch dim (`['batch', 154]`) crashes onnxruntime in the deploy
  code; fix it with `scripts/umt_bundle_to_deploy_npz.py --onnx policy.onnx`.
- If the export produced `policy.onnx.data`, keep it next to `policy.onnx`.

The matching clip is `../params/umt_deploy_1282_27_pre1s_post1s.npz` (path set
by `motion_file` in `config/config.yaml`). To extract a different clip from a
bundle:

```
uv run python scripts/umt_bundle_to_deploy_npz.py \
    /path/to/umt_v1.zip <clip_name> \
    deploy/robots/g1_dex3/config/policy/umt/v0/params/motion.npz
```
