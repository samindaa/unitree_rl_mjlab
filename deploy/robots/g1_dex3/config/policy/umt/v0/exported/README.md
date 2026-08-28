# Drop `policy.onnx` here

Export of smp_v2 task `Umt-Tracking-G1-No-State-Estimation` (actor only).

Requirements (see `DEPLOY_POLICY_FINDINGS.md` at the repo root):

- Input `obs` with a **static** shape `[1, 154]`, output `actions` `[1, 29]`.
  A dynamic batch dim (`['batch', 154]`) crashes onnxruntime in the deploy
  code; fix it with `scripts/umt_bundle_to_deploy_npz.py --onnx policy.onnx`.
- If the export produced `policy.onnx.data`, keep it next to `policy.onnx`.

The matching clip goes to `../params/motion.npz` (path set by `motion_file`
in `config/config.yaml`):

```
uv run python scripts/umt_bundle_to_deploy_npz.py \
    /path/to/umt_v1.zip <clip_name> \
    deploy/robots/g1_dex3/config/policy/umt/v0/params/motion.npz
```
