# UMT policy v2

Export of smp_v2 task `Umt-Tracking-G1-No-State-Estimation` (actor only).

Current file: `smp_v2/ckpts/287265686-hiphi_umt_no_se_model_52000.onnx`
(run 287265686, iter 52000, actor 2048-2048-1024-1024-512-512, **Mish**,
soft-clip 5.0, obs normalizer baked in). Placed 2026-09-08; graph verified
(Mish ops, static `obs [1,154] -> actions [1,29]`), not yet run in
unitree_mujoco.

`params/` (deploy.yaml + clip) is copied from v1 — same obs/action contract.
Export requirements (wrong activation = robot falls, static batch dim,
`--activation mish`) are documented in `../../v0/exported/README.md` and
`DEPLOY_POLICY_FINDINGS.md` at the repo root.

To run v2: in `config/config.yaml` set `policy_dir: config/policy/umt/v2/`
(or `config/policy/umt` — `param.h` auto-picks the highest-sorted version
dir containing `exported/`). `motion_file` stays as-is; the clip is the same.

Version history: v0 = run 286665166 iter 18500, v1 = run 287264885
iter 46000, v2 = run 287265686 iter 52000.
