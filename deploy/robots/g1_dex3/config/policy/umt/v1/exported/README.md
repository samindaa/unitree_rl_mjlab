# UMT policy v1

Export of smp_v2 task `Umt-Tracking-G1-No-State-Estimation` (actor only).

Current file: `smp_v2/ckpts/287264885-hiphi_umt_no_se_model_46000.onnx`
(run 287264885, iter 46000, actor 2048-2048-1024-1024-512-512, **Mish**,
soft-clip 5.0, obs normalizer baked in). Placed 2026-09-08; graph verified
(Mish ops, static `obs [1,154] -> actions [1,29]`), not yet run in
unitree_mujoco.

Previous v1 content: `286448593-hiphi_umt_no_se_model_16500.onnx`
(run 286448593, iter 16500; action_rate -0.5 + joint_acc_l2 -2.5e-7 recipe,
exported 2026-09-04).

`../params/` (deploy.yaml + clip) is identical to v0 — same obs/action
contract. Export requirements (wrong activation = robot falls, static batch
dim, `--activation mish`) are documented in `../../v0/exported/README.md`
and `DEPLOY_POLICY_FINDINGS.md` at the repo root.

To run v1: in `config/config.yaml` set `policy_dir: config/policy/umt/v1/`
(or `config/policy/umt` — `param.h` auto-picks the highest-sorted version
dir containing `exported/`). `motion_file` stays as-is; the clip is the same.
