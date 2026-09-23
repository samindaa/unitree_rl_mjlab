# UMT policy vdcm

Export of smp_v2 task `Umt-Tracking-G1-No-State-Estimation` (actor only),
"dcm" training variant.

Current file: `smp_v2/ckpts/291827423-hiphi_umt_nose_dcm_model_18500.onnx`
(run 291827423 `hiphi_umt_nose_dcm`, iter 18500, actor
2048-2048-1024-1024-512-512, **Mish**, soft-clip 5.0, obs normalizer baked
in). Placed 2026-09-22; graph verified (Mish ops, static
`obs [1,154] -> actions [1,29]`) — same observation/action contract as
umt/v0..v2, so `params/` (deploy.yaml + clip) is a copy of v2's.

Version history: v0 = run 286665166 iter 18500, v1 = run 287264885
iter 46000, v2 = run 287265686 iter 52000, vdcm = run 291827423 iter 18500.
Export requirements are in `../../v0/exported/README.md`.
