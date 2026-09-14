# hiphi UMT base v0

`smp_v2/ckpts/283112747-hiphi_umt_model_49999.onnx` — export of
`283112747-hiphi_umt_model_49999.pt` (task `Umt-Tracking-G1`, WITH state
estimation: 160 obs incl. motion_anchor_pos_b + base_lin_vel), Mish,
2048-2048-1024-1024-512-512, soft-clip 5.0. The frozen base the pnp64_res_v0
student was distilled on (`umt_checkpoint` of Hiphi-Tracking-Multi-V2-Distill-G1).
Exported 2026-09-14 with `scripts/export_policy_onnx.py --activation mish`.
