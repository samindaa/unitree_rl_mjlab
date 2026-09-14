# hiphi depth student v0

`smp_v2/ckpts/pnp64_res_v0_student_9999.onnx` — export of
`logs/rsl_rl/g1_hiphi_umt_multi_v2_distill/2026-09-13_01-24-51_pnp64_res_v0/model_9999.pt`
(`student_state_dict`, SpatialSoftmaxCNNModel: conv 16/32 + spatial softmax
-> 64 latent, MLP 1024-1024-512-256 ELU, soft-clip 5.0) with
`smp_v2/scripts/export_student_onnx.py` (2026-09-14; torch vs ORT max diff
1e-5). Inputs `student_state [1,196]`, `student_camera [1,1,36,64]`; output
`actions [1,43]`. Runs on top of `../../hiphi_umt_base/v0` (State_HiphiStudent).
