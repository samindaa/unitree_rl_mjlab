# hiphi depth student v1

`smp_v2/ckpts/pnp64_res_rand_60x80_student_9999.onnx` — export of
`289575085-g1_hiphi_umt_multi_v2_distill_rand_model_9999.pt` (run
`2026-09-14_20-06-24_pnp64_res_rand_60x80_full`, SE UMT base 283112747;
distilled with `--randomize --depth_hw=60,80 --depth_cutoff_m=2.0
--object_mask_p=0.1`, depth noise 5 mm / dropout 0.05 / shift 2 px, estimator
bias 0.01 m / 0.05 m/s, noise 0.002 m / 0.02 m/s) with
`smp_v2/scripts/export_student_onnx.py --height 60 --width 80` (2026-09-16).
Inputs `student_state [1,196]`, `student_camera [1,1,60,80]`; output
`actions [1,43]`. Depth contract: 60x80, cutoff 2.0 m (params/deploy.yaml;
`sources.depth` and `simulate/config.yaml depth_camera` must render 80x60).
