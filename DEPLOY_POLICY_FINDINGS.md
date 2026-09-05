# Deploy findings: gdm_0 mimic policy crashes (2026-08-04)

Debugging notes from deploying `deploy/robots/g1/config/policy/mimic/gdm_0/` on the G1
deploy stack. Two crashes were root-caused; both were format mismatches between what the
policy exporter produced and what the C++ deploy code can consume. Workaround patches were
applied in place to the `gdm_0` files (originals kept as `.bak`); the proper fix is in the
exporter so future policies work untouched.

## Issue 1: motion npz crashes cnpy — `load_the_npy_file: failed fread`

**Symptom**
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  load_the_npy_file: failed fread
```
Thrown while loading `params/740ae4e9.npz` via `cnpy::npz_load` in
`deploy/robots/g1/include/State_Mimic.h` (`MotionLoader_::load_data_from_npz`).

**Root cause (a) — deflate + zip64 local headers.** The npz was written with
`np.savez_compressed` by a modern numpy. Each zip entry is deflate-compressed and the zip
*local* header stores placeholder sizes `0xFFFFFFFF` (real sizes live in the zip64 extra
field). The vendored cnpy (`deploy/thirdparty/cnpy/cnpy.cpp`, `npz_load`, ~line 268) reads
the compressed size from a fixed offset in the local header, gets 4294967295, and the
subsequent `fread` fails. Stored (uncompressed) entries from plain `np.savez` are immune:
for `method=0` cnpy ignores the zip sizes and parses the inner `.npy` header instead —
that is why the shipped `dance1_subject2.npz` works despite having the same zip64
placeholders.

**Root cause (b) — unicode string arrays desync cnpy (latent, would crash next).** The
npz also contained string/metadata arrays: `joint_names` (`<U26`), `body_names`,
`object_name`, `object_scale`, `object_mesh_file`, `source`, `object_*_w`. cnpy parses
`<U26` as 26 bytes per element, but numpy `<U` is UCS-4 = 4 bytes/char, so the real element
size is 104 bytes. cnpy under-reads the entry, the stream desyncs, and it silently stops
loading — in testing, everything after `joint_names.npy` was dropped, including
`body_pos_w`/`body_quat_w` that `State_Mimic.h` requires. The working `dance1_subject2.npz`
contains only the 7 numeric arrays.

**Workaround applied.** Repacked `params/740ae4e9.npz` in place with `np.savez`
(uncompressed), keeping only the 7 numeric keys matching the working policy:
`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`.
Verified all 7 load with correct shapes/values through a test program compiled against the
vendored cnpy. Original preserved as `params/740ae4e9.npz.compressed.bak`.

**Proper fixes (pick one or both):**
- Exporter: write the deploy npz with `np.savez` (NOT `savez_compressed`) and include only
  the numeric arrays listed above.
- C++ (more robust): fix/replace the vendored cnpy — read sizes from the zip central
  directory or zip64 extra field, support deflate+zip64, and either skip or correctly size
  `<U` (UCS-4) entries instead of silently truncating the archive.

## Issue 2: ONNX policy crashes ORT — `tried creating tensor with negative value in shape`

**Symptom**
```
terminate called after throwing an instance of 'Ort::Exception'
  what():  tried creating tensor with negative value in shape
```

**Root cause.** `exported/policy.onnx` was exported with a dynamic batch dimension:
input `obs ['batch', 160]`, output `actions ['batch', 29]`. The deploy code
(`deploy/include/isaaclab/algorithms/algorithms.h`, ~lines 42–81) passes
`GetTensorTypeAndShapeInfo().GetShape()` verbatim to `Ort::Value::CreateTensor`; a dynamic
dim is reported as -1, which CreateTensor rejects. The working `dance1_subject2` policy has
a static `[1, 154]` input, so it never hit this.

**Workaround applied.** Patched `gdm_0/exported/policy.onnx` in place: set the first dim of
graph input and output to a static 1 (`obs [1, 160]`, `actions [1, 29]`). Passes
`onnx.checker`; a forward pass with a zero `(1, 160)` obs produces `(1, 29)` actions.
Original preserved as `exported/policy.onnx.dynbatch.bak`.

Patch snippet (for the exporter or a post-export step):
```python
import onnx
m = onnx.load(path)
for v in list(m.graph.input) + list(m.graph.output):
    d = v.type.tensor_type.shape.dim[0]
    if not d.HasField('dim_value'):
        d.ClearField('dim_param'); d.dim_value = 1
onnx.checker.check_model(m)
onnx.save(m, path)
```

**Proper fix.** Export ONNX with static shapes (no `dynamic_axes`, batch dim = 1) for the
deploy stack. Alternatively harden `algorithms.h` to clamp any dim < 0 to 1 before
`CreateTensor` — that makes deploy tolerant of any export.

## Not yet verified — likely next failure point

- **Obs dimension:** gdm_0 expects 160 obs; dance1_subject2 uses 154. The obs terms in
  `gdm_0/params/deploy.yaml` must produce exactly 160 floats or ORT will throw a shape
  mismatch at inference time. Check this before the next on-robot run.

## Checklist for future policy exports targeting this deploy stack

1. ONNX: static shapes only — input `[1, num_obs]`, output `[1, num_actions]`.
2. Motion npz: `np.savez` (uncompressed), numeric arrays only
   (`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`).
3. No string/object dtypes anywhere in the npz.
4. Confirm `deploy.yaml` obs terms sum to the ONNX input width.

## Issue 3: UMT policy falls in unitree_mujoco — ONNX exported with the wrong activation (2026-08-28)

**Symptom.** `g1_dex3` UMT controller: every observation/action term checked out against the
training code, ONNX had static shapes and the soft-clip tail, yet the robot fell within a
second in `unitree_mujoco` while the same clip played fine in mjlab.

**Root cause.** `smp_v2/scripts/export_policy_onnx.py` rebuilds the actor from the
`actor_state_dict` weight shapes and takes the activation from a flag (default `relu`).
Run `283861004-hiphi_umt_no_se` was trained with **Mish** (and a 2048-2048-1024-1024-512-512
actor, not the 1024-wide one in `zest_tracking/rl_cfg.py`). Activations carry no weights, so
the strict `load_state_dict` succeeds and the script's own torch-vs-ONNX round-trip check
passes — both sides are equally wrong.

**Fix.** Re-export with `--activation mish`. Quick check on any exported policy:
`Counter(n.op_type for n in onnx.load(p).graph.node)` — a Mish actor shows
`Softplus`/`Tanh`/`Mul` per layer, a ReLU one shows `Relu`.

**Lesson.** The export script cannot detect activation or `mean_clip_scale` from the
checkpoint; record them with the run (or read them from the run's saved agent cfg) rather
than trusting the defaults.
