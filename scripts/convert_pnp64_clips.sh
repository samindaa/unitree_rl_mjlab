#!/usr/bin/env bash
# Convert pnp64 v2 bundle scenes into deploy motion clips for the g1_dex3
# motion library (config/motions/pnp64, picked up by `motions.dir`).
#
# The v2 bundle (smp_v2/datasets/hiphi_bundles/pnp64_dex3_v2.npz) is a
# consolidated file; mjlab unpacks each scene into a per-clip npz under
# /tmp/smp_motion_bundles/pnp64_dex3_v2-<hash>/ the first time the task is
# built (train / play / rollout_hiphi_student.py). This script takes those
# per-clip files (43-joint entity, 46 bodies, plus object / cws / name arrays
# the deploy loader cannot read) and writes numeric-only deploy clips with the
# layout embedded, via scripts/umt_bundle_to_deploy_npz.py.
#
#   scripts/convert_pnp64_clips.sh                 # the default 16 (manifest scenes 0-15, single-object)
#   scripts/convert_pnp64_clips.sh 3358_40_1520622632 4239_10_1520616797   # explicit scene names
#   CACHE_DIR=... OUT_DIR=... scripts/convert_pnp64_clips.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-$(ls -d /tmp/smp_motion_bundles/pnp64_dex3_v2-*/ 2>/dev/null | head -1)}"
OUT_DIR="${OUT_DIR:-$REPO/deploy/robots/g1_dex3/config/motions/pnp64}"
DEFAULT_SCENES=(9204_13_1520609083 9204_6_1520608954 9257_17_1520622437 9258_14_1520622865 9261_5_1520624821
  9265_10_1520625674 9271_18_1520628719 9301_18_1520608388 9253_20_1520618315 9256_16_1520620254
  9206_2_1520617296 9297_19_1520604869 2907_10_1520622428 3000_10_1520624754 3010_20_1520620820 3020_40_1520620712)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${DEFAULT_SCENES[@]}")
if [ -z "$CACHE_DIR" ] || [ ! -d "$CACHE_DIR" ]; then
  echo "no unpacked pnp64_dex3_v2 cache under /tmp/smp_motion_bundles; run any mjlab play/rollout of the v2 task once, or set CACHE_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
for s in "${SCENES[@]}"; do
  src="$CACHE_DIR/$s.npz"
  [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
  uv run python "$REPO/scripts/umt_bundle_to_deploy_npz.py" "$src" "$OUT_DIR/$s.npz" 2>&1 | grep -E "^wrote|error" || true
done
echo "$(ls "$OUT_DIR"/*.npz | wc -l) clips in $OUT_DIR"
