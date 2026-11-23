#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
rt=/home/ma-user/work/CADRetrieval_runtime
run=$1
case "$run" in "$rt"/runs/idea01_in_domain_pilot_*) ;; *) exit 2 ;; esac
if [[ -n "$(GIT_OPTIONAL_LOCKS=0 git -C "$repo" status --porcelain)" ]] || git -C "$repo" symbolic-ref -q HEAD; then
    printf 'A clean detached deployment is required\n' >&2
    exit 2
fi
stage="$run/hiercad"
mkdir "$stage"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$stage/controller.exit"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/end_utc.txt"
}
trap finish EXIT
phase() {
    local tag=$1 rc=0
    shift
    printf '%q ' "$@" > "$stage/$tag.argv"
    printf '\n' >> "$stage/$tag.argv"
    "$@" > "$stage/$tag.stdout" 2> "$stage/$tag.stderr" || rc=$?
    printf '%s\n' "$rc" > "$stage/$tag.exit"
    return "$rc"
}
git -C "$repo" rev-parse HEAD > "$stage/code_sha.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/start_utc.txt"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
asset="$rt/pilot_assets_20261123"
export PYTHONPATH="$repo:$repo/experiments/src:$asset/model_overlay:$asset/geometry_overlay:$asset/geometry_extracted_v2/lib/python3.11/site-packages"
py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export LD_LIBRARY_PATH="$asset/geometry_extracted_v2/lib:$asset/geometry_freeimage/lib:$asset/geometry_cpp_runtime_v3/lib:${LD_LIBRARY_PATH:-}"
if [[ -e "$asset/geometry_freeimage/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$asset/geometry_freeimage/lib/libstdc++.so.6"
elif [[ -e "$asset/geometry_cpp_runtime_v3/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$asset/geometry_cpp_runtime_v3/lib/libstdc++.so.6"
fi
cd -- "$repo"
common=(
    "$py" -B -u experiments/scripts/run_in_domain_pilot_native.py hiercad
    --manifests "$run/manifests"
    --cad-zip "$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
    --minimal-zip "$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/text2cad_v1.0/misc/minimal_json/minimal_json_0000_0099.zip"
    --hiercad-repo "$asset/vendor/hiercad"
    --hiercad-base "$asset/qwen2.5-coder-14b-instruct"
    --hiercad-adapter "$asset/hiercad-14b"
    --hiercad-validation "$asset/hiercad-validation/validation.parquet"
)
phase focused "$py" -B -m pytest -p no:cacheprovider -q experiments/tests/test_in_domain_pilot_native.py
phase prepare "${common[@]}" --stage prepare --out "$stage/prepared_targets" || true

# Wait on the existing download's actual receipt. No accelerator is allocated in this loop.
while [[ ! -f "$asset/models_mirror_exit.txt" ]]; do
    sleep 60
done
test "$(cat "$asset/models_mirror_exit.txt")" = 0
for asset_receipt in qwen2.5-coder-14b-instruct_verified_3 hiercad-14b_verified_1 hiercad-validation_verified_2; do
    test -s "$asset/$asset_receipt.json"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/assets_ready_utc.txt"
for attempt in $(seq 1 61); do
    npu-smi info > "$stage/npu_before_launch.txt"
    device=
    if [[ "$(grep -c 'Ascend910' "$stage/npu_before_launch.txt")" = 4 ]]; then
        if grep -q 'No running processes found in NPU 2' "$stage/npu_before_launch.txt"; then
            device=0
        elif grep -q 'No running processes found in NPU 3' "$stage/npu_before_launch.txt"; then
            device=2
        fi
    fi
    if [[ -n "$device" ]]; then break; fi
    if [[ "$attempt" = 61 ]]; then exit 91; fi
    date -u +'%Y-%m-%dT%H:%M:%SZ' >> "$stage/device_wait_utc.txt"
    sleep 60
done
printf '%s\n' "$device" > "$stage/visible_device.txt"
export ASCEND_RT_VISIBLE_DEVICES="$device"
phase smoke "${common[@]}" --stage smoke --out "$stage/smoke" --device npu:0
phase pilot "${common[@]}" --stage all --out "$stage/pilot" --device npu:0 \
    --batch-size 1 --timing-candidates 8 --timing-queries 4 --timing-warmups 1 --timing-repeats 2 --training-steps 1
