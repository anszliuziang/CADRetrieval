#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
rt=/home/ma-user/work/CADRetrieval_runtime
run=$1
deadline_utc=$2
deadline=$(date -u -d "$deadline_utc" +%s)
case "$run" in "$rt"/runs/idea01_in_domain_pilot_*) ;; *) exit 2 ;; esac
if [[ -n "$(GIT_OPTIONAL_LOCKS=0 git -C "$repo" status --porcelain)" ]] || git -C "$repo" symbolic-ref -q HEAD; then
    printf 'A clean detached deployment is required\n' >&2
    exit 2
fi
stage="$run/hiercad_with_occ"
mkdir "$stage"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$stage/controller.exit"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/end_utc.txt"
}
trap finish EXIT
phase() {
    local tag=$1 rc=0 remaining
    shift
    printf '%q ' "$@" > "$stage/$tag.argv"
    printf '\n' >> "$stage/$tag.argv"
    remaining=$((deadline - $(date -u +%s) - 30))
    if ((remaining > 0)); then
        timeout --kill-after=30s "${remaining}s" "$@" > "$stage/$tag.stdout" 2> "$stage/$tag.stderr" || rc=$?
    else
        rc=124
        printf 'Night execution window ended\n' > "$stage/$tag.stderr"
    fi
    printf '%s\n' "$rc" > "$stage/$tag.exit"
    return "$rc"
}
git -C "$repo" rev-parse HEAD > "$stage/code_sha.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/start_utc.txt"
printf '%s\n' "$deadline_utc" > "$stage/deadline_utc.txt"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
asset="$rt/pilot_assets_20261123"
export PYTHONPATH="$repo:$repo/experiments/src:$asset/model_overlay:$asset/geometry_overlay:$asset/geometry_extracted_v2/lib/python3.11/site-packages"
py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export LD_LIBRARY_PATH="$asset/geometry_cpp_runtime_v3/lib:$asset/geometry_extracted_v2/lib:$asset/geometry_freeimage/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$asset/geometry_cpp_runtime_v3/lib/libgcc_s.so.1 $asset/geometry_cpp_runtime_v3/lib/libstdc++.so.6"
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
if [[ -s "$asset/hiercad-validation_verified_2.json" ]]; then
    phase prepare "${common[@]}" --stage prepare --out "$stage/prepared_targets" || true
fi

wait_until_next_check() {
    local remaining
    remaining=$((deadline - $(date -u +%s)))
    if ((remaining <= 0)); then return 124; fi
    sleep "$((remaining < 60 ? remaining : 60))"
}

# Wait on the existing download's actual receipt. No accelerator is allocated in this loop.
while [[ ! -f "$asset/models_mirror_exit.txt" ]]; do
    wait_until_next_check
done
test "$(cat "$asset/models_mirror_exit.txt")" = 0
for asset_receipt in qwen2.5-coder-14b-instruct_verified_3 hiercad-14b_verified_1 hiercad-validation_verified_2; do
    test -s "$asset/$asset_receipt.json"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/assets_ready_utc.txt"
while true; do
    npu-smi info > "$stage/npu_before_launch.txt"
    devices=()
    if [[ "$(grep -c 'Ascend910' "$stage/npu_before_launch.txt")" = 4 ]]; then
        if grep -q 'No running processes found in NPU 2' "$stage/npu_before_launch.txt"; then devices+=(0 1); fi
        if grep -q 'No running processes found in NPU 3' "$stage/npu_before_launch.txt"; then devices+=(2 3); fi
    fi
    if [[ "${#devices[@]}" -ge 2 ]]; then break; fi
    date -u +'%Y-%m-%dT%H:%M:%SZ' >> "$stage/device_wait_utc.txt"
    wait_until_next_check
done

lane() {
    local source=$1 device=$2
    printf '%s\n' "$device" > "$stage/$source.visible_device"
    export ASCEND_RT_VISIBLE_DEVICES="$device"
    phase "smoke_$source" "${common[@]}" --stage smoke --source "$source" \
        --out "$stage/smoke_$source" --device npu:0 || return "$?"
    phase "pilot_$source" "${common[@]}" --stage all --source "$source" \
        --out "$stage/pilot_$source" --device npu:0 \
        --batch-size 1 --timing-candidates 8 --timing-queries 4 --timing-warmups 1 --timing-repeats 2 --training-steps 1
}
lane t2cad-v1.0 "${devices[0]}" &
t_pid=$!
lane ct-deepcad "${devices[1]}" &
ct_pid=$!
status=0
wait "$t_pid" || status=1
wait "$ct_pid" || status=1
exit "$status"
