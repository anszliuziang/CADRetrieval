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
test "$(cat "$run/manifests.exit")" = 0
test -f "$run/manifests/manifest_receipt.json"
stage=$run/models
mkdir -- "$stage"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
support=$rt/pilot_assets_20261123
export PYTHONPATH="$repo:$repo/experiments/src:$support/model_overlay"
base_py=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
retrieval_py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd -- "$repo"
git rev-parse HEAD > "$stage/code_sha.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/start_utc.txt"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$stage/controller.exit"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stage/end_utc.txt"
}
trap finish EXIT
phase() {
    local name=$1
    shift
    local rc=0
    printf '%q ' "$@" > "$stage/$name.argv"
    printf '\n' >> "$stage/$name.argv"
    "$@" > "$stage/$name.stdout" 2> "$stage/$name.stderr" || rc=$?
    printf '%s\n' "$rc" > "$stage/$name.exit"
    return "$rc"
}
bert="$rt/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"
cad="$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
t_weights="$rt/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth"
qwen="$rt/synchronized/20261123_af4bd77/in_domain_pilot_inputs/models/Qwen3-Embedding-0.6B/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
manifests="$run/manifests"
phase focused_tests "$retrieval_py" -B -m pytest -p no:cacheprovider -q \
    experiments/tests/test_in_domain_pilot.py \
    experiments/tests/test_in_domain_pilot_native.py \
    experiments/tests/test_in_domain_pilot_retriever_objectives.py
phase retrieval_imports "$retrieval_py" -B -c \
    'import torch,torch_npu,transformers,peft; from transformers import AutoModel, BertModel, Qwen3Model; print(torch.__version__,torch_npu.__version__,transformers.__version__,peft.__version__)'
phase text2cad_imports "$base_py" -B -c \
    'import torch,torch_npu,transformers; from transformers import BertModel; print(torch.__version__,torch_npu.__version__,transformers.__version__)'
for attempt in $(seq 1 31); do
    npu-smi info > "$stage/npu_before_launch.txt"
    devices=()
    if grep -q 'No running processes found in NPU 2' "$stage/npu_before_launch.txt"; then devices+=(0 1); fi
    if grep -q 'No running processes found in NPU 3' "$stage/npu_before_launch.txt"; then devices+=(2 3); fi
    if [[ "${#devices[@]}" -gt 0 && "$(grep -c 'Ascend910' "$stage/npu_before_launch.txt")" = 4 ]]; then
        break
    fi
    if [[ "$attempt" = 31 ]]; then
        printf 'Awaiting a fully idle Ascend device group\n' >&2
        exit 91
    fi
    date -u +'%Y-%m-%dT%H:%M:%SZ' >> "$stage/device_wait_utc.txt"
    sleep 60
done
printf '%s\n' "${devices[@]}" > "$stage/visible_devices.txt"

phase npu_smoke env "ASCEND_RT_VISIBLE_DEVICES=${devices[0]}" "$retrieval_py" -B -c \
    'import torch,torch_npu; torch.manual_seed(17); a=torch.arange(4,dtype=torch.float32).reshape(2,2); b=a.to("npu:0"); out=(b@b).cpu(); torch.testing.assert_close(out,a@a); print(out.tolist(),torch.npu.get_device_name(0))'
dual() {
    local label=$1 source=$2 device=$3
    phase "$label" env "ASCEND_RT_VISIBLE_DEVICES=$device" \
        "$retrieval_py" -B -u experiments/scripts/run_in_domain_pilot_retrievers.py \
        --manifest-root "$manifests" --source "$source" --method dual-tower \
        --official-zip "$cad" --output "$stage/$label" --device npu:0 \
        --precision bf16 --bert-dir "$bert" --workers 8
}
spectra() {
    local label=$1 source=$2 device=$3
    phase "$label" env "ASCEND_RT_VISIBLE_DEVICES=$device" \
        "$retrieval_py" -B -u experiments/scripts/run_in_domain_pilot_retrievers.py \
        --manifest-root "$manifests" --source "$source" --method spectrabrep \
        --official-zip "$cad" --output "$stage/$label" --device npu:0 --precision bf16 \
        --qwen-dir "$qwen" --upstream "$support/vendor/spectrabrep" --workers 8
}
lane_zero() {
    local device=$1
    local failed=0
    phase text2cad_native env "ASCEND_RT_VISIBLE_DEVICES=$device" \
        "$base_py" -B -u experiments/scripts/run_in_domain_pilot_native.py \
        text2cad --stage readout --manifests "$manifests" --out "$stage/text2cad_native" \
        --cad-zip "$cad" --checkpoint "$t_weights" --bert-dir "$bert" --device npu:0 --batch-size 4 || failed=1
    dual dual_ct ct-deepcad "$device" || failed=1
    return "$failed"
}
lane_single_group() {
    local device=$1 source=$2 dual_label=$3 spectra_label=$4
    local failed=0
    dual "$dual_label" "$source" "$device" || failed=1
    spectra "$spectra_label" "$source" "$device" || failed=1
    return "$failed"
}
if [[ "${#devices[@]}" -eq 4 ]]; then
    lane_zero "${devices[0]}" &
    pids=("$!")
    dual dual_t t2cad-v1.0 "${devices[1]}" &
    pids+=("$!")
    spectra spectra_t t2cad-v1.0 "${devices[2]}" &
    pids+=("$!")
    spectra spectra_ct ct-deepcad "${devices[3]}" &
    pids+=("$!")
else
    (
        failed=0
        lane_zero "${devices[0]}" || failed=1
        spectra spectra_ct ct-deepcad "${devices[0]}" || failed=1
        exit "$failed"
    ) &
    pids=("$!")
    lane_single_group "${devices[1]}" t2cad-v1.0 dual_t spectra_t &
    pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
