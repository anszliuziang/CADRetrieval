#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
rt=/home/ma-user/work/CADRetrieval_runtime
python=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
run=$1
case "$run" in "$rt"/runs/idea01_in_domain_pilot_*) ;; *) exit 2 ;; esac
if [[ -n "$(GIT_OPTIONAL_LOCKS=0 git -C "$repo" status --porcelain)" ]] || git -C "$repo" symbolic-ref -q HEAD; then
    printf 'A clean detached deployment is required\n' >&2
    exit 2
fi
mkdir -- "$run"
export PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- "$repo"
git rev-parse HEAD > "$run/code_sha.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run/cpu_started_utc.txt"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$run/cpu_exit.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run/cpu_ended_utc.txt"
}
trap finish EXIT
phase() {
    local name=$1
    shift
    local rc=0
    printf '%q ' "$@" > "$run/$name.argv"
    printf '\n' >> "$run/$name.argv"
    "$@" > "$run/$name.stdout" 2> "$run/$name.stderr" || rc=$?
    printf '%s\n' "$rc" > "$run/$name.exit"
    return "$rc"
}
t="$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
ct="$rt/synchronized/20261123_af4bd77/in_domain_pilot_inputs/benchmark_source_checks_20261120/cad_translator_dataset.zip"
phase sanity "$python" -B -m pytest -p no:cacheprovider -q experiments/tests/test_in_domain_pilot.py
phase manifests "$python" -B -u experiments/scripts/run_in_domain_pilot.py prepare \
    --annotations "$t/text2cad_v1.0/text2cad_v1.0.csv" \
    --split "$t/text2cad_v1.0/train_test_val.json" \
    --official-zip "$t/cad_seq.zip" \
    --minimal-zip "$t/text2cad_v1.0/misc/minimal_json/minimal_json_0000_0099.zip" \
    --ct-zip "$ct" \
    --old-test-gallery "$rt/evaluation_inputs/m0_v3/gallery.jsonl" \
    --out "$run/manifests" --workers 12
status=0
pids=()
for source in t2cad-v1.0 ct-deepcad; do
    for subset in pilot full; do
        name="ridge_$source"_"$subset"
        phase "$name" "$python" -B -u experiments/scripts/run_in_domain_pilot.py ridge \
            --manifests "$run/manifests" --source "$source" --subset "$subset" \
            --out "$run/$name" --workers 8 &
        pids+=("$!")
    done
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
