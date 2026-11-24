#!/usr/bin/env bash
set -euo pipefail
umask 027

kind=$1
out=$2
repo=$(git rev-parse --show-toplevel)
sha=$(git rev-parse HEAD)
rt=/home/ma-user/work/CADRetrieval_runtime
py="$rt/envs/cadretrieval-910a-cann80-torch240/bin/python"
manifests="$rt/four_tables_inputs_20261124/prior_share4/runs/idea01_in_domain_pilot_20261123/manifests"
assets="$rt/four_tables_inputs_20261124/four_tables_model_assets"
cad="$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
bert="$rt/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"
test -z "$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)"
test -x "$py" && test -d "$manifests" && test -s "$cad"
test -d "$out" && test ! -e "$out/started_utc.txt"
date -u +%FT%TZ > "$out/started_utc.txt"
printf '%s\n' "$sha" > "$out/git_sha.txt"
trap 'code=$?; date -u +%FT%TZ > "$out/ended_utc.txt"; printf "%s\n" "$code" > "$out/controller_exit.tmp"; mv "$out/controller_exit.tmp" "$out/controller_exit.txt"' EXIT

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE MASTER_ADDR MASTER_PORT ASCEND_RT_VISIBLE_DEVICES
npu-smi info > "$out/npu_before.txt"
ss -H -lntu > "$out/listeners_before.txt"

free_card() {
    grep -Eq "^\|[[:space:]]+$1[[:space:]]+910ProA[[:space:]]+\|[[:space:]]+OK\b" "$out/npu_before.txt" &&
        grep -Eq "^\|[[:space:]]+No running processes found in NPU $1[[:space:]]+\|" "$out/npu_before.txt"
}

logged() {
    local where=$1 limit=$2 code=0
    shift 2
    mkdir "$where"
    printf '%q ' "$@" > "$where/argv.txt"
    date -u +%FT%TZ > "$where/started_utc.txt"
    timeout --signal=TERM --kill-after=25s "$limit" "$@" > "$where/stdout.log" 2> "$where/stderr.log" || code=$?
    date -u +%FT%TZ > "$where/ended_utc.txt"
    printf '%s\n' "$code" > "$where/exit_code.txt"
    return "$code"
}

if [[ "$kind" == spectra || "$kind" == spectra4 ]]; then
    gate="$rt/runs/idea01_four_tables_spectra_formal_gate_20261124/t2cad-v1.0"
    test "$(cat "$gate/exit_code.txt")" = 0
    if [[ "$kind" == spectra4 ]]; then
        # The 910A four-rank topology requires all four physical IDs in one group.
        for chip in 4 5 6 7; do free_card "$chip"; done
        for port in 22713 $(seq 22912 22927); do
            if grep -Eq ":${port}[[:space:]]" "$out/listeners_before.txt"; then
                printf 'Reserved probe port %s is already bound\n' "$port" >&2
                exit 11
            fi
        done
        export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_IF_BASE_PORT=22912
        prefix=("$py" -B -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4
                --master_addr=127.0.0.1 --master_port=22713 --max_restarts=0 -m)
        logged "$out/hccl" 95s "${prefix[@]}" experiments.scripts.probe_yunnao_910a hccl
        per_rank=4 updates=4 limit=24m device_label=npu
    else
        free_card 1
        export ASCEND_RT_VISIBLE_DEVICES=1
        prefix=("$py" -B -u -m)
        per_rank=16 updates=12 limit=44m device_label=npu:0
    fi
    logged "$out/timing" "$limit" "${prefix[@]}" experiments.scripts.run_in_domain_pilot_retrievers \
        --source t2cad-v1.0 --method spectrabrep --train-split full --manifest-root "$manifests" --official-zip "$cad" \
        --shared-feature-store "$rt/runs/idea01_four_tables_effective_inputs_20261124/deepcad_spectra" \
        --qwen-dir "$assets/qwen3_embedding_0_6b" --upstream "$assets/spectrabrep" --precision fp32 --attention eager \
        --device "$device_label" --batch-size "$per_rank" --eval-batch-size 16 --require-full-validation-text --seed 17 \
        --spectra-timing-updates "$updates" --output "$out/timing/results"
    test -s "$out/timing/results/spectra_timing_probe.json"
    exit 0
fi

test "$kind" = native
test -d "$bert"
for chip in 0 1 2 3 4 5; do free_card "$chip"; done
for port in $(seq 22701 22703) $(seq 22800 22847); do
    if grep -Eq ":${port}[[:space:]]" "$out/listeners_before.txt"; then
        printf 'Reserved probe port %s is already bound\n' "$port" >&2
        exit 11
    fi
done

run_arm() (
    # Isolate the compiler working directories and HCCL port blocks for the three layouts.
    arm=$1 chips=$2 port=$3 hccl_port=$4 multiple=$5
    cd "$out/$arm/repo"
    export ASCEND_RT_VISIBLE_DEVICES="$chips" HCCL_IF_BASE_PORT="$hccl_port"
    torchrun=("$py" -B -u -m torch.distributed.run --nnodes=1 --nproc_per_node=2
              --master_addr=127.0.0.1 --master_port="$port" --max_restarts=0 -m)
    trainer=("${torchrun[@]}" experiments.scripts.run_formal_text2cad_training --source t2cad-v1.0
             --device npu --manifests "$manifests" --cad-zip "$cad" --bert "$bert" --train-only)
    if [[ -n "$multiple" ]]; then trainer+=(--train-text-pad-multiple "$multiple"); fi
    logged "$out/$arm/hccl" 95s "${torchrun[@]}" experiments.scripts.probe_yunnao_910a hccl
    logged "$out/$arm/timing" 38m "${trainer[@]}" --stop-after-updates 8 --out "$out/$arm/timing/results"
    for batch in third tail; do
        logged "$out/$arm/$batch" 12m "${trainer[@]}" --audit-batch "$batch" --out "$out/$arm/$batch/results"
    done
)

for arm in natural pad32 pad64; do
    mkdir "$out/$arm"
    git worktree add --quiet --detach "$out/$arm/repo" "$sha"
done

run_arm pad32 0,1 22701 22800 32 &
pad32=$!
run_arm natural 2,3 22702 22816 '' &
natural=$!
run_arm pad64 4,5 22703 22832 64 &
pad64=$!
printf 'pad32 %s\nnatural %s\npad64 %s\n' "$pad32" "$natural" "$pad64" > "$out/arm_pids.txt"
failed=0
for arm in pad32 natural pad64; do
    code=0
    wait "${!arm}" || code=$?
    printf '%s\n' "$code" > "$out/$arm/arm_exit.txt"
    if (( code != 0 )); then failed=1; fi
done
for arm in pad32 pad64; do
    for batch in third tail; do
        if [[ ! -s "$out/natural/$batch/results/update-witness.pt" || ! -s "$out/$arm/$batch/results/update-witness.pt" ]]; then
            failed=1
            continue
        fi
        logged "$out/$arm/compare_$batch" 180s "$py" -B -m experiments.scripts.compare_formal_text2cad_updates \
            padding "$out/natural/$batch/results" "$out/$arm/$batch/results" || failed=1
    done
done
exit "$failed"
