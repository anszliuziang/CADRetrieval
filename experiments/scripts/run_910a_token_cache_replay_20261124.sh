#!/usr/bin/env bash
set -euo pipefail
umask 027

out=$1
rt=/home/ma-user/work/CADRetrieval_runtime
py="$rt/envs/cadretrieval-910a-cann80-torch240/bin/python"
manifests="$rt/four_tables_inputs_20261124/prior_share4/runs/idea01_in_domain_pilot_20261123/manifests"
cad="$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
bert="$rt/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"
snapshot="$rt/runs/idea01_910a_speed_source_snapshot_step00064_20261124"
replay=experiments.scripts.replay_910a_training_token_cache
test -d "$out" && test ! -e "$out/started_utc.txt"
test -z "$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)"
date -u +%FT%TZ > "$out/started_utc.txt"
git rev-parse HEAD > "$out/git_sha.txt"
trap 'code=$?; date -u +%FT%TZ > "$out/ended_utc.txt"; printf "%s\n" "$code" > "$out/controller_exit.tmp"; mv "$out/controller_exit.tmp" "$out/controller_exit.txt"' EXIT

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE MASTER_ADDR MASTER_PORT ASCEND_RT_VISIBLE_DEVICES
check_resources() {
    local phase=$1 chip port
    npu-smi info > "$out/npu_$phase.txt"
    ss -H -lntu > "$out/listeners_$phase.txt"
    for chip in 4 5 6 7; do
        grep -Eq "^\|[[:space:]]+$chip[[:space:]]+910ProA[[:space:]]+\|[[:space:]]+OK\b" "$out/npu_$phase.txt"
        grep -Eq "^\|[[:space:]]+No running processes found in NPU $chip[[:space:]]+\|" "$out/npu_$phase.txt"
    done
    for port in 23101 23102 $(seq 23200 23231); do
        if grep -Eq ":$port[[:space:]]" "$out/listeners_$phase.txt"; then
            printf 'Reserved replay port %s is already bound\n' "$port" >&2
            return 11
        fi
    done
}
check_resources before_cache

files=(checkpoint.json progress-b.pt run.json steps.jsonl validation-support.json)
sha=$(git rev-parse HEAD)
(
    cd "$snapshot"
    sha256sum --strict -c <<'SHA'
cfb09fd49e5ba8b0bbad69be9c571980db544c3ab3f9d8050e1ab1bb76817390  checkpoint.json
34cf770210d359005310ad7aea3eaa20622345261aca5adbf171e5b1dff6d056  progress-b.pt
7dd54fc345f5741129ab5c46879832f2e0ec17c59339db185d16a5beb4b8a04e  run.json
3be4b475b1d7e6cbe77e25da326d480cd2fd4c2a999c99991134ae2334d3f4f2  steps.jsonl
b2266d2dc6e112b9f81fcf2b68072363121cb19aca9f41bb0a21c17dadee9c91  validation-support.json
SHA
) > "$out/snapshot_check.txt"
for arm in online cached; do
    mkdir -p "$out/$arm/results"
    git worktree add --quiet --detach "$out/$arm/repo" "$sha"
    for f in "${files[@]}"; do cp --reflink=auto -- "$snapshot/$f" "$out/$arm/results/$f"; done
done

logged() {
    local where=$1 limit=$2 code=0
    shift 2
    mkdir "$where" || return "$?"
    printf '%q ' "$@" > "$where/argv.txt"
    date -u +%FT%TZ > "$where/started_utc.txt"
    timeout --signal=TERM --kill-after=25s "$limit" "$@" > "$where/stdout.log" 2> "$where/stderr.log" || code=$?
    date -u +%FT%TZ > "$where/ended_utc.txt"
    printf '%s\n' "$code" > "$where/exit_code.txt"
    return "$code"
}

run_arm() (
    arm=$1 chips=$2 port=$3 hccl_port=$4
    shift 4
    cd "$out/$arm/repo"
    export ASCEND_RT_VISIBLE_DEVICES="$chips" HCCL_IF_BASE_PORT="$hccl_port"
    prefix=("$py" -B -u -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1
            --master_port="$port" --max_restarts=0 -m)
    logged "$out/$arm/hccl" 95s "${prefix[@]}" experiments.scripts.probe_yunnao_910a hccl
    logged "$out/$arm/train" 110m "${prefix[@]}" "$replay" replay --manifests "$manifests" --bert "$bert" \
        --cad-zip "$cad" --out "$out/$arm/results" "$@"
    test -s "$out/$arm/results/replay_receipt.json"
)

(
    export ASCEND_RT_VISIBLE_DEVICES=6
    logged "$out/cache_build" 35m "$py" -B -u -m "$replay" build --manifests "$manifests" --bert "$bert" --out "$out/token_cache"
)
check_resources before_arms
run_arm online 4,5 23101 23200 &
online=$!
run_arm cached 6,7 23102 23216 --cache "$out/token_cache" &
cached=$!
printf 'online %s\ncached %s\n' "$online" "$cached" > "$out/arm_pids.txt"
online_rc=0
wait "$online" || online_rc=$?
printf '%s\n' "$online_rc" > "$out/online/arm_exit.txt"
cached_rc=0
wait "$cached" || cached_rc=$?
printf '%s\n' "$cached_rc" > "$out/cached/arm_exit.txt"
test "$online_rc" -eq 0
test "$cached_rc" -eq 0
logged "$out/check" 180s "$py" -B -m "$replay" check --online "$out/online/results" \
    --cached "$out/cached/results" --out "$out/comparison.json"
