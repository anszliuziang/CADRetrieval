#!/usr/bin/env bash
# Launch only from a clean detached project checkout on yunnao-share4.
set -eo pipefail
umask 027
mode=$1
auth=$2
run=$3
attempt=${4:-first}
rt=/home/ma-user/work/CADRetrieval_runtime
code=$(cd -- "$(dirname -- "$0")" && pwd -P)
py=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python

case "$mode" in
    probe|score) ;;
    *) printf 'Unexpected benchmark stage: %s\n' "$mode" >&2; exit 2 ;;
esac
[[ "$attempt" =~ ^[a-z0-9_]+$ ]]
test -x "$py"
test -f "$run/frozen/freeze.json"
test -f "$run/control/historical_replay.json"
test -z "$(git -C "$code" status --porcelain)"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
control="$run/control/$mode"
if [[ "$attempt" != "first" ]]; then
    control="$control"_"$attempt"
fi
mkdir -- "$control"
mkdir -- "$control/hf" "$control/tmp" "$control/xdg"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GIT_TERMINAL_PROMPT=0
export HF_HOME="$control/hf" XDG_CACHE_HOME="$control/xdg" TMPDIR="$control/tmp"
unset ASCEND_RT_VISIBLE_DEVICES
args=(
    "$py" -B -u "$code/run_retrieval_benchmark_text2cad.py" controller
    --mode "$mode" --auth-sha "$auth"
    --input "$run/frozen"
    --replay "$run/control/historical_replay.json"
    --checkpoint "$rt/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth"
    --bert-dir "$rt/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"
    --cad-zip "$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
)
if [[ "$mode" == "probe" ]]; then
    args+=(--out "$run/probe")
else
    args+=(--out "$run/full" --probe-result "$run/probe/probe_result.json")
fi
printf '%q ' "${args[@]}" > "$control/launch_command.txt"
printf '\n' >> "$control/launch_command.txt"
date -u +%FT%TZ > "$control/launch_time.txt"
git -C "$code" rev-parse HEAD > "$control/implementation.sha"
nohup bash -c '
    set +e
    result_file=$1
    code=$2
    shift 2
    "$@"
    result=$?
    if [[ $result == 0 && -n "$(git -C "$code" status --porcelain)" ]]; then
        result=2
    fi
    printf "%s\n" "$result" > "$result_file.tmp"
    mv -- "$result_file.tmp" "$result_file"
    exit "$result"
' benchmark "$control/controller.actual_exit" "$code" "${args[@]}" \
    </dev/null >"$control/stdout.log" 2>"$control/stderr.log" &
printf '%s\n' "$!" > "$control/launcher.pid"
printf 'controller_pid=%s\n' "$!"
