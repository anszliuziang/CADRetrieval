#!/usr/bin/env bash
# Reuse the isolated environment from the verified publisher example.
set -euo pipefail
umask 027
run=$1
approved_sha=$2
code=$(cd -- "$(dirname -- "$0")" && pwd -P)
py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
demo="$run/duoduoclip_cpu_demo"
renders="$run/duoduoclip_full/renders"
control="$run/control/duoduoclip_full"
output="$run/external/duoduoclip"

test -x "$py"
test "$(git -C "$code" rev-parse HEAD)" = "$approved_sha"
test -z "$(git -C "$code" status --porcelain)"
if git -C "$code" symbolic-ref -q HEAD >/dev/null; then
    printf 'DuoduoCLIP requires a clean detached execution checkout\n' >&2
    exit 2
fi
test "$(cat "$demo/demo.actual_exit")" = 0
test -d "$demo/overlay"
test -s "$renders/render_summary.json"
test ! -e "$output"
mkdir -- "$control"
mkdir -- "$control/hf" "$control/tmp"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES="" ASCEND_RT_VISIBLE_DEVICES=""
export HF_HOME="$control/hf" XDG_CACHE_HOME="$control/hf" TMPDIR="$control/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$demo/overlay"
args=(
    "$py" -B -u "$code/encode_retrieval_benchmark_duoduoclip.py"
    --snapshot "$demo/snapshot" --threads 8 encode
    --frozen "$run/frozen" --renders "$renders" --output "$output"
    --image-batch 2 --text-batch 32
    --condition ct_deepcad --condition l2 --condition l0 --condition l1 --condition l3
)
printf '%q ' "${args[@]}" > "$control/argv.txt"
printf '\n' >> "$control/argv.txt"
printf '%s\n' "$approved_sha" > "$control/implementation.sha"
date -u +%FT%TZ > "$control/started.txt"
nohup nice -n 5 bash -c '
    set +e
    result_file=$1
    code=$2
    shift 2
    "$@"
    result=$?
    if [[ "$result" == 0 && -n "$(git -C "$code" status --porcelain)" ]]; then
        result=2
    fi
    printf "%s\n" "$result" > "$result_file.tmp"
    mv -- "$result_file.tmp" "$result_file"
    exit "$result"
' duoduoclip "$control/actual_exit" "$code" "${args[@]}" < /dev/null \
    > "$control/stdout.log" 2> "$control/stderr.log" &
printf '%s\n' "$!" > "$control/launcher.pid"
printf 'duoduoclip_pid=%s\n' "$!"
