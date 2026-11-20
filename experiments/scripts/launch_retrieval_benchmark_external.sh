#!/usr/bin/env bash
# Three independent frozen CPU baselines on the shared DeepCAD gallery.
set -euo pipefail
umask 027
run=$1
rt=/home/ma-user/work/CADRetrieval_runtime
code=$(cd -- "$(dirname -- "$0")" && pwd -P)
py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
phase=${3:-first}

test -x "$py"
test -z "$(git -C "$code" status --porcelain)"
test "$(git -C "$code" rev-parse HEAD)" = "$2"
test -f "$run/frozen/freeze.json"
case "$phase" in
    first)
        control="$run/control/external"
        conditions=(--condition ct_deepcad --condition l2)
        ;;
    native_tail)
        control="$run/control/external_native_tail"
        conditions=(--condition l1 --condition l3)
        if git -C "$code" symbolic-ref -q HEAD >/dev/null; then
            printf 'The native follow-up requires a detached execution checkout\n' >&2
            exit 2
        fi
        for method in clip openshape mmcad_v2; do
            prior="$run/control/external/$method.actual_exit"
            if [[ "$method" == mmcad_v2 && -d "$run/control/mmcad_v2_retry1" ]]; then
                prior="$run/control/mmcad_v2_retry1/actual_exit"
            fi
            test "$(cat "$prior")" = 0
            for file in gallery.npy gallery_status.jsonl gallery_meta.json l0_replay.json; do
                test -s "$run/external/$method/$file"
            done
            for text_axis in texts_l1 texts_l3; do
                for suffix in .npy _status.jsonl _meta.json; do
                    test ! -e "$run/external/$method/$text_axis$suffix"
                done
            done
        done
        ;;
    *) printf 'Unexpected external benchmark phase: %s\n' "$phase" >&2; exit 2 ;;
esac
mkdir -- "$control"
mkdir -- "$control/cache" "$control/tmp"
if [[ "$phase" == first ]]; then
    mkdir -- "$run/external"
fi
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES="" ASCEND_RT_VISIBLE_DEVICES=""
export HF_HOME="$control/cache" XDG_CACHE_HOME="$control/cache" TMPDIR="$control/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

for method in clip openshape mmcad_v2; do
    args=(
        "$py" -B -u "$code/encode_retrieval_benchmark_external.py" encode --method "$method"
        --frozen "$run/frozen" --replay "$run/replay/$method" --output "$run/external/$method"
        "${conditions[@]}"
    )
    case "$method" in
        clip)
            args+=(
                --renders "$run/ext_assets/clip_renders"
                --clip-weights "$run/ext_assets/clip_model/model.safetensors"
                --clip-processor "$run/ext_assets/clip_model"
            ) ;;
        openshape)
            args+=(
                --vector-cache "$rt/idea01_openshape_g5_20261116/pc_embeds"
                --bigg-checkpoint "$rt/hf_home/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/743c27bd53dfe508a0ade0f50698f99b39d03bec/open_clip_pytorch_model.bin"
            ) ;;
        mmcad_v2)
            args+=(
                --vector-cache "$rt/idea01_mmcad_g5_v2_20261116/embeds_cache"
                --cache-checksums "$rt/idea01_mmcad_g5_v2_20261116/SHA256SUMS"
                --upstream "$rt/MMCAD-official"
                --checkpoint "$rt/mmcad_weights/baseline_trimodal_v4.pth"
                --stats "$rt/mmcad_weights/brep_norm_stats_v4.npz"
                --text-model "$rt/mmcad_weights/embeddinggemma-300m"
            ) ;;
    esac
    printf '%q ' "${args[@]}" > "$control/$method.argv"
    printf '\n' >> "$control/$method.argv"
    nohup bash -c '
        set +e
        result_file=$1
        shift
        "$@"
        result=$?
        printf "%s\n" "$result" > "$result_file.tmp"
        mv -- "$result_file.tmp" "$result_file"
        exit "$result"
    ' external "$control/$method.actual_exit" "${args[@]}" < /dev/null \
        > "$control/$method.stdout.log" 2> "$control/$method.stderr.log" &
    printf '%s\n' "$!" > "$control/$method.pid"
    printf '%s_pid=%s\n' "$method" "$!"
done
