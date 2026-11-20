#!/usr/bin/env bash
# Three independent frozen CPU baselines on the shared DeepCAD gallery.
set -euo pipefail
umask 027
run=$1
rt=/home/ma-user/work/CADRetrieval_runtime
code=$(cd -- "$(dirname -- "$0")" && pwd -P)
py=/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python
control="$run/control/external"

test -x "$py"
test -z "$(git -C "$code" status --porcelain)"
test "$(git -C "$code" rev-parse HEAD)" = "$2"
test -f "$run/frozen/freeze.json"
mkdir -- "$control"
mkdir -- "$control/cache" "$control/tmp" "$run/external"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES="" ASCEND_RT_VISIBLE_DEVICES=""
export HF_HOME="$control/cache" XDG_CACHE_HOME="$control/cache" TMPDIR="$control/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

for method in clip openshape mmcad_v2; do
    args=(
        "$py" -B -u "$code/encode_retrieval_benchmark_external.py" encode --method "$method"
        --frozen "$run/frozen" --replay "$run/replay/$method" --output "$run/external/$method"
        --condition ct_deepcad --condition l2 --condition l1 --condition l3
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
