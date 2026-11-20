# DuoduoCLIP-B/32 benchmark implementation plan

This baseline uses the frozen eight-view condition in §4 of the [retrieval benchmark plan](IDEA01_RETRIEVAL_BENCHMARK_FILL_2026-11-20.md). The [configuration](../configs/retrieval_benchmark_duoduoclip.json) records the publisher revision, model and processor manifests, geometry inputs, cameras, rendering and text window. The eight cube-corner views are the CAD benchmark protocol. The authors report varying the number of views at inference, including experiments through 12 views. [ICLR paper, §§3–4 and Fig. 5](https://proceedings.iclr.cc/paper_files/paper/2025/file/77d8a1387c3dbceab2a90c9af0e8d830-Paper-Conference.pdf)

## Checkpoint and inference

- `3dlg-hcvc/DuoduoCLIP-B-32` is a public MIT release. Its pinned model snapshot is `5eb499c3cbd9503240cc539a97b566e0f8f8582c`. The fully converted visual and text weights occupy 605,156,676 bytes, with SHA-256 `6447cba02eacfbaba869ed9eab9db34ed985a58100db2f267536b157b9e89c12` from the publisher's LFS pointer and model API. The checkpoint bytes themselves are pending download. Both custom Python files were read at the exact revision and hashed. They load the local CLIP modules, implement the six cross-view attention layers and view pooling, and return raw 512-dimensional projections. [Pinned model card](https://huggingface.co/3dlg-hcvc/DuoduoCLIP-B-32/blob/5eb499c3cbd9503240cc539a97b566e0f8f8582c/README.md) · [Weight metadata](https://huggingface.co/api/models/3dlg-hcvc/DuoduoCLIP-B-32/revision/5eb499c3cbd9503240cc539a97b566e0f8f8582c?blobs=true) · [Model source](https://huggingface.co/3dlg-hcvc/DuoduoCLIP-B-32/raw/5eb499c3cbd9503240cc539a97b566e0f8f8582c/modeling_duoduoclip.py)
- The model card supplies a runnable demo with three 512 × 512 RGBA images and three example texts. All three image SHA-256 values were checked against the downloaded bytes. Decode to PIL RGB as the demo does. The documented image call sends `(1,3,3,224,224)` and returns `(1,512)`. A batch of `B` eight-view CAD models must send `(B,8,3,224,224)` and return `(B,512)`. A flat `(B*8,3,224,224)` call without `num_views=8` encodes individual views, so it would bypass joint encoding. Readout is cosine after per-row L2 normalization. [Publisher quick start and demo files](https://huggingface.co/3dlg-hcvc/DuoduoCLIP-B-32/tree/5eb499c3cbd9503240cc539a97b566e0f8f8582c)
- Use Transformers `4.57.6`, the version in the checkpoint configuration, and record the actual PyTorch, Pillow, tokenizers, NumPy and OCC builds. The Windows `cadclip` environment provides the render stack. Its Transformers `5.20.2` lacks `clip_loss` and moved `CLIPTextModel.text_model`, both used by the publisher source. The local `hf_venv` has Transformers `4.53.1` and lacks PyTorch. There is no local DuoduoCLIP weight cache or hosted publisher demo inference endpoint. The genuine smoke forward needs a compatible compute environment and the verified weight file. [Pinned model configuration](https://huggingface.co/3dlg-hcvc/DuoduoCLIP-B-32/raw/5eb499c3cbd9503240cc539a97b566e0f8f8582c/config.json) · [Transformers 4.57.6 CLIP source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/clip/modeling_clip.py)
- Report the authors' disclosed four training datasets, Objaverse, ABO, ShapeNet and 3D-FUTURE, and OpenCLIP initialization. CAD benchmark training overlap remains unverified. [ICLR paper, §4.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/77d8a1387c3dbceab2a90c9af0e8d830-Paper-Conference.pdf)

## Candidate and text inputs

- DeepCAD uses `frozen/gallery.jsonl` from the current runtime. Its 8,032 UID-keyed STEP exports in the sibling `mmcad_steps_20261115` directory match every frozen UID with zero extra files. Sum of STEP bytes is 307,261,212. The source manifest hashes `uid TAB lowercase-file-SHA256 LF` in gallery order. The renderer resolves `0000/00000093` to `0000__00000093.step` and uses the original noncontiguous `cad_index` for ties. The 8,032 historical PNGs use a single camera and provide provenance for the same G5 minimal-JSON build, while this baseline produces a new eight-view set. [G5 STEP export provenance](../scripts/run_mmcad_step_export_20261115.py) · [G5 render specification](../configs/clip_g5_render_spec_20261115.json)
- These STEP files come from the floating minimal-JSON reconstruction used by G5 CLIP and MM-CAD. The native Text2CAD score uses its quantized program. The historical `create_cad_model()` can return a valid prefix when a later sketch build fails. All 16 pilot STEP solids matched the strict floating minimal-JSON replay in solid count, bounding box and volume. Maximum absolute differences were `3.41e-13` for a bounding-box coordinate and `1.33e-13` for volume. The full library lacks this strict audit and quantized replay can differ. [Pinned Text2CAD reconstruction](https://github.com/SadilKhan/Text2CAD/blob/669a81472ede2269837c22a4d00070eaac360c82/CadSeqProc/cad_sequence.py#L900)
- Fusion waits for the final retained gallery, HDF5-to-program conversion and verified solid for each UID. The direct CadTranslator archive has HDF5 but zero STEP files and zero exact filename UID matches against the Autodesk reference archive. The old E1 Fusion and the CadTranslator Fusion input have different identity spaces. Use the validated shape associated with each final Fusion UID and apply the same camera, tessellation, material and processor. Give the final geometry manifest its own SHA-256. Retain unannotated candidates and use the finalized qrels. [Fusion conversion and checks](../scripts/prepare_retrieval_benchmark_fusion.py)
- DeepCAD text axes are `l0`, `l1`, `l2`, `l3` and `ct_deepcad`, with 3,502, 6,300, 7,604, 7,614 and 2,029 unique strings respectively. Read `texts_<condition>.jsonl` in `text_index` order and use the exact strings with the `{q}` template. Count complete tokens including special tokens and keep the right-truncated prefix at 77 tokens, preserving the required end token. Each `queries_<condition>.jsonl` row registers one target independent of text reuse. Encode gallery geometry once for all five conditions. [Axis preparation](../scripts/prepare_retrieval_benchmark_fill.py)

## Tasks and acceptance

- [x] Render the pilot with `render_retrieval_benchmark_duoduoclip.py`. Validate the transferred solid, normalize its exact B-Rep box center and half diagonal, and use the E1 OCC tessellator. Render all eight corners with the fixed orthographic scale and camera-relative Lambertian illumination. Verify 448 × 448 RGB images, a nonempty foreground, stable pixel hashes on a repeat, correct occlusion and ordered UID status. Record Python, OCC, NumPy and Pillow versions on the actual render run. Reviewed through-holes on `0000/00000093` and the asymmetric part `0007/00070266` in the saved contact sheet.
- [ ] Stage and hash the pinned checkpoint in the authorized model environment. Compare both official Python source files and every invoked processor file against the config before importing. Use a private offline Hugging Face code cache and a local snapshot containing the pinned bytes. Use FP32 in `eval()` and `inference_mode()` and `attn_implementation="eager"`. Reject missing, unexpected and mismatched loaded parameters. Execute the publisher's three-image text/shape demo and record the raw shape/dtype, normalized vectors' norms and the three scores. Repeat one multi-view item with a reversed view order and as part of a second batch, comparing normalized output within `2e-5` absolute tolerance. Exercise a real retained CAD with eight different images and confirm the joint image API returns one vector.
- [ ] Create a dense C-order little-endian float64 gallery embedding matrix in gallery row order, one status row per UID and a metadata sidecar. Use all eight validated renders in one call per object, allowing multiple objects per batch. If an object fails, record a zero row with the stable UID and a reason. For every text condition, save the equivalent text matrix and token length, truncation and failure status in text index order. Reuse `normalize_rows` with `epsilon=0` and `write_axis` from `encode_retrieval_benchmark_external.py`. Set `method="duoduoclip"`, record `weights` identically in both sidecars, and record the geometry, renderer, snapshot and axis digests. `gallery.npy`, `gallery_status.jsonl` and `gallery_meta.json` share one gallery. Name text artifacts `texts_<condition>.npy`, `texts_<condition>_status.jsonl` and `texts_<condition>_meta.json`.
- [ ] Run `eval_retrieval_benchmark_fill.py --runtime <run> external --condition <condition> --method duoduoclip --input <embedding-dir>` for each DeepCAD text condition. The shared evaluator checks axes, matrix digests and unit norms, ranks registered UIDs by cosine with ascending `cad_index` for exact ties, and gives an unencoded query or positive target rank `N+1`. The evaluator supports an arbitrary external method ID and needs no historical replay for DuoduoCLIP. Preserve the five independent summaries, per-query ranks and CT–DeepCAD's unique-raw-text R@20. On Fusion use the same embedding contract after the finalized Fusion evaluator joins the separate UID axis. [Common evaluator](../scripts/eval_retrieval_benchmark_fill.py)

## Pilot and budget

Run from the execution worktree. The production render drops `--limit` and `--stride`, and uses the separate output directory `duoduoclip_render`. The summary commits ordered geometry status and hashes after all selected candidates have been processed. A restart verifies and reuses completed views with the same source, code, versions and configuration. Shapes with a render error keep their original row and a failure reason.

```powershell
$assetRuntime = 'D:/projects/IdeaSpark/CADRetrievalIdea1_runtime'
$benchmarkRun = "$assetRuntime/idea01_retrieval_benchmark_fill_20261120"
D:/miniconda/envs/cadclip/python.exe -B experiments/scripts/render_retrieval_benchmark_duoduoclip.py `
  --frozen "$benchmarkRun/frozen" --steps "$assetRuntime/mmcad_steps_20261115" `
  --output "$benchmarkRun/duoduoclip_render_pilot" --limit 16 --stride 501
```

Two clean pilots rendered 16/16 UIDs and 128/128 valid RGB PNGs in 14.58 and 14.41 seconds. Independent verification matched the UID, historical index and PNG bytes across both runs. The 128 images occupied 597,230 bytes, with 3,903-byte median and 10,119-byte p95. Eight poses can produce identical images for symmetric parts. All eight remain in the ordered input. Runtime evidence is in `duoduoclip_render_pilot`, `duoduoclip_render_pilot_repeat` and `duoduoclip_render_pilot_contact_sheet.png` under the benchmark run root. The camera and occlusion tests passed with the installed Python 3.10.22, OCC 7.7.2, NumPy 2.2.6 and Pillow 12.3.0.

Naive extrapolation is about 1.9 hours of rendering in one CPU process and 0.300 GB of PNG payload for 64,256 views. Budget 0.50 GB for images, status and filesystem overhead. Renderer tail time depends on tessellation complexity. The five DeepCAD text axes contain 27,049 unique rows. The 512-D float64 gallery and text matrices total 143,691,776 payload bytes. The embeddings and axis statuses suffice to reconstruct the 1.738 GB dense score matrix. Fusion needs eight images per retained UID and 4,096 bytes per candidate or unique text embedding row. Allow 0.61 GB for the weight snapshot, plus another 0.61 GB when an independent staged copy is present.

The G5 single-view CPU image pass took 585 seconds for 8,032 shapes. Eight times the images gives a rough 78-minute starting estimate before DuoduoCLIP's extra cross-view attention work. Treat 1.5–3 hours for image and text CPU encoding as planning only. Measure with the pinned model on the intended hardware. Start the encoder smoke batch at two shapes and record peak memory and throughput. Available Windows memory varied from 0.87 to 2.2 GB during the prep. Peak checkpoint load memory still needs measurement. [Historical CLIP timing](../submissions/idea01_g5_external_baselines_20261115/CLIP_G5_RESULTS_2026-11-15.md)

## Local text census

This census used the exact publisher `tokenizer.json` with the installed tokenizers 0.21.4 backend. The longest L3 input retained the right-truncated prefix and end token. Confirm the counts on the actual model's `CLIPProcessor` in the encoder preflight. Only seven DuoduoCLIP L3 targets fit in the original window even before intersection with other methods' coverage.

| Condition | Unique text >77 tokens | Registered queries >77 tokens | Maximum tokens |
| --- | ---: | ---: | ---: |
| L0 | 210 / 3,502 | 224 / 7,614 | 102 |
| L1 | 71 / 6,300 | 71 / 7,614 | 533 |
| L2 | 5,115 / 7,604 | 5,115 / 7,614 | 502 |
| L3 | 7,607 / 7,614 | 7,607 / 7,614 | 1,698 |
| CT–DeepCAD | 0 / 2,029 | 0 / 7,513 | 20 |

## Running the model and evaluator

Stage the ten model files and three demo images from the pinned public revision. The [encoder](../scripts/encode_retrieval_benchmark_duoduoclip.py) checks the complete snapshot before importing model code, rejects extra files that could override the model, and loads using an empty offline module cache. It runs CPU FP32 with Transformers 4.57.6, `CLIPImageProcessor`, `CLIPTokenizerFast` and eager attention. Select `python` from the isolated model environment, set `benchmark_run` to the current runtime and set `snapshot` to the staged revision directory.

```bash
python -B experiments/scripts/encode_retrieval_benchmark_duoduoclip.py --snapshot "$snapshot" audit

python -B experiments/scripts/encode_retrieval_benchmark_duoduoclip.py --snapshot "$snapshot" --threads 4 \
    demo --output "$benchmark_run/duoduoclip_demo_only.json"

python -B experiments/scripts/encode_retrieval_benchmark_duoduoclip.py --snapshot "$snapshot" --threads 4 \
    encode --frozen "$benchmark_run/frozen" --renders "$benchmark_run/duoduoclip_render" \
    --output "$benchmark_run/external/duoduoclip" --image-batch 2 --text-batch 32 \
    --condition ct_deepcad --condition l2

python -B experiments/scripts/eval_retrieval_benchmark_fill.py --runtime "$benchmark_run" \
    external --condition ct_deepcad --method duoduoclip --input "$benchmark_run/external/duoduoclip"
```

`audit` verifies the local files without importing Torch. `demo` runs the publisher's three images and three texts, records scores and unit norms, and checks invariance to view reversal. `encode` also verifies the full 8,032-row render manifest, checks joint encoding on two real eight-view CADs, and tests individual and reversed-view calls against the joint batch. It reads each valid UID's eight PNGs by checksum and feeds an exact `(B,8,3,224,224)` input. Text encoding verifies the right-truncated prefix and EOS against each untruncated token sequence.

Omit `--condition` to encode all five text axes. Use multiple `--condition` arguments to prioritize a subset, or reuse the directory with the same run settings to add more conditions. Each complete axis is reused after checking provenance, digests and finite unit-or-zero rows. The limits are 1–8 shapes per image batch, 1–128 queries per text batch, and 1–32 CPU threads. The render pilot directory has partial coverage and cannot be used for full encoding. When an encoding stops during publication, the runner rejects an output axis missing its metadata. A fresh encoding directory can rerun that attempt.
