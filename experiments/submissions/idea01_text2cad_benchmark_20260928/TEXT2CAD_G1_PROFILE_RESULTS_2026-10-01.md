# Text2CAD 固定64对扩展 G1 profile-consistency 结果报告（门失败，如实提交）

- 日期：2026-10-01；执行者：本地 Claude Code / Kimi K3；任务 `idea01_text2cad_benchmark_20260928`。
- **结论（事实）：门未通过。`passed=false`，真实 EXIT=3（gate_failed）。仅完成 reference + batch16 + batch32 共 192/704 评分对；batch32 有 1/64 对超出固定 logit 容差，runner 按 fail-fast 停止，其余 8 个变体（trim32、cache32、trim+cache ×6 档）从未运行。**
- 本报告事实与分析分节；不含 R@K、语义有效性或吞吐结论；不重试、不改容差、不进入 G2。

## 1. 授权与运行 SHA（事实）

- `ACTIVE_TASK.md` 状态 `pilot_authorized`，授权文档 commit：`c45cf32e7937d7e4d963796c6fbfd7b0b48322b5`（origin/master HEAD，runner 从 origin/master 读取并记录 blob SHA256 `9a07bf6febc1e7601b5d58cf859614a67423b4bca9901a7d2e1dd72f65650523`）。
- 批准运行 SHA：`a2219e25b7f018aa12e820616466672fd8d70729`；runner 记录 `approved_sha == actual_sha == a2219e25…0729`。
- run_id：`idea01_text2cad_g1_profile_20261001`；invocation_id：`20261001T062948Z-4fb6b1fc`；主机 `notebook-ceff1d49-f99f-4086-a3b2-3d6f13ed4225`；设备 `npu:0`，`cards_used=1`，FP32，`score_name=two_head_factorized_logprob_sum`，`spec_version=text2cad-vector-score-v1`。
- 结果 JSON：`bench_profile-consistency_20261001T062948Z-4fb6b1fc.json`（214,259 B，SHA256 `0da99847d794bbfb5460a11d168afb5000b84862e5c556fd52802e3230607c00`）。

## 2. 飞行前核查（事实，全部通过）

- 本地：`git status --porcelain` 为空；HEAD=`c45cf32…b5`；`origin/master` 同 SHA；`a2219e2` 为 master 祖先。
- 远端 yunnan-share2：checkout `/home/ma-user/work/CADRetrieval` 干净（原 detached `598dd40b…`）；`git fetch origin` 后 `git checkout --detach a2219e25…0729`，`git rev-parse HEAD` 核对一致，detach 后再查 `git status --porcelain` 为空。运行结束后再次核查仍干净、HEAD 不变（证据在 `preflight_snapshot.txt` / `postrun_snapshot.txt`）。
- `npu-smi info`：两张 Ascend910 均 `OK`，AICore 0%，`No running processes found`；HBM 静态占用 chip0 3,102/65,536 MB、chip1 2,872/65,536 MB（无进程列表项）。
- 目标 OUT `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_profile_20261001` 运行前不存在（`OUT_NOT_EXISTS`），随后 `mkdir -p` 新建，未覆盖任何已有结果。
- 资产逐项 SHA（远端实测 == 已审指纹）：
  - token 审计 `derived/text2cad_text_token_lengths_20261001.json`：`9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8` ✓
  - G2 配置 `experiments/configs/text2cad_g2_pair_selection_20261001.json`（检出 SHA 内）：`fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce` ✓
  - pair_manifest_v4 四项：`pair_manifest.json`=`62086beb…2557`、`texts.jsonl`=`c59b4aef…a21`、`cad_store.jsonl`=`2f441110…1380`、`cad_length_index.jsonl`=`daac66a1…ea0c3` ✓（结果 JSON 内 `pair_manifest_sha256` 亦为 `62086beb…`）
  - checkpoint SHA256（runner 记录）：`235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，与 ACTIVE_TASK 钉值一致；BERT 五文件 SHA 记录在结果 JSON `environment.bert_file_sha256`。

## 3. 执行命令与环境（事实）

远端唯一一次调用（完整命令留存于 run 产物 `run_command.sh`，与授权给定命令逐字一致，变量统一定义 REPO/RT/PY/OUT）：

```
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" TMPDIR="$RT/tmp" timeout 3600 "$PY" "$REPO/experiments/scripts/run_text2cad_speed_bench.py" --mode profile-consistency --checkpoint "$RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth" --bert-dir "$RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b" --cad-seq-zip "$RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip" --pair-manifest-dir "$RT/pair_manifest_v4" --expected-keys-manifest "$REPO/experiments/configs/text2cad_checkpoint_expected_keys_20260930.json" --selection "$REPO/experiments/configs/text2cad_g2_pair_selection_20261001.json" --token-audit "$RT/derived/text2cad_text_token_lengths_20261001.json" --out-dir "$OUT" --device npu:0 --cards-used 1 --repo-root "$REPO"
```

- 经 `nohup` 启动（PID 1438768，runner 实际 PID 1438770），`stdout.log`/`stderr.log` 直接落 OUT，无管道掩盖；EXIT 由包装脚本原样写入 `exit_code.txt`（内容 `3`）。
- 环境（结果 JSON `environment`）：Python 3.11.10，torch 2.7.1+cpu，torch_npu 2.7.1.post1，Transformers 4.53.1，platform Linux-5.10.0-182 aarch64；`env_vars_present`=[ASCEND_TOOLKIT_HOME, HF_HOME, PYTHONDONTWRITEBYTECODE, TMPDIR, XDG_CACHE_HOME]；`time_utc=2026-10-01T06:30:26Z`。
- 未安装/修改任何依赖，未改远端源码，未添加 warmup，未重试。

## 4. 运行过程与门结果（事实）

- 时间线（UTC+8）：14:29 启动；cold_start_s=35.3686；measurement window=39.0036 s；runner `elapsed_s`=3.0639；14:30 结束。`timeout 3600` 未触发。
- 完成变体 3/11（每个变体 64 对，合计 192 对）：
  - `reference_b1`（batch1，trim none，cache false）：基准，`profile_logits_reference_b1.pt`。
  - `batch16`（真实批 [16,16,16,16]，0.2838 s）：**passed=true**，64/64 对通过；max_abs_logit_diff=2.4319e-05，max_abs_score_diff=3.0696e-05，逐对 logit violations=0。
  - `batch32`（真实批 [32,32]，0.2226 s）：**passed=false**，63/64 对通过；**失败对：pair=7（text_index=160，cad_index=164）**，`n_logit_violations=6` / `n_rows_compared=211`，`max_abs_logit_diff=3.528594970703125e-05`；该对 `counts_exact=true`、`score_ok=true`（score_total_ref=-327.91014675516635 vs alt=-327.91013456787914，差 1.22e-05，在 score 容差内）。batch32 整体 max_abs_score_diff=6.1754e-05。
  - 固定容差（两变体相同）：`logits_atol=1e-05, logits_rtol=1e-04, score_atol=1e-04, score_rtol=1e-06`。
- 失败后 runner 按协议 fail-fast：trim32、cache32、trim+cache batch1/2/4/8/16/32 共 8 个变体**从未运行**，对应 logits 文件不存在（不是丢失）；`total_scored_pairs=192`。
- logits 产物 3 份（各 10,081,891 B）：`profile_logits_reference_b1.pt`、`profile_logits_batch16.pt`、`profile_logits_batch32.pt`；全部 `all_finite=true`。逐对字段含双头分数（score_x/score_y/total，ref 与 alt）、逐头有效计数（n_valid_x/y/total，ref 与 alt）、`n_real_tokens_ref`、`valid_seq_len_ref`、逐项 logit/score 差与归一化差，供 Codex 独立复算。
- stdout 仅一行：`{"status": "gate_failed", "result": ".../bench_profile-consistency_20261001T062948Z-4fb6b1fc.json"}`。stderr 仅 torch_npu `collect_env` owner 警告与一条驱动版本 Warning（`Driver Version: … is invalid or not supported yet`），无 Python 异常、无 OOM、无超时。
- HBM：runner 未记录在运行程内 HBM 遥测；仅有运行前后 `npu-smi` 快照（数值相同，见 §2）。**没有运行中 HBM 数据，不编造。**

## 5. 预算账本（事实，未编辑）

- 本 SHA 共享账本 `budgets/text2cad_m1c_a2219e25b7f018aa12e820616466672fd8d70729.json` 已 `finalized=true`：`invocations`=1（exit_code=3，note=`gate_failed`），`pairs_charged=192`，`remaining_pairs=512`，`card_seconds_spent=38.44` / 上限 3600；`stale_invocations_auto_finalized=[]`。
- 账本只读复制为 run 产物 `budget_snapshot_a2219e25.json`（SHA256 `e5b60a77…31389`），原件未动；旧 spot/batch 账本未触碰。

## 6. 取回与归档（事实）

- 本地镜像：`experiments/runs/idea01_text2cad_g1_profile_20261001/`（12 文件：11 项产物 + SHA256SUMS），scp 取回后 `sha256sum -c` 11/11 OK。
- 云脑源目录保留未删：`/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_profile_20261001/`。
- 共享 OBS 归档：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g1_profile_20261001/`，`sha256sum -c` 11/11 成功；cvteam14 仅存储操作、未占 GPU。
- 三方（云脑/本地/OBS）`SHA256SUMS` 清单文件本身哈希一致：`2d232f224ad39713edd7aaf27d6ae9cefa652d70a67ac10aa05070cc807514bd`。
- 关键产物 SHA256：结果 JSON `0da99847…c700`；logits reference `eaed292a…83ee7`、batch16 `e8d7cfa7…9ef07`、batch32 `f3ab9e81…85829`；stderr `e5b8f5fb…ca91a`、stdout `94c0c671…d9003`。

## 7. 分析（执行者判断，非授权、非结论）

- 失败面非常窄：192 对中仅 batch32 的 1 对、211 个比较行中 6 行超 `logits_atol=1e-05`（最大 3.53e-05）；同一对的双头分数、总分与有效计数全部在容差内，batch16 全部通过。现象与“批尺寸增大引入的数值归约差异”一致，但单对单点不能定位根因（kernel 归约顺序、padding 路径或数据相关均有可能）；且本门只测等价性，不能据此推断吞吐或精度趋势。
- 直接后果：按协议 production 仍只能是 reference；trim/cache 任何组合本次未获得任何真实证据（未运行），不能被视为已验证或已否定。
- 建议（供 Codex 决策，非自行执行）：从 3 份原始 logits 独立复算确认 pair 7 的 6 个违规行分布（哪一头、哪些行位）；评估是协议性收紧实现（如批内 padding/归约路径对齐）还是重审容差口径；在未获新授权前不运行任何补档。
- 已知局限：仅 3/11 变体；无运行中 HBM 遥测；wall/variant 计时仅为门副产品，不是测速结论。

## 8. 提交信息

- 本报告为本次唯一新增 Git 文件；精确 stage 仅此文件，不 push，不改 ACTIVE_TASK/CLAUDE/plans/reviews/assets/configs/src 及旧报告。
- 首次提交 SHA：（见下方回填）；回填 commit（仅本文件）：（见下方回填）。
- 状态建议：**results_submitted**（门失败事实提交，等待 Codex 复核与下一步决定；不进入 G2）。
