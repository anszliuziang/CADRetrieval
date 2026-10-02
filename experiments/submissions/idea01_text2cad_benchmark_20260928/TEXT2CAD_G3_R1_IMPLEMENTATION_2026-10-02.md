# Text2CAD G3 R1：证据准入与断点校验修复报告（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`；本轮真实模型/BERT/NPU 预算 0，未做任何真实前向）
- 依据：`experiments/plans/IDEA01_M1C_G3_R1_FIX_2026-10-02.md`（active_subplan）；上位合同 `experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md`；审查 `experiments/reviews/IDEA01_M1C_G3_CODEX_CODE_REVIEW_R1_2026-10-02.md`
- 起点 HEAD：`6df6f94`。原实现 `759691b9` / 原报告 `5422ed8` 保留未动；本阶段未 SSH、未 push、未下载/安装依赖、未做真实模型/BERT 前向或 NPU 操作、未读 qrels/UID 映射/原 CSV/旧模型分数；全部工作为本地 CPU（`D:\miniconda\python.exe`，Python 3.13.5，torch CPU，pytest 9.1.1）
- **实现提交完整 SHA：`7f650d22a6259bb7b25bdc45abb8de989579ff79`**（修复 + 测试；本文件随后单独提交回填）
- 固定选择配置未改：SHA 仍为 `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`（提交内 blob 复算一致，测试钉死该值）
- 预定 run_id：`idea01_text2cad_g3_probe_20261002`（仅预定，未运行；总 804600 对、**最多** 12 卡小时/6 小时墙钟为后续授权上限，不是当前额度，也不是"至少 12"）

## 1. 改动（事实）

| 文件 | 改动 |
| --- | --- |
| `experiments/src/idea01/text2cad_g3_probe.py` | 修改（1232→1974 行）。R1 四项修复全部在此模块的 ETA 证据准入层；仍纯 CPU、import 不拉 torch（测试钉死） |
| `experiments/src/idea01/text2cad_frozen_runner.py` | 修改（1804→1944 行）。probe resume/sidecar 门收紧；主循环 probe 单 query 体与 summary 装配**逐字抽取**为 `execute_probe_query`/`build_run_summary`（行为不变，供真实 producer 测试复用）；非 probe 路径逐字节同前 |
| `experiments/tests/test_text2cad_g3_probe.py` | 修改（1192→2330 行，36→63 测试 + 19 子测试）。原空 chunk 成功 fixture 已改为真实 8046 行合格 chunk；空文件改判拒绝；新增约 27 个测试方法 |

未改：模型数学/score/shift/mask/loader/SharedBudgetLedger 计费算法/batch16/尾14/FP32/完整272行/无 trim/无 cache/原 8046 图库/固定 100-text 配置/原 G3 报告。七个已被 Root AST 比对的函数/类（`assemble_batch`、`forward_and_score_batch`、`score_pair_token_diagnostic`、`load_gated_model_and_store`、`SharedBudgetLedger`、`authorize_text2cad_run`、`ZipCandidateStore`）与 HEAD **AST 逐项相同**（复核脚本输出见 §4）。

## 2. R1 四项修复（事实，对应审查三项 + 同轮 resume 项）

### 2.1 账本/统计/时序联核（审查项 2；Root 见证 1–3）

`verify_probe_budget_evidence` 重写 + 新增 `verify_final_probe_ledger`，ETA 不再只看成功字段：

- 快照（每卡 summary 内嵌账本）：`max_pairs` 必须恰为批准的 804600；`max_card_seconds` 必须 >0 且 **≤43200（最多 12 卡小时）**；本卡 invocation 恰一条、finalized、exit_code 整数 0、cards_used=1、shard 对应、host 有效、`pairs_charged` 恰 402300、`card_seconds` 与 `[started_epoch, ended_epoch]` 区间一致（容差 0.02s，账本 round-2）；其余 invocation 至多一条（对卡）且不得与本卡同 shard（双跑拒绝）、finalized 者必须 exit_code==0（失败片段拒绝）、pending 者不得带 exit code；快照 `pairs_charged` 总数 == 各 invocation 之和；`card_seconds_spent` == 已 finalized invocation 之和且 ≤ 上限。**合法的第一完成卡快照中对卡 pending 被接受**（`test_pending_peer_snapshot_accepted`）。
- **新增必填 `--final-ledger`**：两卡都退出后取回的共享账本 JSON。必须 version=1、绑定两卡的 approved SHA、恰含两条 invocation（每片一条，旧/额外/失败即拒绝）、各自 finalized/exit 0/cards 1/同 host/`pairs_charged` 402300、`started/ended/card_seconds` 与每卡快照**逐值一致**（不一致即"snapshot and final ledger disagree"）、总 `pairs_charged` 恰 804600、`card_seconds_spent` == 两条之和且 ≤ 上限；两卡快照的 cap 必须一致。
- 统计与计划：`pairs_planned`=402300、`stats.pairs_forwarded`=402300、`stats.batches`=25150（50×503）、`stats.pairs_rejected`=0；fresh-run 专证：`chunks_completed_now`=50、`chunks_previously_completed`=**0**（resume 形态拒绝）、`probe_queries_total`=50。
- 时序绑定：每条 timing 的 `invocation_id` 必须等于本卡成功 invocation（Root 的 `not-in-ledger` 拒绝），且每条 query 窗口必须落在该 invocation 的 `[started, ended]` 内（Root 的 epoch 1–3 vs 1000+ 拒绝）；冷启动不得大于 invocation 区间。
- ETA 顶层：无 `final_ledger`（None/空）直接拒绝。

### 2.2 实际 chunk 内容核验（审查项 1；reviewer 见证 empty_chunks）

新增 `verify_probe_chunk_content`，`eta-g3` 对每条 timing 的 chunk **读取并核验**，不再 exists 检查：

- 严格逐行 JSON（撕裂/非法行拒绝），复用已审 `verify_chunk_records`：header shard/range/计数/pair 端点、全部 CHUNK_BINDING_FIELDS + `probe_selection_sha256` 双向 scope、恰好 8046 条稠密升序全局 pair_index、全部 scored、六个分数有限、valid 计数自洽；缺行/重复/空文件/非有限/错 scope 全部拒绝。
- 每行 `score_name`/`spec_version` 必须为固定值；每行 `n_valid_x`/`n_valid_y`/`n_real_tokens` 必须等于**已审 worker `cad_length_index.jsonl`** 的实际有效项数（CLI 从 manifest 目录真实加载）。
- 与 timing 联核：header scored/rejected == timing 对应值；8046 行 valid_total 之和 == timing `n_valid_total`。
- 每卡目录内 `chunk_s*_t*_*.jsonl` 集合必须与已验 timing 的 chunk 集合**一一对应**（外来/多余 chunk 拒绝）。

### 2.3 固定资产绑定（审查项 3；reviewer 见证 wrong_asset_binding）

`verify_probe_summary` 强化，彼此一致不再够用：

- `pair_manifest_sha256` 必须等于 reviewed `text2cad_pair_manifest_expected_20260930.json` 的指纹（CLI 同时用它核验活 manifest；两卡一致的全 0 值拒绝）。
- `bert_file_sha256` 必须逐项等于五份固定 BERT 文件 SHA（两卡一致的全 0 值拒绝）；`max_cad_sequence_length` 必须恰 272（两卡一致的 128 拒绝）；checkpoint/score/dtype/trim/cache/batch/texts_per_chunk 维持原固定检查。
- **实际 load/preload 证据**（新增 `verify_probe_load_evidence`）：`load_report.checkpoint.checkpoint.sha256` 必须为实测固定 checkpoint SHA；`load_report.bert_files` 必须恰为五文件且实测 SHA 逐项等于固定值；`strict_load` 必须 172 张量、0 missing、0 unexpected、numeric_equality 为真；`eval_mode`/`all_parameters_frozen` 为真；`candidate_preload` 必须 8046/8046/0。
- 授权本身：`authorization` 必须 `task_state=pilot_authorized`、`task_id` 为本任务、`actual_sha==approved_sha==git_sha`、`max_pairs` 恰 804600、`card_hours` ∈ (0,12]（最多 12 卡小时）、`cards_used=1`。
- probe/token/G2 指纹沿用原实际绑定（committed 配置钉路径 + 活工件重算 + token 审计文件 SHA 与内嵌表一致 + 冻结 G2 配置 SHA + 配置 SHA 与两卡 scope 一致），且 ETA 拒绝 probe 总数 != 804600 的选择。

### 2.4 resume/sidecar 断点校验（同轮补齐项；reviewer 见证 orphan_timing）

- **fresh-run 遇已有 sidecar 即拒绝**（`load_probe_sidecar_timed_texts`，在任何新 forward 前）：不向旧计时追加、不混证据。
- resume 改用新的 `verify_probe_resume_records`：每条 timing 行按与 ETA 相同的 per-record 规则严格核验（本片固定 probe 成员、scope 全等、shard/device、层/token 事实、窗口有限有序、E2E==end-start、n_pairs=8046/n_batches=503/n_scored=8046/n_rejected=0/valid>0/chunk 文件名），外来/重行/撕裂/缺 time/计数无效都在 **begin_invocation 与模型加载之前**拒绝；不要求 50 条全覆盖（断点本就允许部分完成）。
- sidecar 与已验 chunk **一一对应**：`plan_chunks` 原有"已验 chunk 缺计时行拒绝"保留；新增 `require_sidecar_chunk_correspondence` 拒绝反向——孤立 timing 行（无已验 chunk）不得把该 query 放回 todo 重评（reviewer 的 `orphan_timing_rescored` 现在拒绝）。
- 无 probe 默认路径完全不改。

## 3. 真实 producer → 真实 reader 的 CPU 端到端（事实）

新测试类 `G3ProducerReaderEndToEndTest`（100 文本 × 12 CAD 合成空间，50 picks/片，两"卡"在同一 CPU 进程交错执行以产生真实重叠窗口）：

- **producer 全部是生产代码**：`FrozenMatrixScorer`（假模型 build_tiny_model + 真实候选 zip/`ZipCandidateStore.preload`）→ 真实 `execute_probe_query`（评分 → 原子发布 chunk → 追加 sidecar）→ 真实 `SharedBudgetLedger`（begin/charge/heartbeat/finalize）→ 真实 `build_run_summary`（含账本快照；卡 0 的 summary 在对卡 finalize **之前**生成，天然得到合法 pending-peer 快照）。模型 load_report 为镜像 loader schema 的夹具（本机无真实 checkpoint/BERT，无法真实加载；候选 preload 报告是 store 真实输出）。
- **reader 全部是生产验证**：`verify_probe_summary`（`require_hbm=False` 的小型辅助参数；生产 estimate 永远默认 True，CPU summary 在 `require_hbm=True` 下确证被 HBM 门拒绝）、`verify_probe_budget_evidence`、`verify_probe_timing_records`（含 invocation/窗口绑定）、每个真实产出 chunk 的 `verify_probe_chunk_content`（cad_valid_counts 由 zip 张量真实 `validate_cad_vec` 计算）、真实落盘最终账本的 `verify_final_probe_ledger`；交错窗口重叠率 ≥0.8 实测通过。
- 腐化路径：产出 chunk 改非有限/清空拒绝；追加孤立 timing 行拒绝；最终账本 `pairs_charged` 改 0 拒绝。

小型辅助验证接口（`expected_max_pairs`/`max_card_*_cap`/`require_hbm`/`shard_id` 显式参数）仅供假数据测试；生产 `estimate_g3_probe_eta`/CLI 始终传钉死常量，无任何放宽开关。

## 4. CPU 测试证据（真实输出，已提交状态复核）

```
pytest experiments/tests/test_text2cad_g3_probe.py -q
=> 63 passed, 19 subtests passed in 51.24s

pytest experiments/tests/test_text2cad_frozen_runner.py experiments/tests/test_text2cad_pair_index.py -q
=> 79 passed, 3 subtests passed in 31.90s
```

- 原 36 测试全部保留并通过；新增：chunk 内容核验（有效通过 + 缺文件/空/撕裂/缺行/重行/非 finite/非 scored/scope/binding/score 身份/审记长度索引/timing 联核拒绝）、resume 门（fresh 拒旧 sidecar、严格记录拒绝、孤立/一一对应）、ETA R1（最终账本必需、零统计/零计划/零扣账、超卡时三形态、外来 invocation、epoch 错位、冷启动超区间、pending-peer 通过×2、额外/失败/同片/第三条 invocation、最终账本缺卡/失败/漂移/不一致、resume 形态、固定资产 14 项破坏）、producer→reader 端到端、常量交叉核对（272/172/TASK_ID/chunk schema/804600/12 卡时）、committed 配置 SHA 钉死。
- Root 见证三例（`CODEX_G3_ETA_REFUSAL_WITNESS_20261002.json`）与 reviewer 四例（`READONLY_REVIEW_WITNESS.json`：empty_chunks/wrong_asset_binding/contradictory_ledger/orphan_timing）逐项有对应拒绝测试；两卡一致的全 0 manifest/BERT/128 均按 reviewer 原样（两卡一致地错）拒绝。
- AST 复核（本机实际输出，对比 HEAD 与已提交实现）：

```
SharedBudgetLedger IDENTICAL / ZipCandidateStore IDENTICAL / assemble_batch IDENTICAL
authorize_text2cad_run IDENTICAL / forward_and_score_batch IDENTICAL
load_gated_model_and_store IDENTICAL / score_pair_token_diagnostic IDENTICAL
三个改动文件均无 CR（LF）；text2cad_g3_probe 子进程导入不加载 torch（测试断言）
```

- 未跑无关全套（cadmium/p1/speed_bench 全套）；受影响范围为 g3 probe/frozen runner/pair index 三个文件。

## 5. 准确后续命令（**当前未授权**；Codex 重审批准完整 SHA 与预算后按批准 SHA 使用）

两卡 probe 运行命令与原 G3 报告 §6 相同（runner CLI 未变：`--mode score --probe-selection <committed> --token-audit <derived> --batch-size 16 --texts-per-chunk 1 --cards-used 1 --device npu:{0,1} --shard-id {0,1} --n-shards 2`，out-dir 分别为 `$RUN/card0`、`$RUN/card1`；前置干净 detach 批准 SHA、两卡空闲、`$RUN` 无旧产物；两进程可错开 3 秒）。ACTIVE_TASK 批准口径应为 `max_scored_pairs=804600`、`gpu_budget` 对应**最多 12 卡小时**（43200 卡秒；是上限不是下限）。

CPU ETA（两卡成功完成后，纯 CPU）——**新增必填 `--final-ledger`**：

```bash
# 两卡 exit 0 后，先从共享账本路径取回最终账本（与 summaries 同批归档、计入 SHA256SUMS）：
#   cp "$RT/budget/text2cad_m1c_<approved_sha>.json" "$RUN/final_ledger.json"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/experiments/src" \
  "$PY" -m idea01.text2cad_g3_probe --mode eta-g3 \
  --pair-manifest-dir "$RT/pair_manifest_v4" \
  --token-audit "$RT/derived/text2cad_text_token_lengths_20261001.json" \
  --repo-root "$REPO" \
  --card-summary "$RUN/card0/summary_<invocation_card0>.json" \
  --card-summary "$RUN/card1/summary_<invocation_card1>.json" \
  --final-ledger "$RUN/final_ledger.json" \
  --out "$RUN/eta_g3_20261002.json"
# exit 0 = 通过七天门；exit 3 = 超限；exit 2 = 任何核验拒绝
```

归档要求（在上一轮"100 chunks/timing/summaries/SHA256SUMS"之上新增）：`$RUN/final_ledger.json`（两卡都 finalized 后取回的最终共享账本）必须随 run 产物一并取回本地并归档 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g3_probe_20261002/`；缺它 ETA 直接拒绝（exit 2）。账本文件生产路径为 `$RT/budget/text2cad_m1c_<approved_sha>.json`（`canonical_budget_path`，checkout 外；ACTIVE_TASK 未批准前该文件不应存在，存在即说明有旧运行，ETA 会因额外 invocation 拒绝）。

## 6. 已知限制（事实）

1. `eta-g3` 仍无真实 probe 输出可算；本轮只验证验证器与拒绝路径。G2 的 38.28394 小时仍是唯一真实测量基线；本修复不改变任何 ETA 数学（慢卡/每层最慢 query/冷启动/1.30/604800 秒逐字保留）。
2. 生产 ETA 只接受 fresh-run 每卡一次：resume 成功续跑后的证据（`chunks_previously_completed>0` 或账本含旧 invocation）会被 ETA 拒绝——这是 R1 合同的明确口径；若真实 probe 中断并 resume 成功，须改合同才能用该证据定价，届时回报原因。
3. producer 端到端测试的模型 load_report 为 schema 镜像夹具（本机无真实权重）；checkpoint/BERT 的实测 SHA 绑定逻辑由手造合法/破坏 fixture 单独覆盖，真实加载证据只能在 NPU 运行后核验。
4. 快照至多容忍"本卡 + 对卡（可 pending 或已成功）"两条 invocation；第一完成卡快照中"对卡已 finalized 但 exit_code=None"（陈旧回收）按失败处理拒绝——该情形在 fresh 轮本不应出现。
5. `--final-ledger` 与两卡 summary 快照的卡秒 cap 若不一致（同文件不可能合法漂移）只分别独立核验 ≤43200 与花费 ≤ cap；两快照 cap 不一致本身已拒绝。
6. runner main() 的 CPU 覆盖仍到授权前的门（生产 checkout 钉死）；resume 门与 producer 路径经函数级 + 端到端测试覆盖，main() 内部顺序（sidecar 门在 begin_invocation/加载前）需 Codex 审 diff 复核。
7. G2/G1 预算与账本不可复用；本轮未申请也未消耗任何前向预算，未 SSH、未 push。

## 7. 状态

`review_required`：实现与测试已提交（SHA 见文首），原 G3 报告未覆盖，等待 Codex 复算 CPU 证据与独立审查；批准后另行 push 与 NPU 授权（总 804600 对、**最多** 12 卡小时/6 小时墙钟为上限，不是当前额度）。
