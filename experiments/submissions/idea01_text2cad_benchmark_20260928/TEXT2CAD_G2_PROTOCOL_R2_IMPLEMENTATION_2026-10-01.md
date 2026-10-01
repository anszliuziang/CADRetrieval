# Text2CAD G2 测速协议 R2 修复：实现报告（2026-10-01）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`，真实 checkpoint/BERT 模型前向/NPU 预算全部 0）
- 依据：`experiments/reviews/IDEA01_M1C_G2_R1_CODEX_REVIEW_2026-10-01.md` 的 8 项必须修复 + 同轮补齐项 + 补充已核问题（throughput 强制双卡同步、ETA 核验同步证据）
- 合同：`experiments/plans/IDEA01_M1C_G2_PROTOCOL_REPAIR_2026-10-01.md`
- 起点 HEAD：`4fc2825e4126e5d4757332a07d2de0c4c99e1b84`；R1 报告 `TEXT2CAD_G2_PROTOCOL_IMPLEMENTATION_2026-10-01.md` 保留未改
- 本阶段未连接服务器、未部署、未 push、未安装依赖、未读 qrels/CSV/UID 正例映射、未生成真实 token 资产或 G2 配置；全部修复与测试为本地 CPU 假数据。

## 1. 问题 → 修复 → 针对性红绿证据

红绿日志（checkout 外）：`D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g2_protocol_r2_cpu_20261001/`
`red_r2a.txt`、`red_r2bc.txt`、`red_r2de.txt`、`red_r2fg.txt`、`green_r2_final.txt`。

### R2-1 双卡 barrier 共同绑定（review #1 + 补充已核问题）
- 问题：R1 各卡把自己的单 shard `tier_plans_sha256` 放进要求双方一致的 binding，两卡必然首轮同步互拒；且 sync 可整体省略绕过。
- 修复：新增 `dual_shard_plan_binding(selection, tiers)`（协议模块，CLI 唯一构造入口）：按**固定 shard 顺序** `[shard0, shard1]` 计算 `combined_tier_plans_sha256` 作为共同 `common_plan_sha256` 写入 barrier binding；各卡 `local_plan_sha256` 单独记录，驱动在每个同步点宣告本方 local SHA 并核验对端等于预期的另一 shard local SHA（`expected_peer_local_plan_sha256`），不符即拒。同步点 key 保持共同 point（`t{tier}_tt{t}_{layer}_r{rep}`，无 shard 前缀）。throughput **强制** `--sync-dir` 与 `--sync-group` 同时给出（在任何资产/清单加载前拒绝，缺一或全缺都拒）；结果 `sync` 块记录 `enabled/group/timeout/common/local/shard_id`。
- 证据（红：`dual_shard_plan_binding` 不存在 3 failed）→ 绿：`TwoShardSyncDriverTest` 用**来自两个不同 shard 的真实计划**、两个线程各持真实文件 `SyncBarrier`、以 CLI 同一构造函数产出共同 binding 同步通过（18 个共同同步点、18 个真实测量窗口事后重叠核验；连续 5 次运行稳定）；错误预期对端 local SHA 被拒；`DualShardPlanBindingTest` 证明 common ≠ 两端 local 且篡改任一 shard 计划即变；`ThroughputSyncArgsTest` 证明缺任一同步参数 rc=2。

### R2-2 显式 device 的 HBM 采集（review #2）
- 修复：`peak_device_memory_bytes`/`reset_peak_device_memory` 显式把 device 传给 `torch.npu.max_memory_allocated(device)` / `reset_peak_memory_stats(device)`；**真实 NPU 读取失败直接抛错**（不再 try/except 成 None），CPU 假驱动才允许 None。tier 记录新增 `hbm_device`（实测目标卡），ETA 核验两卡各自 device 参数。
- 证据（红：2 failed）→ 绿：`HbmDeviceTest` 用 fake `torch.npu` 断言以 `"npu:1"` 调用、读取失败抛 RuntimeError、CPU 返回 None。

### R2-3/4 ETA 完整两卡核验（review #3/#4 + 补充已核问题）
- 修复（`estimate_sharded_eta` 重写核验段）：
  - 逐卡独立核：身份（env.device 与归档键一致）、float32/trim none/cache false（**两卡分别**）、invocation 互异、git 40-hex 一致、checkpoint/pair_manifest/score name+spec 一致、selection SHA 一致。
  - **同步证据强制**：两份结果 `sync.enabled=true`、同 group、共同 `common_plan_sha256`；缺 sync 块/伪 disabled/不同组均拒（R2 新增用例）。
  - **计划绑定重建**：用 selection + 记录的 `batch_tiers` 重建双 shard 计划，核对两结果 common SHA 等于重建值、各卡 local SHA 等于对应 shard 重建值；两卡 `batch_tiers` 必须相同、无重复档、`tiers` 记录与之一一对应（拒 dup/缺失/额外 tier）。
  - **逐 cell 精确**：测量 cell 键集合与 selection 该 shard 期望集合严格相等（拒缺失/额外 cell）；每 cell rep 编号 ≥3 且唯一连续自 0 起；逐 rep `n_batches/n_pairs` 与 selection 样本批一致；**逐 batch 真实 pair/顺序/尺寸**与重建样本批逐一相等；e2e 有限为正。
  - **并发凭据**：两卡 point×rep 键集合完全相同（卡内无重复点）后，逐对窗口重叠 ≥ 较短 80%；两卡同时缺同一 rep 的结果也会被 rep 数核验拒绝（R2 新增用例）。
  - **HBM 真实有效**：所计档 `peak_device_memory_bytes` 必须为正整数且 `hbm_device` 等于该卡。
- 证据（红：9 failed）→ 绿：`ShardedETATest` 新增 6 个 R2 测试方法覆盖上述全部拒绝类，原慢卡/精确权重/overhead/红绿边界用例在升级后的富格式夹具上全绿。

### R2-5 selection 严格键集合（review #5）
- 修复：`_diff_path` 双向严格（actual 多出键即 drift，覆盖任意嵌套层）；`verify_g2_selection` 顶层 allowlist = 派生小节 + `provenance`，其余一律拒；`verify_g2_selection_against_manifest` 要求 provenance 键精确等于六元 allowlist（texts/cad_length/pair_manifest/token_audit/bert_file/generated_by）。
- 证据（红：顶层 qrels/evaluation_map/captions 注入、cell 内 evaluation_map、profile64 内 qrels、provenance 注入全部通过=漏洞）→ 绿：注入逐项拒绝，缺键仍拒绝。

### R2-6 运行时 token-audit 真实文件绑定（review #6）
- 修复：新 `verify_g2_token_audit(selection, token_audit_path, manifest_dir)`：`--token-audit` 必须给出且文件存在；`sha256_file` 等于 selection provenance 的 `token_audit_sha256`；复用 `token_lengths_from_audit` 核 schema + texts/BERT provenance；审计内容逐文本等于 selection 内嵌表。`_gated_setup(g2=True)`（profile-consistency/throughput）与 `run_eta_g2` 均强制执行，未给或不符拒绝。本轮仍不生成真实资产/config。
- 证据（红：4 failed + EtaG2CliTest 缺 `--token-audit` 用例）→ 绿：`G2TokenAuditBindingTest`（happy/缺参/缺文件/SHA 不符/内容不符）与 `EtaG2CliTest`（真实审计文件入夹具、缺参 rc=2）全绿。

### R2-7 每 cell 暖身 + 预算同源重计（review #7）
- 修复：`run_production_batch_tiers` 对**每个 cell** 先 2 passes（该 cell 前两个样本批、扣账不计时）再 3 次计时 rep；`plan_production_forwards` 改为同一逐 cell 枚举。**G2 预计对数更新（不再沿用 47,160）**：tier 1/2/4/8/16/32 = 480 / 960 / 1,980 / 4,020 / 8,100 / 15,780 对/卡，合计 **31,320 对/卡、两卡 62,640 对**（已用生产结构合成数据经 `plan_production_forwards` 实算复核）。
- 证据（红：410 != 570 两处）→ 绿：5×37 fixture 计划=扣账=570（逐 cell 手算），charge 事件结构断言 20×(2×2+3×2)+4×(2+3)。

### R2-8 非有限失败证据 + 账本退出状态（review #8）
- 修复：`score_pairs_detailed` 新增 `evidence_dir/evidence_label`：任一批前向/评分抛错（如非有限 logits 触发 `ScorerInputError(nonfinite_logits)`）时，先把该批**原始 logits + variant + 逐 pair 索引 + 批编号**原子落盘 checkout 外（`failed_logits_<variant>_b<ordinal>.pt`，weights_only 可读），再原样抛出；`run_profile_consistency` 对参考与每个变体传入。`_write_result(exit_code=, note=)`：gate_failed 的 batch-consistency/trim-check/cache-check/profile-consistency 以 `exit_code=3, note="gate_failed"`  finalize 账本（通过仍 0/"ok"）。
- 证据（红：无证据文件、`_write_result` 无参、per-head 字段缺失 5 failed）→ 绿：NaN 注入真实评分路径产生 `failed_logits_batch32_b0.pt`（variant/ordinal/32 pairs/NaN logits 逐项核验）；`WriteResultLedgerTest` 用真实 `SharedBudgetLedger` 核 exit 3/gate_failed 与 exit 0/ok 两种最终状态；**gate 失败用故障注入的 BertTextCache（+100 腐蚀缓存嵌入）走真实比较与容差判定失败**，不再 mock 掉 `compare_gate_results`（charges=[64]×5、恰 4 变体、logits 证据齐全）。

### 同轮补齐：profile 逐对报告
- `_profile_variant_report` 逐对新增：`abs/normalized_score_diff_{x,y,total}`、`n_valid_x/y`（ref/alt）、`valid_seq_len`、`n_real_tokens`；变体汇总 `max_abs/max_normalized_score_diff` 取**三个分数分量**的最大（单头漂移不会被另一头抵消）。`test_per_pair_head_diffs_and_counts_reported` 逐变体核验。

## 2. 受影响测试与结果（本地 CPU，`D:\miniconda\python.exe`；未跑全套、未装历史缺依赖）

```
pytest experiments/tests/test_text2cad_bench_protocol.py \
       experiments/tests/test_text2cad_speed_bench.py \
       experiments/tests/test_text2cad_frozen_runner.py -q
154 passed, 18 subtests passed in 61.49s      # green_r2_final.txt
```

- 两shard真实同步线程测试连续 5 次运行全部通过（抗调度抖动：100ms 假工作 + 合同 80% 规则留给 ETA 单测）。
- 未改动：模型数学、固定容差、原 G1 配置与报告、旧门代码语义（仅 `_write_result` 账本退出状态如实化）、CLAUDE/ACTIVE/plans/reviews/configs。

## 3. 完整后续命令（**全部未授权**；统一变量、全 revision 路径、无省略）

```bash
RT=/home/ma-user/work/CADRetrieval_runtime
PY=/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python
REPO=/home/ma-user/work/CADRetrieval
CKPT=$RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth
BERT=$RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b
CADZIP=$RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip
MANIFEST=$RT/pair_manifest_v4
AUDIT=$RT/derived/text2cad_text_token_lengths_20261001.json
G2CFG=experiments/configs/text2cad_g2_pair_selection_20261001.json
KEYS=experiments/configs/text2cad_checkpoint_expected_keys_20260930.json
ENVS="PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RT/hf XDG_CACHE_HOME=$RT/xdg TMPDIR=$RT/tmp"
```

### 3.1 CPU 派生资产（tokenizer 审计；**未授权**；可在远端生成派生资产、零模型前向）
```bash
env $ENVS $PY $REPO/experiments/scripts/run_text2cad_speed_bench.py \
  --mode audit-text-lengths \
  --pair-manifest-dir $MANIFEST --bert-dir $BERT \
  --out $AUDIT --repo-root $REPO
```

### 3.2 G2 selection 配置（**未授权；必须本地运行**：配置只在本地生成、提交后经 Git 同步；不在服务器生成配置回传入库）
```bash
# 在本地仓库机器上，以取回的派生资产（非代码）为输入：
python experiments/scripts/run_text2cad_speed_bench.py \
  --mode select-g2 \
  --pair-manifest-dir <本地资产目录>/pair_manifest_v4 \
  --token-audit <本地资产目录>/derived/text2cad_text_token_lengths_20261001.json \
  --out <本地checkout外>/text2cad_g2_pair_selection_20261001.json
# Codex 核验 provenance/权重后提交为 $G2CFG，再 push、服务器 fetch/detach
```

### 3.3 704 对扩展 G1（profile-consistency；**未授权**，需 pilot_authorized + ≥704 对额度）
```bash
env $ENVS $PY $REPO/experiments/scripts/run_text2cad_speed_bench.py \
  --mode profile-consistency \
  --checkpoint $CKPT --bert-dir $BERT --cad-seq-zip $CADZIP \
  --pair-manifest-dir $MANIFEST \
  --expected-keys-manifest $REPO/$KEYS \
  --selection $REPO/$G2CFG --token-audit $AUDIT \
  --out-dir $RT/runs/<profile_run_id> --device npu:0 --cards-used 1 \
  --repo-root $REPO
# 计划=扣账固定 704 对；11 个 profile_logits_*.pt 与结果 JSON 落 out-dir；gate_failed 时账本记 exit 3
```

### 3.4 G2 生产批测速（throughput；**未授权**，需双卡 pilot_authorized；两进程强制同步）
精确预算（`plan_production_forwards` 同源，含每 cell 暖身）：480 / 960 / 1,980 / 4,020 / 8,100 / 15,780 = **31,320 对/卡、两卡 62,640 对**。
```bash
env $ENVS $PY $REPO/experiments/scripts/run_text2cad_speed_bench.py \
  --mode throughput \
  --checkpoint $CKPT --bert-dir $BERT --cad-seq-zip $CADZIP \
  --pair-manifest-dir $MANIFEST \
  --expected-keys-manifest $REPO/$KEYS \
  --selection $REPO/$G2CFG --token-audit $AUDIT \
  --out-dir $RT/runs/<g2_run_id>/card0 --device npu:0 --cards-used 1 \
  --batch-tiers 1,2,4,8,16,32 --allow-large-batch --reps 3 --warmup-passes 2 \
  --sync-dir $RT/runs/<g2_run_id>/sync --sync-group g2_<run_id> \
  --repo-root $REPO
# 卡1：同上，--device npu:1 --out-dir $RT/runs/<g2_run_id>/card1（同 --sync-dir/--sync-group）
```

### 3.5 较慢卡全量 ETA（eta-g2；纯 CPU，**未授权**，输入须为 3.4 的真实 v2 结果）
```bash
env $ENVS $PY $REPO/experiments/scripts/run_text2cad_speed_bench.py \
  --mode eta-g2 \
  --pair-manifest-dir $MANIFEST \
  --production-batch-size 32 \
  --card-result $RT/runs/<g2_run_id>/card0/bench_throughput_<inv0>.json \
  --card-result $RT/runs/<g2_run_id>/card1/bench_throughput_<inv1>.json \
  --token-audit $AUDIT \
  --out $RT/runs/<g2_run_id>/eta_g2.json \
  --repo-root $REPO
# 退出码 0=过七天门，3=未过；ETA 重建双 shard 计划核 common/local 绑定、逐 batch pair 与窗口
```

## 4. 已知限制
- 真实 BERT token 审计、G2 配置、真实双卡测速与 704 对门均未授权未运行；G2 前向/ETA 入口在配置提交前仍 fail-closed（合同预期）。
- 线程级同步演示受单机 CPU 调度限制，真实 80% 窗口重叠只能由未来获授权的双 NPU 运行给出；本报告不外推任何吞吐数字。
- `peak HBM` 的 device 语义按 Codex 只读核验的 `max_memory_allocated(device)`/`reset_peak_memory_stats(device)` 签名实现，真机行为待授权运行确认。
- 完整 29,319,624 对精确评分与七天门不变；ETA 仍严格校验 pair 权重和，不允许子集/rerank。

## 5. Git 交付
- 暂存仅本轮：`experiments/src/idea01/text2cad_bench_protocol.py`、`experiments/src/idea01/text2cad_speed_bench.py`、`experiments/tests/test_text2cad_bench_protocol.py`、`experiments/tests/test_text2cad_speed_bench.py`、本 R2 报告。未改 frozen_runner/scripts/configs/Codex 文档；无数据/权重/缓存。
- R2 修复提交完整 SHA：`55e30d156c6c4c793d1ba5aea146dd19b8c7b9da`（本 SHA 回填紧随其后一个仅本文档的小提交；均未 push，等 Codex 审核）。

review_required
