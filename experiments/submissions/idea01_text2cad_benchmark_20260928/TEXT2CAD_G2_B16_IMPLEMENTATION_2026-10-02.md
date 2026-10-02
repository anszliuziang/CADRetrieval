# Text2CAD G2 batch16 双卡测速：启动前同步最小修复实现报告（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`；模型/BERT 前向与 NPU 预算均为 0，本轮无 GPU 授权）
- 依据：`experiments/plans/IDEA01_M1C_G2_B16_2026-10-02.md` §2；上位子合同 M1c 2026-09-30；最新审核 `experiments/reviews/IDEA01_M1C_G1_PROFILE_CODEX_REVIEW_2026-10-01.md`
- 起点 HEAD：`db7fa65`（Codex 新计划提交）；本阶段未连接服务器、未部署、未 push、未安装依赖、未做任何模型/BERT 前向、未读 qrels/UID 评价映射；全部工作为本地 CPU。
- 预定 run_id：`idea01_text2cad_g2_b16_20261002`（仅预定，未运行）。

## 1. 最小改动（事实）

唯一源码改动，`experiments/src/idea01/text2cad_bench_protocol.py:778`：

```diff
-        poll_interval_s: float = 0.2,
+        poll_interval_s: float = 0.001,
```

保留不变：`timeout_s=180`（`SYNC_TIMEOUT_SECONDS`）、marker 原子写与 exists 拒绝、绑定/schema/同设备/peer 失败核查、marker 永不删除、等待计入卡时而不混入计时窗口、`MIN_WINDOW_OVERLAP_FRACTION=0.80` 门。全仓库仅此一处默认值（`grep poll_interval` 确认：runner 与脚本均未显式传该参数，仅测试用显式小值）。未改模型、score、配置、ETA 数学、分片/候选清单、正式 runner、容差，未新增同步协议或自动重试。

测试改动，`experiments/tests/test_text2cad_bench_protocol.py` 的 `SyncBarrierTest` 新增 2 个 CPU 回归测试（共 +43 行）：

- `test_default_poll_interval_is_sub_skew_resolution`：不显式传参构造 `SyncBarrier`，断言默认 `poll_interval_s == 0.001`（钉死合同值）。
- `test_default_poll_short_window_real_overlap`：真实双线程、**默认**轮询（不覆盖参数）、注入 0.03 s 到达差（npu:1 晚到）、`arrive` 返回后才计时 0.10 s 模拟窗口（同步等待不混入窗口），断言 `window_overlap_fraction >= MIN_WINDOW_OVERLAP_FRACTION`。

## 2. CPU 红绿证据与见证（事实）

红（修复前，合同附件，真实双进程 spawn）：`D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g2_b16_cpu_20261002/sync_witness_before.json`，SHA256 `37092f684fdf7a57d4cef5f3644434991cbfcc0323464691d9a071619b3140cc`。默认 poll=0.2 s 下 `overlap_fraction=0.0`（窗口 1790909921.483–.584 与 .654–.755 完全错开），`meets_existing_0_8_rule=false`。

绿（修复后，同一见证几何，真实双进程 spawn）：见证脚本复制为 `.cache/text2cad_sync_cpu_witness_after_20261002.py`，**仅**替换输出文件名（输出 `sync_witness_after.json`、拒绝覆盖 before），注入到达差 0.03 s 与模拟工作 0.10 s 原样不变：

```
PYTHONDONTWRITEBYTECODE=1 D:/miniconda/python.exe .cache/text2cad_sync_cpu_witness_after_20261002.py
=> overlap_fraction = 0.9947520604166964, meets_existing_0_8_rule = true
```

产物 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g2_b16_cpu_20261002/sync_witness_after.json`，SHA256 `7cd08fc82b76ca7e10c4cd40a80ddd75b335498942d34c77a4a38ad6740b446d`。两窗口起点差约 1.1 ms（.4747527 vs .4758725）；before 文件未动。

测试命令与结果（本地 CPU，`D:\miniconda\python.exe`，Python 3.13.5，pytest 9.1.1，未安装任何依赖）：

```
pytest experiments/tests/test_text2cad_bench_protocol.py -q -k "SyncBarrier or RepWindowOverlap"
=> 13 passed, 45 deselected in 10.45s

pytest experiments/tests/test_text2cad_bench_protocol.py experiments/tests/test_text2cad_speed_bench.py -q
=> 110 passed, 25 subtests passed in 36.96s
```

110 = R3 已审 108 + 本轮新增 2；无 skip/xfail/警告性豁免。

## 3. 静态计划核对（事实，纯 CPU 枚举，无模型前向）

用已审固定配置 `experiments/configs/text2cad_g2_pair_selection_20261001.json`（SHA256 `fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce`，未改）调用 `dual_shard_plan_binding(selection, [16])` + `plan_production_forwards(..., warmup_passes=2, reps=3)`：

- shard0 = 8,100 forwards；shard1 = 8,100 forwards；合计 16,200（含 warmup），与合同 §3 预计一致。
- 每 shard 33 个 cell；30 个 cell 各 4 批、3 个 cell 各 2 批（完整批 16、尾批诚实建模）。
- `common_plan_sha256 = e88343f660ab955dccb1844befd7029362cedd7d89b0dfc49b431d8fc831a603`。
- 完整矩阵 3,644×8,046 = 29,319,624 对；每卡分片权重 14,659,812 对。
- token 审计 SHA `9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8`（未改）。

## 4. 未来两卡命令（**未授权**，仅供 Codex 审核后按批准 SHA 使用）

服务器 `yunnan-share2`，checkout `/home/ma-user/work/CADRetrieval`（=`$REPO`），环境 `/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python`（=`$PY`），运行根 `$RT=/home/ma-user/work/CADRetrieval_runtime`，`$RUN=$RT/runs/idea01_text2cad_g2_b16_20261002`。两进程同 host、同 run 的 `sync/`、唯一 sync_group，分别：

```bash
# card 0（文本分片 [0,1822)，device npu:0）
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" TMPDIR="$RT/tmp" timeout 3600 \
  "$PY" "$REPO/experiments/scripts/run_text2cad_speed_bench.py" --mode throughput \
  --checkpoint "$RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth" \
  --bert-dir "$RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b" \
  --cad-seq-zip "$RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip" \
  --pair-manifest-dir "$RT/pair_manifest_v4" \
  --expected-keys-manifest "$REPO/experiments/configs/text2cad_checkpoint_expected_keys_20260930.json" \
  --selection "$REPO/experiments/configs/text2cad_g2_pair_selection_20261001.json" \
  --token-audit "$RT/derived/text2cad_text_token_lengths_20261001.json" \
  --batch-tiers 16 --allow-large-batch --warmup-passes 2 --reps 3 \
  --cards-used 1 --device npu:0 \
  --sync-dir "$RUN/sync" --sync-group g2_b16_20261002 \
  --out-dir "$RUN/card0" --repo-root "$REPO"

# card 1：同上，仅 --device npu:1 与 --out-dir "$RUN/card1" 不同
```

前置条件（未满足前不得执行）：Codex 在 ACTIVE_TASK 另写 `pilot_authorized` 与完整批准 SHA、远端 fetch/detach 到该 SHA 且 `git status --porcelain` 为空、两卡只读核验健康且无其它 NPU 进程、`$RUN` 下 `card0/card1/sync` 无任何旧产物或 marker。本机当前无任何此类授权；`gpu_budget=0`、`max_scored_pairs=0`。

## 5. 分析（解释与判断，与上面事实分开）

- 根因与修复匹配：before 见证中晚到方（+0.03 s）立即看到对端 marker 并开始计时，早到方却在 0.2 s 轮询里睡过整个 0.10 s 窗口（起点差约 0.171 s），overlap=0。poll 降至 0.001 s 后，轮询唤醒延迟对 0.10 s 窗口可忽略（实测起点差约 1.1 ms，overlap 0.9948），修复直接消除该诊断症状，且未触碰任何评分/计时语义。
- Windows 本地 `time.sleep` 粒度最坏约 15.6 ms，即使如此起点差上界约 0.03+0.016 s，仍满足 ≥80%（需 ≤0.02 s 起点差）之外的余量很小但实测远好于上界；服务器 Linux 上 sleep 分辨率更高，预期不差于本机。这是判断，不是实测保证。
- CPU 见证只证明同步轮询分辨率修复有效；按计划 §4，真实 NPU 双卡每 rep 窗口重叠 ≥80% 仍须在测速中逐点取证，不能由本轮 CPU 结果默认通过。
- 静态枚举与合同预计的 8,100/卡、16,200 总对一致；Codex 仍须独立复核后才写授权，我的枚举不替代其复核。

## 6. 剩余限制与未做事项

- 未做任何模型/BERT 前向、未 SSH、未部署、未下载、未安装依赖、未改远端 checkout；G1 batch32 失败未重试，trim/cache 未测试，旧账面 512 对余额未用。
- 未改 ACTIVE_TASK/CLAUDE.md/plans/reviews/assets/configs/旧报告；`.cache/` 两个见证脚本与 checkout 外见证 JSON 均不进 Git（`.gitignore` 根 `/*` 规则）。
- 两卡真实并发、HBM、吞吐与七天 ETA 全部未知，等待 Codex 审核与批准 SHA 后的授权测速。

## 7. Git 交付

- 暂存仅本轮：`experiments/src/idea01/text2cad_bench_protocol.py`、`experiments/tests/test_text2cad_bench_protocol.py`、本报告。
- 实现提交完整 SHA：**（提交后回填，见下行）**
- SHA 回填：实现提交为 `IMPLEMENTATION_SHA_PLACEHOLDER`（随后一个仅本文档的回填提交记录该 SHA；均未 push，等 Codex 审核）。

review_required
