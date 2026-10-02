# Text2CAD G3 R2：卡时授权绑定与同卡串行时间轴 — 实现报告

2026-10-02。状态：**review_required**。任务 `idea01_text2cad_benchmark_20260928`；合同 `experiments/plans/IDEA01_M1C_G3_R2_BINDING_FIX_2026-10-02.md`；审查依据 `experiments/reviews/IDEA01_M1C_G3_CODEX_CODE_REVIEW_R2_2026-10-02.md` 的两项 Important。

- 起点 HEAD：`6d8445e`。R1 实现 `7f650d22` / R1 报告回填 `da476b23` 保留未动；本阶段未 SSH、未 push、未下载/安装依赖、未做真实模型/BERT 前向或任何 NPU 操作、未读 qrels/UID 映射/原 CSV/旧模型分数；全部工作为本地 CPU（`D:\miniconda\python.exe`，Python 3.13.5，torch CPU，pytest 9.1.1）。
- 固定选择配置 SHA 复核不变：`experiments/configs/text2cad_g3_probe_selection_20261002.json` = `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`（`test_committed_config_is_lf_and_regeneration_is_byte_identical` 通过）。score/shift/mask/FP32/full272/b16/尾14/无 cache trim/8046 原图库/ETA 数学均未触碰。
- 本报告完整实现 SHA：**见文末"提交回填"**（提交前为空）。

## 1. 改动文件（仅两个，+233/-4）

| 文件 | 改动 |
| --- | --- |
| `experiments/src/idea01/text2cad_g3_probe.py` | 两门新检查 + 文档串；不改 SharedBudgetLedger 算法、不改 ETA 公式 |
| `experiments/tests/test_text2cad_g3_probe.py` | 两个受影响旧测试自洽化、两处直调补必传参、三个新测试 |

未改 `experiments/src/idea01/text2cad_frozen_runner.py`（生产 runner 在 `text2cad_frozen_runner.py:1788` 本就按 `authorization.card_hours * 3600.0` 创建账本 cap；本轮绑的是 ETA 读取外部证据时的核验）。未改任何计划/审查/CLAUDE/ACTIVE_TASK/旧报告。

## 2. 门 1：卡时授权 == 本卡 snapshot cap == final ledger cap

对应审查 Important 1（`text2cad_g3_probe.py:863/946/1082` 各自只查上限、可矛盾）。

- `verify_probe_budget_evidence`（现 `text2cad_g3_probe.py:978-996`）：在读出本卡 snapshot `max_card_seconds` 并通过 ≤43200 上限检查后，新增绑定——`authorization.card_hours` 必须是有限正数，且 `card_hours*3600` 与 snapshot cap 相等（`math.isclose`，rel_tol=1e-9，abs_tol=1e-6）。授权 0.1 小时（360 卡秒）对 snapshot 43200、实际花费 3000 的审查见证现在拒绝：`... != the ledger snapshot cap 43200.0; the approved card-time budget must equal the ledger cap it is charged under`。返回字典新增 `authorized_card_seconds` 字段（合同许可的"扩验证器返回字段"）。
- `estimate_g3_probe_eta`（现 1736-1741）：新增两卡授权相等检查 `len(authorized_seconds_set) == 1`，两卡授权不一致拒绝。既有两卡 snapshot cap 相等（`ledger_caps`）不变。
- `verify_final_probe_ledger`：新增**必传**关键字参数 `expected_max_card_seconds`（无默认值，fail-closed；合同许可的"扩传参"），要求 final ledger `max_card_seconds` 与两卡 snapshot cap 相等（同容差，现 1144-1150）。final cap 3001 对 snapshot 43200 的审查见证现在拒绝：`... must carry the SAME approved card-time cap`。`estimate_g3_probe_eta` 在 `ledger_caps` 唯一后传 `next(iter(ledger_caps))`。
- 上限语义不变：三处仍各自要求 ≤ `G3_APPROVED_MAX_CARD_HOURS`（12 卡小时 = 43200 卡秒）；本轮后续实际授权固定 12/43200，生产路径无需任何改动即可满足绑定。
- 合法路径保留：第一完成卡 snapshot 中 peer pending 仍合法（`test_pending_peer_snapshot_accepted` 通过）；SharedBudgetLedger 计费算法零改动（runner/pair 不变项复核通过）。

## 3. 门 2：同卡 query 时间轴按生产 text_index 升序串行不重叠

对应审查 Important 2（原 `text2cad_g3_probe.py:1303` 只核每条窗口落在 invocation 内）。

- `verify_probe_timing_records`（现 1405-1421）：在既有逐条检查（scope/事实/E2E=end-start/完整 8046/批次/invocation 绑定与包含）和 50 条全覆盖检查之后，按 text_index 升序（即生产 todo 顺序，`probe_texts_for_shard` 为排序后的全局 text_index）逐对检查 `next.window.start_epoch >= previous.window.end_epoch`；同起点、相邻重叠、逆序均拒绝，报 `non-serial timeline: text_index ... starts at ... before the previous production query ...`。
- 只按生产顺序（text_index）排序检查，绝不按窗口时间重排来"凑"串行；invocation 范围未扩大。
- 保留不变：每条 E2E/epoch 绑定、invocation 包含、跨卡 80% 并发门（`test_window_overlap_gate` 通过）、每层最慢 query、冷启动、1.30 系数、604800 秒七天门（`test_seven_day_gate_both_sides` 通过；该测试本就用 overhead_seconds 而非假大时长，无需重建 spacing）。大时长合法 fixture 已按合同重建为合法 spacing + 匹配账本区间（见下）。

## 4. 测试与证据（CPU，零前向）

命令（工作目录为仓库根，`D:\miniconda\python.exe`）：

```
python -m pytest experiments/tests/test_text2cad_g3_probe.py -v
python -m pytest experiments/tests/test_text2cad_frozen_runner.py experiments/tests/test_text2cad_pair_index.py -q
```

结果：

- probe 模块：**66 passed、19 subtests passed，52.65s**，0 skip 0 fail。原始输出留存 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g3_probe_cpu_r2_20261002/test_text2cad_g3_probe_r2_20261002.txt`（含逐条 PASSED 清单）。
- 不变项复核（runner 131 行 import probe 模块，pair_index 为共享底层）：**79 passed、3 subtests passed，37.59s**。原始输出同目录 `test_runner_pair_invariance_r2_20261002.txt`。

新增/修改测试（全部在 `experiments/tests/test_text2cad_g3_probe.py`）：

- `G3ETATest::test_authorization_snapshot_cap_binding`（新）：两卡授权均改 0.1 小时（360 卡秒）、snapshot 仍 43200、实际 3000（审查见证原样）→ 拒绝；单卡授权 11 小时对 snapshot 43200 → 拒绝；两卡各自自洽但授权不一致（12h/43200 与 11h/39600）→ 拒绝（different card-second caps）；final cap 3001 与 43199 对 snapshot 43200 → 拒绝（SAME approved）；未变异证据仍通过。
- `G3ETATest::test_same_card_serial_timeline_required`（新）：同卡 50 条 start 全改 1000、保留各条 E2E（审查见证原样）→ 拒绝；相邻重叠（record1 start 落入 record0 窗口内）→ 拒绝；窗口整体逆序（window 与 e2e_s 同步置换，保持每条内部一致）→ 拒绝；合法 back-to-back（next.start==prev.end，step=e2e=8）→ 通过；合法大时长（e2e=300、step=400、invocation 区间重建为 [500,21400]、每卡 20900 卡秒、两卡合计 41800 ≤ 43200 cap）→ 通过且每卡 compute = 300×1822 不变。未放宽任何既有门。
- `G3ProducerReaderEndToEndTest::test_budget_evidence_binds_authorization_to_the_ledger_cap`（新）：真实产出的 summary（授权 1.0 卡时、账本 cap 3600）直调 `verify_probe_budget_evidence` 通过且 `authorized_card_seconds==3600.0`；授权改 0.5 卡时或删除 card_hours → 拒绝。
- `G3ETATest::test_over_card_budget_refused`（改）：前两个超支案例把 authorization 同步改为与变异后 cap 一致（1/3600、1000/3600 卡时），使拒绝仍归因于"花费超 cap"而非新绑定；最后一个 final cap 2000 案例改写为审查见证 final cap 3001 对 snapshot 43200 的绑定拒绝。
- 两处 `verify_final_probe_ledger` 直调（producer-reader 合法路径与漂移账本路径）补传 `expected_max_card_seconds=3600.0`。

保留的 R1 全部真实检查均通过：真实 chunk 内容读取、live 资产/load/resume/producer→reader（`test_real_produced_artifacts_pass_the_real_reader`）、固定资产绑定、真实记账、final ledger 强制、新鲜运行唯一性、合法 peer pending。

## 5. 准确账本路径更正（替代 R1 报告的 `$RT/budget/...`）

R1 报告两处写 `$RT/budget/text2cad_m1c_<approved_sha>.json`，**有误**。实际代码 `cadmium_runner.py:147` `PRODUCTION_BUDGET_ROOT = Path("/home/ma-user/work/CADRetrieval_runtime/budgets")`，`text2cad_frozen_runner.py:201` `canonical_budget_path(approved_sha)` 返回 `$RT/budgets/text2cad_m1c_<approved_sha>.json`（`$RT=/home/ma-user/work/CADRetrieval_runtime`，checkout 外，目录名为复数 **budgets**）。G3 probe 两卡都 finalized 后取回最终共享账本的准确命令以本报告为准：

```
RT=/home/ma-user/work/CADRetrieval_runtime
RUN=<run 产物目录>
cp "$RT/budgets/text2cad_m1c_<approved_sha>.json" "$RUN/final_ledger.json"
```

CPU ETA 仍必须 `--final-ledger "$RUN/final_ledger.json"`（缺失 exit 2）；该最终账本随 run 原始证据归档（本地 `experiments/runs/<run_id>/` 可审计副本 + 共享 OBS），归档要求同 R1 报告不变。ACTIVE_TASK 未批准前该账本文件不应存在。

## 6. 已知缺口与限制

- 仅为本地 CPU 验证器修复与假数据/假模型测试；不构成检索实验结果，G3 仍未上卡、真实前向额度 0。
- 新两门只作用于 ETA 证据核验路径（`verify_probe_timing_records`/`verify_probe_budget_evidence`/`verify_final_probe_ledger`/`estimate_g3_probe_eta`）；resume 门（`verify_probe_resume_records`，允许部分覆盖）按合同未加串行检查——审查只点名 ETA 读取路径，未扩大修复。
- 等值绑定容差 rel_tol=1e-9/abs_tol=1e-6：覆盖 0.1×3600 一类二进制浮点表示误差，同时 360 对 43200、3001 对 43200 这类矛盾必拒绝。
- 两卡授权相等检查在 snapshot cap 相等检查之后执行；授权不一致的实际拒绝通常先由逐卡绑定或 cap 集合检查触发（三者传递一致），显式检查用于浮点边界下的兜底与直接可读性。
- ETA 公式、七天门、80% 并发门、1.30 系数、每层最慢 query 定价、固定盲选与模型数学均未变；G2 真实 ETA 38.28394 小时保留，G3 真实 ETA 待授权后两卡实测。

## 7. 提交回填

- 实现提交 SHA：**待提交后回填**
- 提交内容：仅 `experiments/src/idea01/text2cad_g3_probe.py`、`experiments/tests/test_text2cad_g3_probe.py` 及本报告；未 push；未占 Git index 中任何 Codex 文件。
- 提交后状态：`review_required`，等待 Codex 复审新两门及合法路径；G3 pilot 授权（完整 approved SHA/804600 对/最多 12 卡小时/6 小时墙钟）由 Codex 另行发布。
