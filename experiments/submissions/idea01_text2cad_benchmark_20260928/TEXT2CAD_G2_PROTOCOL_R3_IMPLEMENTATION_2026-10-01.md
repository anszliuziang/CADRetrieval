# Text2CAD G2 测速协议 R3 修复：实现报告（2026-10-01）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`，真实 checkpoint/BERT 模型前向/NPU 预算全部 0）
- 依据：`experiments/reviews/IDEA01_M1C_G2_R2_CODEX_REVIEW_2026-10-01.md` 的最后两项修复（R2 主体 8 项已通过）
- 起点 HEAD：`59ffc9444a77d7c6f6a7e55ec08095783c92fbe9`；R1/R2 报告保留未改
- 本阶段未连接服务器、未部署、未 push、未安装依赖、未读 qrels/CSV/UID 映射、未生成真实资产/config；全部修复与测试为本地 CPU 假数据，不扩大范围。

## 1. 最小改动

### R3-1 `reset_peak_device_memory` 移入 peer-signal 范围（review 末项 1）
- 问题：`run_production_batch_tiers` 每档开头在 try 外调用 `reset_peak_device_memory(device)`；NPU 统计异常会裸抛，对端卡只能等满 180 秒 barrier 超时。
- 修复：把该调用移入 tier 级 try 块首行，落入既有 `except RuntimeError/Exception` → `signal(f"t{tier}_run", "error", exc)` → 原样抛出的路径；不改任何计时/数学语义。
- 证据（红：`ResetHbmFailureTest` 1 failed：无 signal_failure 发生）→ 绿：CPU fake `torch.npu.reset_peak_memory_stats` 抛 `RuntimeError("npu reset broken")`、fake sync 收到 `("t1_run", "error", 含原始消息)` 且原异常继续抛出；未运行真实 NPU。

### R3-2 正式 eta-g2 入口复用 budget 运行证据（review 末项 2）
- 修复：新 `verify_eta_card_budget_evidence(results_by_device)`，在 `run_eta_g2` 组装两卡结果之后、调用 `estimate_sharded_eta` 之前执行。对每份 result：按 `invocation_id` 在自身 `budget.invocations` 中找**唯一**记录（缺失/重复拒绝）；核 `host` 非空且两卡相同、`budget_file` 非空且两卡相同、`shard_id` 对应该卡（npu:0→0、npu:1→1）、`cards_used == 1`、`finalized is True`、`exit_code` 为 int 0（bool 拒绝，JSON true 用例覆盖）。只复用现有 ledger 字段，无新身份字段或新 ledger 格式；纯 `estimate_sharded_eta` 数学函数保持不变供 fixture 使用。
- 证据（红：10 个 subTest 全过=漏洞 + 主用例 1 failed）→ 绿：`EtaG2CliTest.test_budget_evidence_refusals` 通过**真实 CLI** 读篡改后的结果文件逐项拒绝：missing_record / duplicate_record / different_host / empty_host / different_ledger / wrong_shard / cards_used_2 / not_finalized / failed_exit / bool_exit 全部 rc=2；未篡改对子仍走到估算并诚实返回 rc=3（门未过）。夹具结果补齐 `budget.budget_file/invocations` 字段。

## 2. 测试命令与结果（本地 CPU，`D:\miniconda\python.exe`）

```
pytest experiments/tests/test_text2cad_speed_bench.py::ResetHbmFailureTest \
       experiments/tests/test_text2cad_speed_bench.py::EtaG2CliTest -q
红：11 failed（red_r3.txt）→ 实现后 4 passed, 10 subtests passed

pytest experiments/tests/test_text2cad_bench_protocol.py \
       experiments/tests/test_text2cad_speed_bench.py -q
108 passed, 25 subtests passed in 26.23s（green_r3_final.txt）
```

日志（checkout 外）：`D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g2_protocol_r3_cpu_20261001/`（`red_r3.txt`、`green_r3_final.txt`）。未重复全套旧测试，未安装历史缺依赖；模型/评分/容差/旧配置/资产/预算口径（31,320 对/卡、704 对 profile）均未改。

## 3. Git 交付

- 暂存仅本轮：`experiments/src/idea01/text2cad_speed_bench.py`、`experiments/tests/test_text2cad_speed_bench.py`、本 R3 报告。未触碰 Codex 文件（含当前未跟踪的两份 review 与 G1 PROFILE plan）、configs/scripts/其它源文件；无数据/权重/缓存。
- R3 修复提交完整 SHA：见回填（本地提交，不 push）。

review_required
