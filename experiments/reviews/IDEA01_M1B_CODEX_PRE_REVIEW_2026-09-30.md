# M1b CADmium runner 草稿审查：暂不准入 NPU

对象：Kimi K3 官方 API 本地草稿 `experiments/src/idea01/cadmium_runner.py`（尚未提交）；此审查只看本地代码。`python -m py_compile` 成功，原有 `test_cadmium_scorer.py` 为 31 passed；**没有 runner 自身的 CPU 测试、真实权重前向或吞吐结果**。

## 必须修正后再请求 pilot

1. **资产完整性未封闭。** `preflight_assets` 把 SHA manifest 作为可选项，只检查 base 两个权重 shard 的存在；adapter 权重也只检查存在。当前合同要求在任何真实前向前校验全部实际使用文件（权重、tokenizer、config、adapter）与已锁定的来源 SHA。生产入口必须有不可绕过的完整 SHA 清单；小文件也不能只查存在。`--no-revision-path-check` 若保留，只能在明确的测试入口，正式 CLI 不可放宽。
2. **运行授权未封闭。** `main()` 接受调用者任意给的 `--max-pairs`，不核对 `ACTIVE_TASK.md` 的 `pilot_authorized`、完整 `approved_sha`、真实 `git rev-parse HEAD`、卡数与墙钟上限。`BudgetLedger` 的默认 0 不能替代此门。正式入口须在载入权重前 fail closed，并将批准的对数/卡时或墙钟绑定到一次运行台账；本地 stub 测试可注入假授权，不能把测试开关暴露给生产命令。
3. **输入隔离需机器检查。** CLI 当前从每行取两个字段，但不拒绝额外的 UID/qrels/annotation 字段。输入必须严格只含 `query_text,candidate_json_desc`，否则非零退出；输出的 pair index 与独立清单映射由外层管理。不能把已有 qrels 或原 CSV 当 runner 输入。
4. **单对 CPU 复核重复前向。** `spot_check_with_cpu_core()` 当前先 `score()` 再重新 forward 同一 pair，共消耗 2 对；G1 的“1 对真实模型”预算与其不符。优先用同一次真实 logits 分别做设备 gather 与已审 CPU 核心对照；若确需二次前向，合同和台账都须明确改为 2，对计时/数值一致性另行解释。
5. **输出与重启声称过强。** `main()` 以 `w` 打开结果文件，失败时可能留下部分/空文件；台账却已在前向前扣除整个 pending 集合。当前不能称“可重入/幂等”。为 batch 测速和长作业做原子分片、输入/代码/模型 SHA 绑定、完成标记和失败后保守扣账；单对阶段至少保证已有结果不被静默覆盖。
6. **交接缺失。** 按 M1b 合同增加 CPU stub 测试，覆盖以上拒绝路径、shift/padding、预算和输出。写 `experiments/submissions/idea01_text2cad_benchmark_20260928/IMPLEMENTATION.md`，含测试命令、真实执行命令、资源预测、未完成项，结尾 `review_required`。批量 benchmark 脚本可以下一次短任务交付，但不能据此草稿进入 NPU。

## 2026-09-30 追审：授权来源仍为阻塞级

Kimi 的第一次修复为上述 1–3 增加了 46 项 CPU 测试（本地 `46 passed`），但新 `authorize_run()` 从 **detached HEAD 工作树里的** `experiments/ACTIVE_TASK.md` 读 `approved_sha` 并要求其等于 HEAD。这在 Git 中不可能成立：一个提交无法在自身文件里预写自己的 SHA。假 `ACTIVE_TASK.md`＋假 git HEAD 的单元测试掩盖了该时序问题。项目已有真实解决方案：`experiments/src/idea01/p1_runner.py::fetch_authorized_active_task` 在确认 origin URL 后 `git fetch origin`，读取 **`origin/master:experiments/ACTIVE_TASK.md`**，再比对该授权文本中的完整 `approved_sha` 与 detached 运行 HEAD；`run_p1_matrix.py` 亦按此执行。M1b 须复用/等价实现这一双版本授权，并测试“代码 SHA 与授权所在提交 SHA 不相同但授权批准代码 SHA”的真实 Git 情形。不得为了绕过自引用悖论放宽 SHA 检查。

审查后仍需复核 `load_model()` 的真实 LoRA/基座适配、NPU FP32 log-softmax 与 CPU 核心同 logits 的偏差、batch=1/B 数值等价和准确 HBM。当前 `ACTIVE_TASK.md` 的 NPU 预算保持 0。
