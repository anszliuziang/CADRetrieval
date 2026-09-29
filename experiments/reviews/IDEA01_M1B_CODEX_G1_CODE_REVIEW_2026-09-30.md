# M1b G1 单对前向代码审查

**后续状态：本审查已失效。** 首次真实前向发现 adapter 缺键警告与可换路径重置预算的缺陷；见 `IDEA01_M1B_G1_INVALIDATION_2026-09-30.md`。不得据下文旧结论再次执行。

审查对象：本次提交所含 `cadmium_runner.py`、对应 CPU 测试、两份固定配置；批准运行的完整代码 SHA 由下一次 `ACTIVE_TASK.md` 授权提交指定。本审查只准入 **G1 的一对单卡 spot check**，不准入 10 对、批量测速或全量矩阵。

## 结论

**有条件通过 G1 代码审查。** 旧版预审的六项缺口中，G1 所需的资产哈希、真实 Git 双版本授权、输入隔离、同一次前向 CPU 对照、预算台账与不覆盖输出均已实现。批量 benchmark 与全量 pair 构建尚未完成，须另审代码和真实测量。

证据：

- `python -m py_compile experiments/src/idea01/cadmium_runner.py` 通过；`.cache/hf_venv/Scripts/python.exe -m pytest experiments/tests -q` 为 **217 passed**。测试包含临时真实 Git 仓库的授权/拒绝场景、输入字段拒绝、资产清单完整性、预算跨重启与原子输出。CPU 测试不证明 NPU 数值正确。
- 本地固定的 base 五文件、adapter 九文件 SHA 清单在 `experiments/configs/cadmium_asset_sha256_20260930.json`；对云脑运行副本 14 个文件逐一执行 `sha256sum`，均与该清单一致。远端路径分别含官方 pinned base revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`、adapter revision `6c6a79d160595207804171be56a037883b369abd`。
- G1 输入只含 `query_text,candidate_json_desc`。选择记录在 `experiments/configs/cadmium_m1b_pilot_selection_20260930.json`；输入文件 SHA 为 `501756a7b5ce0e2dc15c08ffabf37d0c8517b3fb4eef2740aff75e4dee812db7`。由冻结 M0 首条 query 与 pinned CADmium test parquet 首条候选构造，未读 qrels；CPU tokenization 全长 1018，小于训练上下文 4096。
- 生产 CLI 必须在 fresh `origin/master` 读到 `pilot_authorized`、完整批准 SHA、正预算，并确认远端 detached HEAD 等于批准 SHA且工作树干净，才载入权重。当前 `ACTIVE_TASK.md` 仍为 `implementation`、预算 0；本地真实调用已按设计在模型加载前拒绝。
- `--spot-check` 强制单对、batch 1、固定输入 SHA；一次真实前向的设备 FP32 gather 和已审 CPU 核心对同一 logits 比较，完整 token 分数留证；差异超 `rtol=1e-5, atol=1e-6` 即失败。实际 NPU 精度、模型载入与显存须由 G1 运行验证。

## Kimi 交接更正与运行限制

Kimi 的 `IMPLEMENTATION.md` 是其交稿时快照。其“spot-check 尚无 CLI”与旧命令中的 `--keep-token-scores` 已过时；Codex 后续加入 `--spot-check`、单对输入 SHA 固定、预算在模型加载前开始计时，以及输出的原子发布。以本审查和已提交代码为准，不覆盖 Kimi 原文。Kimi 所说的“可重入”只适用于预算和授权保护；当前评分 CLI 对输出采用新 run_id 不覆盖机制，尚不是全量断点续算器。

下一步须先提交/推送本代码，再由 Codex 单独提交 `ACTIVE_TASK.md`，设 `state=pilot_authorized`、`approved_sha=<本代码完整 SHA>`、`max_scored_pairs=1`、单卡短时 `gpu_budget`。远端仅 fetch、detached checkout 审核 SHA；pair 输入传到 checkout 外运行目录并复核 SHA。G1 结果需保存 stdout/stderr、输出 JSONL、summary、budget ledger、环境与 Git 状态。任一门失败不得放大对数。
