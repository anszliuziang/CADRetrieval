# 当前实验任务

- `state`: `review_required`（Kimi 已交付 P1 第一门，Codex 正在审核并要求补正；**NPU pilot 尚未授权**）
- `task_id`: `idea01_p1_conditional_sensitivity_20260925`
- `plan`: `experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md`
- `parent_plan`: `experiments/plans/IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`
- `prior_review`: `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`；P0 原始执行 SHA `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`，补正 SHA `660e4228b338430bc1899f5e5bfe1069adcfd5e7`
- `approved_sha`: 无；Kimi 第一版实现 `10c734451ee22e608eaf0d119521c092fc7b8f62` **未通过代码审核**，见 `experiments/reviews/IDEA01_P1_CODEX_CODE_REVIEW_2026-09-25.md`；补正并重新审核后方可填入完整 SHA、改为 `pilot_authorized`
- `server`: `ssh yunnan-share2`，Git checkout `/home/ma-user/work/CADRetrieval`；当前实现门仅可在远端做 CPU-only 数据/选择预核，不可占用 NPU
- `gpu_budget`: **0 卡小时**（当前门）
- `max_scored_pairs`: **0**（当前门）；计划中的 ≤416 对只是下一门的拟定上限，并非当前授权
- `model`: `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，未来主精度 BF16；当前只可读取 tokenizer/配置，不载入权重做模型前向
- `data`: `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，仅 validation 与既有 P0 审计清单
- `allowed_remote_writes`: 只允许 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/plans/idea01_p1_20260925/` 的 CPU 选择清单、日志和缓存；服务器源码树只 `git fetch`/`checkout --detach <SHA>`，禁止编辑或生成源码
- `handoff`: 本地 `experiments/submissions/idea01_p1_conditional_sensitivity_20260925/IMPLEMENTATION.md`，固定选择清单与计数摘要、代码 SHA、测试证据、候选/评分隔离说明、预计 NPU 成本；若受阻另写 `BLOCKERS.md`

先按 `CLAUDE.md` 在 Windows 本地实现与测试，提交并推送代码；CPU-only 选择预核可在本地缓存或远端**已 Git 检出的提交**上进行，所有远端输出写 checkout 外。**不得运行任何 P1 NPU 前向、416 对评分、R@K、P2/P3 或改模型/数据。** Kimi 交接实现后等待 Codex 审查；Codex 会明确写入 `pilot_authorized`、审核 SHA 和预算，才可以执行计划第二门。
