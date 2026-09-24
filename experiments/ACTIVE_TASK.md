# 当前实验任务

- `state`: `implementation`
- `task_id`: `idea01_p0_scoring_audit_20260924`
- `plan`: `experiments/plans/IDEA01_P0_IMPLEMENTATION_2026-09-24.md`
- `parent_plan`: `experiments/plans/IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`
- `approved_sha`: 无；仅允许本任务限定的 P0 实现与烟雾测试，正式 pilot 尚未审核
- `server`: `ssh yunnan-share2`，远端 Git checkout `/home/ma-user/work/CADRetrieval`
- `gpu_budget`: 单张 Ascend 910 系列 NPU，累计最多 1 卡小时
- `max_scored_pairs`: 60；仅 validation，最多 20 个候选 × 正确/错配/空 query 三种条件
- `model`: `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，主精度 BF16
- `data`: `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`
- `assets`: 完整 OBS 路径、SHA 和 checkout 外的已校验运行副本见 `experiments/assets/IDEA01_ASSET_STATUS.md`
- `allowed_remote_writes`: `/home/ma-user/work/CADRetrieval_runtime/` 下的运行日志、缓存和结果；OBS 的本任务运行产物目录；**Git checkout 中不编辑源码、不写结果**
- `handoff`: `experiments/submissions/idea01_p0_scoring_audit_20260924/{IMPLEMENTATION.md,RESULTS.md,BLOCKERS.md}`，原始数据清单与逐对/逐 token 证据按计划保留

按 `CLAUDE.md` 先在**本地**写代码、必要测试与报告，提交并推送 Git；服务器只 fetch、detach 到执行 SHA 后运行。可完成本任务明示的 P0 单样例及最多 60 对烟雾测试。达到预算、出现评分/哈希/边界异常或远端工作树变脏即停止并回填。**不得运行 P1 的 32×8–16 矩阵、256×256、小库/全库检索、R@K 或其他模型；这些必须等 Codex 审核本轮代码与原始结果后更新任务。**
