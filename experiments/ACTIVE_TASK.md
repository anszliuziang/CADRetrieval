# 当前实验任务

- `state`: `reviewed`（P0 已完成并经 Codex 审核；当前没有获准运行的下一阶段）
- `task_id`: `idea01_p0_scoring_audit_20260924`
- `plan`: `experiments/plans/IDEA01_P0_IMPLEMENTATION_2026-09-24.md`
- `parent_plan`: `experiments/plans/IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`
- `execution_sha`: `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`（P0 原始 60 对实际运行）
- `reviewed_fix_sha`: `660e4228b338430bc1899f5e5bfe1069adcfd5e7`（仅修复记录/停点及 CPU 补证，已推送；没有重新跑模型）
- `approved_sha`: 无；P1/正式 pilot 尚未授权
- `server`: `ssh yunnan-share2`，远端 Git checkout `/home/ma-user/work/CADRetrieval`
- `gpu_budget`: 单张 Ascend 910 系列 NPU，累计最多 1 卡小时
- `max_scored_pairs`: 60；仅 validation，最多 20 个候选 × 正确/错配/空 query 三种条件
- `model`: `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，主精度 BF16
- `data`: `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`
- `assets`: 完整 OBS 路径、SHA 和 checkout 外的已校验运行副本见 `experiments/assets/IDEA01_ASSET_STATUS.md`
- `allowed_remote_writes`: `/home/ma-user/work/CADRetrieval_runtime/` 下的运行日志、缓存和结果；OBS 的本任务运行产物目录；**Git checkout 中不编辑源码、不写结果**
- `handoff`: `experiments/submissions/idea01_p0_scoring_audit_20260924/{IMPLEMENTATION.md,RESULTS.md,BLOCKERS.md}`，原始数据清单与逐对/逐 token 证据按计划保留
- `codex_review`: `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`
- `result_archive`: 本地 `experiments/runs/idea01_p0_20260924_r01/`（Git 忽略）；共享 OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_p0_20260924_r01/`，`MANIFEST_SHA256_v2.txt` 78/78 通过

P0 已收尾。当前只允许读取、复核和维护交接证据，不再运行评分。下一阶段必须由 Codex 另写并激活计划，明确审过的执行 SHA 与资源额度。**不得运行 P1 的 32×8–16 矩阵、256×256、小库/全库检索、R@K 或其他模型。** 遵守 `CLAUDE.md`：本地编辑项目代码，Git 同步，服务器只检出并运行。
