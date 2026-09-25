# 当前实验任务

- `state`: `reviewed`（P1 三阶段评分、固定分析与 Codex 原始证据复审已完成；**当前没有任何新 NPU 评分授权**）
- `task_id`: `idea01_p1_conditional_sensitivity_20260925`
- `plan`: `experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md`
- `parent_plan`: `experiments/plans/IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`
- `prior_review`: `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`；P0 原始执行 SHA `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`，补正 SHA `660e4228b338430bc1899f5e5bfe1069adcfd5e7`
- `approved_sha`: `52ad27f2da5070e3f4a3ab6d66a11f79bd514b3c`（**历史执行代码 SHA，当前授权已关闭**；Codex 复核见 `experiments/reviews/IDEA01_P1_CODEX_CODE_REVIEW_2026-09-25.md`）
- `approved_selection_sha256`: `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964`（服务器 checkout 外权威 CPU 选择清单；32×8、414 对）
- `approved_asset_manifests`: 模型 `SHA256SUMS` 文件 SHA256 `98c1c44b9052218a79f48e2c5f7879b579031981048612c23304bfdc9fe2c785`；数据 `SHA256SUMS` 文件 SHA256 `f75a7a48c9c97fe6933f8b10dbe13e779da95fbeeab88bb1a612104a5f57fd34`；两份与共享 OBS 相同，评分前重验运行副本 11/11、6/6
- `server`: `ssh yunnan-share2`，Git checkout `/home/ma-user/work/CADRetrieval`；**仅可只读审计既有 run，禁止新评分或重评分**
- `gpu_budget`: **当前新增预算 0**；历史三个阶段实际累计 414 次前向、217.19 秒、0.0603 卡小时，原硬上限为 416 次、45 分钟、1 卡小时
- `max_scored_pairs`: **既有 414 条唯一记录，当前不得新增**
- `model`: `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，历史评分精度 BF16
- `data`: `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，仅 validation 与既有 P0 审计清单
- `allowed_remote_writes`: 无新增；远端只读审计 checkout 外既有 `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01/`，服务器源码树保持干净。`/data/data_dev` 挂载卡住期间不得访问；原始产物已按替代路径取回并归档共享 OBS
- `handoff`: Kimi 的原始回填在 `experiments/submissions/idea01_p1_conditional_sensitivity_20260925/RESULTS.md`；Codex 原始结果、统计及解释复审在 `experiments/reviews/IDEA01_P1_CODEX_RESULT_REVIEW_2026-09-25.md`。原始镜像在本地 Git 忽略的 `.claude/worktrees/idea01-p1-impl/experiments/runs/idea01_p1_20260925_r01/` 与共享 OBS `stage3_full_mirror/`；事后同分敏感性分析单独归档为 `codex_p1_tie_review/`

P1 的 414 对评分、固定分析及 Codex 复审已完成。历史 `approved_sha` 和选择清单仅供溯源，`reviewed` 状态使 runner 不得再评分。P1 的通过范围只限固定验证集上的条件敏感性与 exact-UID 配对 sanity；独立语义 qrels、几何等价及设计家族泄漏尚未闭合。**不得启动 P2/P3；下一轮必须有新的计划、代码审查、完整批准 SHA 与 `pilot_authorized`。**
