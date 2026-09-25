# 当前实验任务

- `state`: `pilot_authorized`（**仅阶段 1：1 对真实评分；累计 10 对与全量仍待 Codex 复核后另行授权**）
- `task_id`: `idea01_p1_conditional_sensitivity_20260925`
- `plan`: `experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md`
- `parent_plan`: `experiments/plans/IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`
- `prior_review`: `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`；P0 原始执行 SHA `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`，补正 SHA `660e4228b338430bc1899f5e5bfe1069adcfd5e7`
- `approved_sha`: `52ad27f2da5070e3f4a3ab6d66a11f79bd514b3c`（Kimi 补正后的完整交接提交；Codex 复核见 `experiments/reviews/IDEA01_P1_CODEX_CODE_REVIEW_2026-09-25.md`）
- `approved_selection_sha256`: `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964`（服务器 checkout 外权威 CPU 选择清单；32×8、414 对）
- `approved_asset_manifests`: 模型 `SHA256SUMS` 文件 SHA256 `98c1c44b9052218a79f48e2c5f7879b579031981048612c23304bfdc9fe2c785`；数据 `SHA256SUMS` 文件 SHA256 `f75a7a48c9c97fe6933f8b10dbe13e779da95fbeeab88bb1a612104a5f57fd34`；两份与共享 OBS 相同，评分前重验运行副本 11/11、6/6
- `server`: `ssh yunnan-share2`，Git checkout `/home/ma-user/work/CADRetrieval`；仅允许按已批准 SHA 在单张 NPU 上评分第 0 号配对，完成后停下交审
- `gpu_budget`: **单张 NPU、阶段 1 最多 0.25 卡小时和 10 分钟墙钟**；全程硬上限仍为 1 卡小时、45 分钟
- `max_scored_pairs`: **1**（仅 `--pair-ids "0"`，不得运行 `--resume`、其他配对或分析）；全程设计的 414 对不是当前授权
- `model`: `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，主精度 BF16；仅阶段 1 的一次 teacher-forced 前向获准
- `data`: `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，仅 validation 与既有 P0 审计清单
- `allowed_remote_writes`: 仅 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01/` 的阶段 1 原始评分/环境/日志，以及现有 plans 目录的只读选择清单；服务器源码树只 `git fetch`/`checkout --detach <approved_sha>`，禁止编辑或生成源码。`/data/data_dev` 挂载卡住期间不得访问；按 P1 计划的 2026-09-25 替代路径使用已核验的运行副本与共享 OBS
- `handoff`: 在本地 Git 忽略的 `experiments/runs/idea01_p1_20260925_r01/` 保存阶段 1 原始文件镜像；`experiments/submissions/idea01_p1_conditional_sensitivity_20260925/RESULTS.md` 回填实际 SHA、命令、1 对逐 token/环境/耗时/显存和失败记录，等待 Codex 审查

当前只可按批准 SHA 在 `yunnan-share2` 运行**第 0 号配对的一次 NPU 前向**。运行前核查 Git、选择清单与模型/数据清单 SHA；不得碰卡住的 910C OBS 挂载。完成后停下并交回原始证据，等待 Codex 审核；**不得自行继续累计 10 对或全部 414 对，不计算 R@K，不启动 P2/P3，不改模型/数据。**任何代码或配置变动使本批准 SHA 失效。
