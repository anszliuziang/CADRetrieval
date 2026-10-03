# Research Output Manifest

> `/experiment-plan` 本轮计划产物索引。仓库根目录的 Git 白名单只同步 `CLAUDE.md` 和 `experiments/`，所以项目清单放在 `experiments/`。

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-09-30 | /experiment-plan | experiments/plans/IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md | implementation | 两卡 Ascend 910 的阶段门、预算、科学口径与自动交接路线 |
| 2026-09-30 | /experiment-plan | experiments/plans/IDEA01_TWO_910C_EXECUTION_TRACKER_2026-09-30.md | implementation | 按阶段追踪待交付与审查状态 |
| 2026-09-30 | /experiment-plan | CLAUDE.md | implementation | 更新当前 Kimi K3 官方 API 和仅两卡 910 的执行约束 |
| 2026-09-30 | /run-experiment | experiments/plans/IDEA01_M1B_CADMIUM_NPU_BENCH_2026-09-30.md | implementation | 用户授权验证测速后写入 M1b 本地实现合同与分门预算 |
| 2026-09-30 | /run-experiment | experiments/ACTIVE_TASK.md | implementation | 激活 M1b 本地实现，NPU 预算仍为零 |
| 2026-09-30 | /run-experiment | experiments/reviews/IDEA01_M1B_CODEX_PRE_REVIEW_2026-09-30.md | implementation | CADmium runner 草稿代码审核与 NPU 准入缺口 |
| 2026-09-30 | /run-experiment | experiments/configs/cadmium_asset_sha256_20260930.json | implementation | 本地与云脑 14 个模型实际加载文件的 SHA256 清单 |
| 2026-09-30 | /run-experiment | experiments/configs/cadmium_m1b_pilot_selection_20260930.json | implementation | 不含文本的固定单对 pilot 来源与 SHA256 |
| 2026-09-30 | /run-experiment | experiments/reviews/IDEA01_M1B_G1_INVALIDATION_2026-09-30.md | implementation | 首次 G1 发现 adapter 缺键与预算路径漏洞，原始输出作废、撤销授权并定义复核条件 |
| 2026-09-30 | /run-experiment | experiments/assets/CADMIUM_LORA_CHECKPOINT_AUDIT_2026-09-30.md | reviewed | 官方四种尺寸 adapter SHA 与逐张量核验：LoRA-B 全零，CADmium 臂暂停 |
| 2026-09-30 | /run-experiment | experiments/scripts/audit_lora_safetensors.py | reviewed | 复现 adapter SHA 与 LoRA-B 非零张量数的只读脚本 |
| 2026-09-30 | /experiment-plan | experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md | implementation | 用户选择原版 Text2CAD 后的双头评分、910 移植及七天前验证合同 |
| 2026-09-30 | /run-experiment | experiments/ACTIVE_TASK.md | implementation | 激活 Text2CAD M1c 本地实现；真实模型前向预算仍为零 |
| 2026-09-30 | /run-experiment | experiments/configs/text2cad_original_asset_sha256_20260930.json | implementation | 原版 checkpoint、BERT 与 CAD 向量 ZIP 的本地及云脑 SHA 清单 |
| 2026-09-30 | /run-experiment | experiments/scripts/inspect_text2cad_checkpoint.py | implementation | 不做模型前向的权重结构与单 CAD 向量检查脚本 |
| 2026-09-30 | /run-experiment | experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md | implementation | 原版 checkpoint、BERT、cad_seq 云脑副本 7 项 SHA 与零前向结构证据 |
| 2026-10-02 | /analyze-results | experiments/reviews/IDEA01_M1C_G3_CODEX_RESULTS_REVIEW_2026-10-02.md | review | 804600完整原始行、3219147独立检查、130本地与OBS SHA，G3准入通过 |
| 2026-10-02 | /experiment-plan | experiments/plans/IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md | implementation | 全29319624精确评分、固定七天预算、最小本地启动控制与独立评价合同 |
| 2026-10-02 | /experiment-plan | experiments/plans/IDEA01_M1C_EXECUTION_TRACKER_2026-10-02.md | implementation | 原版Text2CAD逐阶段证据与状态；实际授权仍由ACTIVE定义 |
| 2026-10-02 | /requesting-code-review | experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R1_2026-10-02.md | review_required | full契约/52+6 CPU通过，两生命周期Important真实CPU复现，拒绝正式G4 |
| 2026-10-02 | /systematic-debugging | experiments/plans/IDEA01_M1C_G4_R1_LIFECYCLE_FIX_2026-10-02.md | implementation | 仅新增启动控制的异常收尾/自有进程组修复与必要CPU证据，GPU0 |
| 2026-10-02 | /requesting-code-review | experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R2_2026-10-02.md | review_required | Root18/fresh23针对CPU；两个新反例deadline/末事件及Linux先观察后清理要求 |
| 2026-10-02 | /systematic-debugging | experiments/plans/IDEA01_M1C_G4_R2_COMPLETION_FIX_2026-10-02.md | implementation | 新控制器共同deadline与完成状态最小修复、可靠POSIX测试，GPU0 |
| 2026-10-02 | /requesting-code-review | experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R3_CPU_APPROVAL_2026-10-02.md | reviewed | R2 Root24/fresh27通过，仅批准零NPU Linux见证，不授权正式评分 |
| 2026-10-02 | /requesting-code-review | experiments/plans/IDEA01_M1C_G4_POSIX_CPU_WITNESS_2026-10-02.md | implementation | Git部署56ac0f8、固定22项CPU与清理前原始观测；不改断言/源码/评分 |
| 2026-10-02 | /requesting-code-review | experiments/reviews/IDEA01_M1C_G4_CODEX_FULL_APPROVAL_2026-10-02.md | pilot_authorized | 真实Linux22/清理前状态及16+9本地OBS SHA通过，正式全量完整SHA准入 |
| 2026-10-02 | /experiment-plan | experiments/plans/IDEA01_M1C_G4_EXECUTION_2026-10-02.md | pilot_authorized | 全29319624对、仅两fresh/336卡时/共同七天，已审Git启动与交接合同 |
| 2026-10-02 | /monitor-experiment | experiments/reviews/IDEA01_M1C_G4_CODEX_LAUNCH_REVIEW_2026-10-02.md | pilot_authorized | 两卡正式全量启动；独立PID/预算/首16092行96593检查及9权威本地OBS SHA，full计时口径更正 |
| 2026-10-03 | /monitor-experiment | experiments/reviews/IDEA01_M1C_G4_CODEX_COMPLETION_PREFLIGHT_2026-10-03.md | review_required | 全29319624评分正常结束，3644header及17完成元数据通过，待全行数值和完整归档 |
| 2026-10-03 | /experiment-plan | experiments/plans/IDEA01_M1C_G4_RESULTS_COLLECTION_2026-10-03.md | review_required | 零NPU原始收集、稳定final ledger、三处完整SHA与独立矩阵审核后再CPU评价 |
| 2026-10-03 | /monitor-experiment | experiments/reviews/IDEA01_M1C_G4_ARCHIVE_INTERRUPTION_REVIEW_2026-10-03.md | review_required | 全行及源端/本地SHA已过，收集CLI/OBS运输中断，保留现场等待恢复讨论 |
| 2026-10-03 | /experiment-plan | experiments/plans/IDEA01_M1C_G4_ARCHIVE_RECOVERY_PROPOSAL_2026-10-03.md | pending_user_decision | 保留旧部分OBS目录、以固定原件新目录完整归档，零NPU；未授权实施 |
