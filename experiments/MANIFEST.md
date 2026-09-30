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
