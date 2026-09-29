# 当前实验任务：CADmium 官方 checkpoint 失效，等待下一仪器选择

- `state`: `idle`（四种尺寸官方 CADmium adapter 的 LoRA-B 全零；暂停该臂及全部真实模型前向，等待下一仪器选择）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1B_CADMIUM_NPU_BENCH_2026-09-30.md`（本轮只实现 CADmium 真实模型 runner 和批量测速；其它臂后续单独激活）
- `prior_task`: `idea01_p1_conditional_sensitivity_20260925` 已 `reviewed`，见 `experiments/reviews/IDEA01_P1_CODEX_RESULT_REVIEW_2026-09-25.md`；历史 P1 批准 SHA 已失效
- `phase`: G1 固定单对 NPU 前向已运行，设备/CPU 数值一致，但旧 PEFT loader 未载入 adapter，该输出作废；后续严格载入核验确认官方 1.5B/3B/7B/14B 全部 LoRA-B 张量为零，不能代表微调后的生成模型。详见 `experiments/assets/CADMIUM_LORA_CHECKPOINT_AUDIT_2026-09-30.md`。预算路径漏洞已在本地修复，未重新授权任何前向
- `approved_sha`: 无；此前 `b6a46b2d4800fe5b12cda89c2404dbb9ec05205d` 的 G1 授权已撤销
- `gpu_budget`: 0 卡小时；`max_scored_pairs`: 0；不得再执行任何真实模型前向，直到选择新仪器、审查其代码与资产，并重新授权
- `data`: `SadilKhan/Text2CAD` 官方 v1.0 pinned revision `422d4c93198c3cd98d5c2092c0ab214b3a76a68d`；本地忽略目录 `.cache/text2cad_official/` 中选择性下载必要文件；已审 M0 清单在 `.cache/m0_real_run_v3/`，查询仅 `strip()` 首尾空白，图库 8,046、查询 7,934、qrels 7,934；只读使用既有 `ricemonster/IEEE-T-ASE` pinned revision `4091ce30ad85b774a9231484f9b40372dc0451d3` 与 P0 UID 清单做交叉审计
- `models`: 官方 Text2CAD v1.0 权重、CADmium-3B、已审 Text-to-CadQuery Qwen2.5-3B-SFT；HoLa/STEP-LLM 仅做静态可行性审计，不下载大权重
- `allowed_network`: 官方仓库及固定 revision 的只读资产元信息、小型索引/源码；大权重与数据本体由 Codex 当前选择性准备并核验。不要下载图像/深度图全库或其它大模型
- `server`: `yunnan-share2` 仅用于只读诊断和环境/资产检查；远端 checkout 不编辑。当前无 NPU 前向授权；cvteam14 仅用于共享 OBS 资产核对
- `required_output`: 复核并保存四份公开 adapter 的 SHA/逐张量零值证据；历史 G1 原始日志与结果标记无效留证。下一仪器另立子合同，不把旧 CADmium pilot 授权迁移过去
- `stop_conditions`: 官方资产 hash/UID 不一致、不可确认训练集、不可靠目标表示或 prompt、截断、qrels 进入评分器、许可不明、远端 Git 变脏；记录阻塞，不能改 benchmark 口径绕过

本轮特别核验：Codex 根据 P0 UID 清单只读计算到 Text-to-CadQuery 的 99,236 训练 UID 与官方 Text2CAD v1.0 test 交叉 **4,581**。Claude Code 实现者须独立复算；该 Qwen 权重不得在 8,046 项官方 test 上声称 held-out 检索成绩。它可以在预先确定的无交叉子集或其原生 held-out test 上参加另名协议，不能直接对照 IMMA 的 9.71%。

所有项目源码/计划/配置只在本地编辑、提交并推送 Git；服务器只检出审核 SHA 运行。`CLAUDE.md` 的模型输入隔离、资产清单、OBS、结果回填和代码审查规则继续有效。用户已授权**全量前的验证性测速**；后续测速仍须先审代码与完整 SHA，按批准的对数/卡时分门运行。经代表性实测与 30% 余量保守估算，全量能在双卡 7 天内完成且审核通过，才可另在本文件授权全量运行；否则先讨论。
