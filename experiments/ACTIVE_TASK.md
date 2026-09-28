# 当前实验任务：Text2CAD 检索基准 M0/M1 实现

- `state`: `implementation`（仅 CPU 数据/权重/协议审计与本地评分器实现；**新增 GPU/NPU 评分预算为 0**）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1_CADMIUM_SCORER_2026-09-28.md`（本轮只实现 CADmium 原生 JSON 评分；其它臂后续单独激活）
- `prior_task`: `idea01_p1_conditional_sensitivity_20260925` 已 `reviewed`，见 `experiments/reviews/IDEA01_P1_CODEX_RESULT_REVIEW_2026-09-25.md`；历史 P1 批准 SHA 已失效
- `phase`: M0 query/gallery/qrels 与精确 UID 交叉审计已由 Codex 审核（`b440f2d6447ac6b088ea0168ab1a84629810ffee`，见 `experiments/reviews/IDEA01_TEXT2CAD_M0_CODEX_REVIEW_2026-09-28.md`）；当前仅 M1a CADmium **本地**原生评分器代码、CPU 小例、模型训练配置核实和测试；M1b 其它臂、M2/M3 暂未激活
- `approved_sha`: 无；须 Claude Code（Kimi K3 或 GLM-5.2）提交实现和证据后由 Codex 审码并填写完整 SHA
- `gpu_budget`: 0 卡小时；`max_scored_pairs`: 0；不得启动任何新的模型前向评分、NPU/CUDA 烟测、正式检索或长作业
- `data`: `SadilKhan/Text2CAD` 官方 v1.0 pinned revision `422d4c93198c3cd98d5c2092c0ab214b3a76a68d`；本地忽略目录 `.cache/text2cad_official/` 中选择性下载必要文件；已审 M0 清单在 `.cache/m0_real_run_v3/`，查询仅 `strip()` 首尾空白，图库 8,046、查询 7,934、qrels 7,934；只读使用既有 `ricemonster/IEEE-T-ASE` pinned revision `4091ce30ad85b774a9231484f9b40372dc0451d3` 与 P0 UID 清单做交叉审计
- `models`: 官方 Text2CAD v1.0 权重、CADmium-3B、已审 Text-to-CadQuery Qwen2.5-3B-SFT；HoLa/STEP-LLM 仅做静态可行性审计，不下载大权重
- `allowed_network`: 官方仓库及固定 revision 的只读资产元信息、小型索引/源码；大权重与数据本体由 Codex 当前选择性准备并核验。不要下载图像/深度图全库或其它大模型
- `server`: 暂无新远端代码运行；已有 `ssh yunnan-share2` / `ssh cvteam14` 仅可只读资源/OBS 核验，远端 checkout 不编辑
- `required_output`: `experiments/submissions/idea01_text2cad_benchmark_20260928/IMPLEMENTATION.md`、CADmium 原生 JSON 评分定义与代码/CPU 测试、模型训练配置与许可证据、10 对吞吐测试设计；复用已审 M0 清单，不重建、改动或把 audit/qrels 给评分器读取；写明 `review_required` 后停止
- `stop_conditions`: 官方资产 hash/UID 不一致、不可确认训练集、不可靠目标表示或 prompt、截断、qrels 进入评分器、许可不明、远端 Git 变脏；记录阻塞，不能改 benchmark 口径绕过

本轮特别核验：Codex 根据 P0 UID 清单只读计算到 Text-to-CadQuery 的 99,236 训练 UID 与官方 Text2CAD v1.0 test 交叉 **4,581**。Claude Code 实现者须独立复算；该 Qwen 权重不得在 8,046 项官方 test 上声称 held-out 检索成绩。它可以在预先确定的无交叉子集或其原生 held-out test 上参加另名协议，不能直接对照 IMMA 的 9.71%。

所有项目源码/计划/配置只在本地编辑、提交并推送 Git；服务器只检出审核 SHA 运行。`CLAUDE.md` 的模型输入隔离、资产清单、OBS、结果回填和代码审查规则继续有效。当前用户已授权长作业自动监控，但只有 M0/M1 审核通过、另行写入具体 pilot 预算和完整 SHA 后才可启动。
