# 当前实验任务：Text2CAD G3完整图库稳态验证的本地实现

- `state`: `implementation`（只允许本地CPU实现/测试；NPU额度0）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md`
- `phase`: G2真实双卡batch16测速已审核通过；实现固定盲选100条文本、原全局索引对完整8046 CAD的probe score路径/计时/严格resume及CPU ETA。先实现与审码，不运行真实模型/BERT/任何NPU前向，不下载/安装依赖
- `latest_review`: `experiments/reviews/IDEA01_M1C_G2_B16_CODEX_RESULTS_REVIEW_2026-10-02.md`
- `approved_sha`: `none`（旧G2授权已结束，不能复用余额/旧调用）
- `last_run_sha`: `08a0183c86afc7198eed1c798d289d4b6f0c4a42`；run `idea01_text2cad_g2_b16_20261002`，两卡exit0/finalized，实际16200对。独立26254项通过/99组窗口最低0.992359；全量保守ETA38.28394小时，仍须G3复核
- `gpu_budget`: 0；`max_scored_pairs`: 0真实模型对数。只准本地小型假数据/假模型CPU测试与已有清单复算，不得真实前向或远端烟雾测试
- `prospective_run_id`: `idea01_text2cad_g3_probe_20261002`（待审核授权后总804600对、最多12卡小时/6小时墙钟，不是当前额度）
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；评价恢复7934原查询。worker本地 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/`，token审计同runtime的 `derived/text2cad_text_token_lengths_20261001.json`。选择器及评分器不得读qrels/UID映射/原CSV/旧模型分数
- `model`: 原版Text2CAD固定checkpoint/BERT；唯一score `two_head_factorized_logprob_sum`，FP32、完整272行、batch16/尾14、无trim/无BERT cache。模型/score/shift/mask与G1/G2已审实现保持一致
- `server`: 后续计算只用 `yunnan-share2` 两卡npu:0/1；checkout `/home/ma-user/work/CADRetrieval`，runtime/环境在checkout外。当前实现阶段不需SSH、不占卡。cvteam14仅共享OBS存储归档，不计计算资源；不用卡住的 `/data/data_dev`
- `required_output`: Git内LF固定 `text2cad_g3_probe_selection_20261002.json`、最小score/probe/resume/timing/CPU ETA实现、针对新增风险的CPU测试，及新submission `TEXT2CAD_G3_IMPLEMENTATION_2026-10-02.md`（改动/完整SHA/测试证据/准确两卡和CPU命令/已知限制）
- `ownership`: Kimi只写src/scripts/configs/tests/自己的新submission；不改CLAUDE/ACTIVE/计划/reviews/旧报告，不push。提交实现及SHA回填后停于review_required，Codex审核后另行push与授权
- `stop_conditions`: 科学口径/原资产不符、读取qrels、缩小或重排图库、预算/计时/分片不能核验、错指纹、API失败/额度/上下文中断；保存已完成证据并报告，不擅自换模型/降门槛

用户要求完整A的全查询×全图库精确评分。G2与G3保守ETA均≤604800秒且代码/结果审核通过，才自动准备正式全量G4；否则先讨论。100-query probe不替代正式全量。原版训练label_smoothing=0.1，双头分数不称CAD几何精确NLL。

所有源码/配置/脚本/文档只在本地编辑，经Git推送，服务器仅干净detached完整审核SHA运行；禁止远端编辑或SCP源码。结果缓存留checkout外，重要原始证据逐SHA归档共享OBS。
