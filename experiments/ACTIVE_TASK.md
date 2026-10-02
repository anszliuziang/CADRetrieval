# 当前实验任务：Text2CAD G3双卡100文本对完整图库持续验证

- `state`: `pilot_authorized`（仅下面指定fresh G3；完整矩阵尚未授权）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G3_EXECUTION_2026-10-02.md`（上位G3完整图库probe、R1/R2证据准入合同继续有效）
- `phase`: G3已按批准SHA真实启动，两卡后台运行中；launch提交33fffa97c3f644bba33e194045de22b02e75640e。2026-10-02 08:08Z Root只读确认NPU0/PID3294246、NPU1/PID3294325，完成26/50和25/50完整chunk/timing、总扣账417770，尚无exit/summary；Git干净、无新错误。仅监控，不重复调用评分/改checkout/重启/自动重试
- `latest_review`: `experiments/reviews/IDEA01_M1C_G3_CODEX_CODE_REVIEW_R3_APPROVAL_2026-10-02.md`
- `approved_sha`: `487d02b9ac92f5e2338718e45b1f023c308dd332`
- `last_run_sha`: `487d02b9ac92f5e2338718e45b1f023c308dd332`；run `idea01_text2cad_g3_probe_20261002`，invocations 20261002T075359Z-4037ac89/20261002T075359Z-09217532，start UTC07:53:51/54Z，运行中未完成。前置G2的08a0183双卡16200对/26254核验通过，99窗口最低0.992359、保守38.28394小时保留，仍须真实G3结果复核
- `gpu_budget`: 12（两卡合计卡小时上限43200卡秒，单次timeout21600秒；6小时墙钟上限，不复用G1/G2账本）
- `max_scored_pairs`: 804600（402300/片；每片50文本×8046 CAD、25150批，包含全部实际前向；无额外warmup）
- `run_id`: `idea01_text2cad_g3_probe_20261002`（runtime run/card0、card1；两次fresh scorer，不覆盖已有目录或重启评分）
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；评价恢复7934原查询。worker本地 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/`，token审计同runtime的 `derived/text2cad_text_token_lengths_20261001.json`。选择器及评分器不得读qrels/UID映射/原CSV/旧模型分数
- `model`: 原版Text2CAD固定checkpoint/BERT；唯一score `two_head_factorized_logprob_sum`，FP32、完整272行、batch16/尾14、无trim/无BERT cache。模型/score/shift/mask与G1/G2已审实现保持一致
- `server`: 计算只用 `yunnan-share2` 两卡npu:0/1；checkout `/home/ma-user/work/CADRetrieval`，runtime/既有环境在checkout外。只干净detached批准SHA运行，环境复用不装包。cvteam14仅共享OBS存储归档，不计计算资源；不用卡住的 `/data/data_dev`
- `required_output`: 固定选择SHA f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985；实际命令/PID/环境/源与资产指纹/退出状态，100完整chunks及timing/two summaries/final ledger/CPU ETA；本地+OBS原始结果SHA核验。先交TEXT2CAD_G3_LAUNCH_2026-10-02.md，完成后TEXT2CAD_G3_PROBE_RESULTS_2026-10-02.md，原报告不覆盖
- `ownership`: Kimi执行已审CLI、仅自己的新submission与Git忽略runs；不改源码/配置/CLAUDE/ACTIVE/计划/reviews/旧报告，不push。启动核验后提交launch交接并停止长等待；heartbeat继续。任何新源码改动须重新审核完整SHA。G3完成不自行启动G4
- `stop_conditions`: 科学口径/原资产不符、读取qrels、缩小或重排图库、预算/计时/分片不能核验、错指纹、API失败/额度/上下文中断；保存已完成证据并报告，不擅自换模型/降门槛

用户要求完整A的全查询×全图库精确评分。G2与G3保守ETA均≤604800秒且代码/结果审核通过，才自动准备正式全量G4；否则先讨论。100-query probe不替代正式全量。原版训练label_smoothing=0.1，双头分数不称CAD几何精确NLL。

所有源码/配置/脚本/文档只在本地编辑，经Git推送，服务器仅干净detached完整审核SHA运行；禁止远端编辑或SCP源码。结果缓存留checkout外，重要原始证据逐SHA归档共享OBS。
