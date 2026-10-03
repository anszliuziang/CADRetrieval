# 当前实验任务：Text2CAD G4完成后原始收集

- `state`: `review_required`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_RESULTS_COLLECTION_2026-10-03.md`
- `phase`: 全29319624对评分正常结束；全行117289444检查、源端/本地3666项SHA及11054连接检查通过。Kimi CLI和OBS运输已中断，OBS仅1129个chunk文件名，未完整核验；停止推进、等待用户讨论恢复方案。只读诊断，不派发Kimi/运输/评分或G5
- `latest_review`: `experiments/reviews/IDEA01_M1C_G4_ARCHIVE_INTERRUPTION_REVIEW_2026-10-03.md`
- `pending_proposal`: `experiments/plans/IDEA01_M1C_G4_ARCHIVE_RECOVERY_PROPOSAL_2026-10-03.md`（待用户决定，不是执行授权）
- `approved_sha`: `none`（当前零NPU；历史G4运行SHA保持原件不变）
- `gpu_budget`: 0 卡小时
- `max_scored_pairs`: 0
- `max_download_gb`: 0（不新增模型/数据；允许运输既有14.8587GB原始评分文件与运行证据）
- `run_id`: `idea01_text2cad_g4_full_20261002`
- `last_run_sha`: `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；历史授权ref67e89a20eb6fc3b35d27c3ba2363da4311d9113c，两invocation20261002T131158Z-c3812433/20261002T131159Z-6ce644de均exit0/finalized，controller status=ok/组empty，119885.30卡秒/60019.4233067秒墙钟
- `data`: 8046CAD/3644唯一strip文本完整矩阵；物理分离UID映射留到后续评价恢复7934原查询，本阶段不能读qrels调方法
- `model`: 既有two_head_factorized_logprob_sum/spec text2cad-vector-score-v1；FP32/full272/b16/无trim-cache，不能称纯MLE或CAD几何NLL；不做任何真实模型/BERT/NPU前向
- `server`: yunnan-share2只读原干净detached56ac0f8，不fetch/detach/改源码/装包；RT /home/ma-user/work/CADRetrieval_runtime，既有spec10c18c17。cvteam14仅OBS，避/data/data_dev
- `required_output`: 全3644chunks/two summaries/final ledger/controller与两实际exit/全日志/实际命令/环境与输入SHA；LF SHA256SUMS，云脑、本地experiments/runs/<run_id>及共享OBS完整逐SHA真实exit0；Kimi新TEXT2CAD_G4_FULL_RESULTS_2026-10-03.md与完整报告提交SHA。Codex全行审核后另激活CPU评价
- `ownership`: Kimi仅CPU收集新submission，exact stage/commit自己的报告，不push，不改Codex文档/src/config/tests。Codex审核原件；提交交接前不争用index，不重启/重复评分/续跑
- `stop_conditions`: API/连接/资源/原件或资产指纹/退出/数值/完整性失败保存证据停下交流，不换API/模型、不改阈值/预算、不删除覆盖未知产物、不重试评分

full无G3逐query计时sidecar/cold_start/peak字段；mtime仅进度，不补造测量。两summary是真实时点快照，card1合法peer pending保持原样；稳定final ledger另取副本。仅从本run归档，禁止父目录递归混入CPU fixtures。源码/config/script/doc始终本地Git，服务器不编辑源码；当前仅原始资产运输。
