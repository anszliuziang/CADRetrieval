# 当前实验任务：G5 预注册 CPU 评价正式执行

- `state`: `pilot_authorized`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G5_CPU_EVALUATION_2026-10-03.md`
- `phase`: G5一次正式CPU评价15:07:01Z已启动，operator6612/child7240；仅读正常进度，完成真实exit后独立raw正例rank/指标审核及归档；无新评分
- `cpu_authorization_ref`: `4ad109025ea28557b47f8f467bfd8962882ccf34`
- `cpu_launch_report`: `experiments/submissions/idea01_text2cad_benchmark_20260928/CODEX_G5_CPU_LAUNCH_2026-10-03.md`
- `latest_review`: `experiments/reviews/IDEA01_M1C_G5_CODEX_CODE_APPROVAL_2026-10-03.md`
- `approved_diagnostic_plan`: `experiments/plans/IDEA01_M1C_G4_OBS_RECEIVER_DIAGNOSTIC_PROPOSAL_2026-10-03.md`
- `approved_proposal`: `experiments/plans/IDEA01_M1C_G4_ARCHIVE_RECOVERY_PROPOSAL_2026-10-03.md`（用户授权继续；执行者改为Codex）
- `approved_sha`: `e2d823aa284c800906ed37c0a2d0272e052de96b`
- `last_archive_sha`: `8bc4eeabea57f416982a286add9b65ba35f57ed1`
- `full_matrix_approved`: true
- `matrix_certificate`: `experiments/reviews/IDEA01_M1C_G4_MATRIX_CERTIFICATE_2026-10-03.json`
- `matrix_certificate_sha256`: `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6`
- `readonly_audit_witness_sha`: `43db1b90be46f346cdf6a9bddf83717a7b5d777814631e9d70172f1b654a1323`
- `archive_mode`: `full`
- `diagnostic_max_bytes`: 16777216（两个预定义条件各一次；当前约2MiB+清单）
- `gpu_budget`: 0 卡小时
- `max_scored_pairs`: 0
- `max_download_gb`: 0（不新增模型/数据；允许运输既有14.8587GB原始评分文件与运行证据）
- `run_id`: `idea01_text2cad_g5_cpu_20261003`
- `source_run_id`: `idea01_text2cad_g4_full_20261002`
- `cpu_evaluation_invocations`: 1（新独占输出目录；健康作业只读监控，不重复运行或调方法）
- `last_run_sha`: `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；历史授权ref67e89a20eb6fc3b35d27c3ba2363da4311d9113c，两invocation20261002T131158Z-c3812433/20261002T131159Z-6ce644de均exit0/finalized，controller status=ok/组empty，119885.30卡秒/60019.4233067秒墙钟
- `data`: 8046CAD/3644唯一strip文本完整矩阵；CPU评价恢复7934原查询，query/gallery/qrels/evalmap以G5预注册pins连接，不调方法
- `model`: 既有two_head_factorized_logprob_sum/spec text2cad-vector-score-v1；FP32/full272/b16/无trim-cache，不能称纯MLE或CAD几何NLL；不做任何真实模型/BERT/NPU前向
- `server`: yunnan-share2只读原干净detached56ac0f8，不fetch/detach/改源码/装包；RT /home/ma-user/work/CADRetrieval_runtime，既有spec10c18c17。cvteam6仅共享OBS存储入口（cvteam14 SSH banner超时），无GPU；避/data/data_dev
- `required_output`: G5独立CPU模块/CLI、固定toy证据/完整实现SHA与独立审码；批准后一次正式评价，全7934ranks/3644top20/固定基线/重复上限/5000文本组query加权CI/分层/真实命令退出与输入代码指纹；独立复算和本地OBS逐SHA报告。G4原件/失败现场与Kimi草稿保持
- `ownership`: 用户已授权Codex接手本地实现、运输、报告及CPU评价；保留Kimi草稿，以CODEX_新报告交接。新增代码仍独立审查。零NPU、不重启/重复评分/续跑
- `stop_conditions`: 工程故障保留真实证据，依据根因修复、审查后继续，无需重复许可；源资产/实验数值/研究协议或预算问题保留证据讨论，不换模型/放宽门/扩NPU预算/删除覆盖未知资产，不重试已完成评分

full无G3逐query计时sidecar/cold_start/peak字段；mtime仅进度，不补造测量。两summary是真实时点快照，card1合法peer pending保持原样；稳定final ledger另取副本。仅从本run归档，禁止父目录递归混入CPU fixtures。源码/config/script/doc始终本地Git，服务器不编辑源码；当前仅原始资产运输。
