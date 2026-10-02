# 当前实验任务：Text2CAD G4完整精确矩阵（运行中）

- `state`: `pilot_authorized`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_EXECUTION_2026-10-02.md`
- `phase`: 正式全量已于2026-10-02T13:11:51Z启动且交接审核通过；仅只读监控当前两invocation，不启动新评分。13:18Z两卡各10已发布chunk，首16092条/96593项独立核验通过；全量与指标未完成
- `latest_review`: `experiments/reviews/IDEA01_M1C_G4_CODEX_LAUNCH_REVIEW_2026-10-02.md`
- `approved_sha`: `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`
- `gpu_budget`: 336 卡小时（双卡合计1209600卡秒；共同604800秒墙钟，TERM/KILL宽限在预算内）
- `max_scored_pairs`: 29319624（完整3644×8046，不含额外验证/预热/重试/空查询）
- `max_download_gb`: 0（固定资产齐备，不装包/换环境）
- `run_id`: `idea01_text2cad_g4_full_20261002`
- `last_run_sha`: G4运行`56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；controller3682036、scorer3682124/3682194，invocations20261002T131158Z-c3812433/20261002T131159Z-6ce644de，共同deadline2026-10-09T13:11:51.702770Z。G3及Linux22项证据沿用
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；正式评价后续恢复7934原查询，评分/控制不得读UID/qrels/原CSV/旧分数或probe选择
- `model`: 原版Text2CAD固定checkpoint/BERT/cadzip及既有two_head_factorized_logprob_sum；FP32/full272/b16尾14/trim none/无缓存，旧七核心AST不变。不是纯MLE或CAD几何精确NLL
- `server`: 仅yunnan-share2两卡npu:0/1，checkout /home/ma-user/work/CADRetrieval；RT /home/ma-user/work/CADRetrieval_runtime；既有spec10c18c17环境warm reuse。启动前已核干净Git和origin/master授权，运行中仅只读detached本完整SHA，不fetch/detach；cvteam14只OBS，避/data/data_dev
- `required_output`: Kimi LAUNCH报告9dd46182/回填c9e499b6及独立启动审核已完成，9权威证据本地/OBS逐SHA通过。继续完整29319624矩阵/两summary/最终ledger/真实退出/实际预算与环境资产及全量SHA；审核后另CPU评价。full路径无G3逐query计时sidecar，mtime只供进度，真实总耗时核账本及退出
- `ownership`: Kimi启动CLI session46405已exit0，当前Codex只读监控，不重复派发启动、不改当前运行源码/config/env或Kimi旧报告。运行中远端checkout不fetch/detach/编辑，不重启/重复评分
- `stop_conditions`: 脏Git/未知旧run或ledger/资源冲突、API/连接/数值/数据/代码资产指纹/七天或实际预算失败；保存证据停下交流，不换模型、改阈值/预算、删旧产物或重试评分。正常已授权作业保持controller监管，不能因本地API中断误杀健康作业

正式全3644×8046=29319624对，shard0 [0,1822)、shard1 [1822,3644)，各1822chunk/14659812对/916466批。主score sum，mean只诊断；100-query/短名单不能代全量。控制器import前PYTHONDONTWRITEBYTECODE=1，全部cache/tmp/output在checkout外。所有源码/config/script/doc本地编辑Git同步；服务器仅干净批准SHA运行。
