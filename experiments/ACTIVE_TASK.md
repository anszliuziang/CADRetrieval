# 当前实验任务：Text2CAD G4完整精确矩阵（正式授权）

- `state`: `pilot_authorized`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_EXECUTION_2026-10-02.md`
- `phase`: G2/G3保守七天门、G4源码及实际Linux22项CPU见证已通过；允许固定两卡各一次fresh正式评分，先提交真实启动交接。无预热/重试/空查询，不生成远端启动源码
- `latest_review`: `experiments/reviews/IDEA01_M1C_G4_CODEX_FULL_APPROVAL_2026-10-02.md`
- `approved_sha`: `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`
- `gpu_budget`: 336 卡小时（双卡合计1209600卡秒；共同604800秒墙钟，TERM/KILL宽限在预算内）
- `max_scored_pairs`: 29319624（完整3644×8046，不含额外验证/预热/重试/空查询）
- `max_download_gb`: 0（固定资产齐备，不装包/换环境）
- `run_id`: `idea01_text2cad_g4_full_20261002`
- `last_run_sha`: G3 `487d02b9ac92f5e2338718e45b1f023c308dd332`；CPU见证已在56ac0f8干净detached完成22项、exit0/无torch，16原始+9独立记录本地和OBS逐SHA通过
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；正式评价后续恢复7934原查询，评分/控制不得读UID/qrels/原CSV/旧分数或probe选择
- `model`: 原版Text2CAD固定checkpoint/BERT/cadzip及既有two_head_factorized_logprob_sum；FP32/full272/b16尾14/trim none/无缓存，旧七核心AST不变。不是纯MLE或CAD几何精确NLL
- `server`: 仅yunnan-share2两卡npu:0/1，checkout /home/ma-user/work/CADRetrieval；RT /home/ma-user/work/CADRetrieval_runtime；既有spec10c18c17环境warm reuse。先干净Git/fetch核最新origin/master授权，运行只detached本完整SHA；cvteam14只OBS，避/data/data_dev
- `required_output`: 两次fresh的controller/两scorer真实PID与完整argv、共同start/deadline、ledger恰两invocation/cap29319624及1209600、两卡首chunk/timing/健康日志/干净Git；自己的新TEXT2CAD_G4_FULL_LAUNCH_2026-10-02.md及完整报告SHA。随后只读监控，完成矩阵/实际预算/退出及归档审核后另CPU评价
- `ownership`: 当前Kimi只按已审Git控制器启动及写自己的新LAUNCH报告，不改任何源码/config/env/Codex文档或旧报告；exact stage自己的新报告，commit及SHA回填后停止交接，不push。运行中远端checkout不fetch/detach/编辑，不重启/重复评分
- `stop_conditions`: 脏Git/未知旧run或ledger/资源冲突、API/连接/数值/数据/代码资产指纹/七天或实际预算失败；保存证据停下交流，不换模型、改阈值/预算、删旧产物或重试评分。正常已授权作业保持controller监管，不能因本地API中断误杀健康作业

正式全3644×8046=29319624对，shard0 [0,1822)、shard1 [1822,3644)，各1822chunk/14659812对/916466批。主score sum，mean只诊断；100-query/短名单不能代全量。控制器import前PYTHONDONTWRITEBYTECODE=1，全部cache/tmp/output在checkout外。所有源码/config/script/doc本地编辑Git同步；服务器仅干净批准SHA运行。
