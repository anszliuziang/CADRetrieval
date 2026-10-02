# 当前实验任务：Text2CAD G4 Linux CPU见证（零模型/NPU）

- `state`: `implementation`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_POSIX_CPU_WITNESS_2026-10-02.md`
- `phase`: R2代码e8ef0cd及报告ef12ba1/56ac0f8交接完成，session64259已exit0。Root24/fresh27针对CPU通过、两Important关闭。只授权指定Git SHA的Linux短CPU lifecycle见证，无正式controller/scorer/模型/BERT/NPU
- `latest_review`: `experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R3_CPU_APPROVAL_2026-10-02.md`
- `approved_sha`: `none`（正式G4评分未授权；CPU SHA另列，不得当NPU授权）
- `last_run_sha`: `487d02b9ac92f5e2338718e45b1f023c308dd332`；G3 run `idea01_text2cad_g3_probe_20261002`已结束，完整804600行/3219147独立检查/130本地及OBS SHA通过，真实G3 ETA28.8517907小时，G2保守38.2839355小时；正式估计取两者较大值
- `cpu_approved_sha`: `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；CPU墙钟上限600秒，源码e8ef0cd未变
- `gpu_budget`: 0（仅本地CPU实现与测试）
- `max_scored_pairs`: 0（禁止任何新模型/BERT/NPU前向）
- `max_download_gb`: 0（资产全部已固定，不装包/换环境）
- `run_id`: `m1c_g4_posix_cpu_20261002`（独立CPU证据；不创建正式G4产物/账本）
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；正式评价恢复7934原查询。worker本地 D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/。评分/启动控制不得读qrels/UID映射/原CSV/旧分数或使用probe选择；CPU测试使用临时目录/stub
- `model`: 现有原版Text2CAD固定checkpoint/BERT及已审原生score，FP32/full272/batch16尾14/无trim-cache；不改任何既有scorer/loader/math/ledger/worker/config/env
- `server`: yunnan-share2既有spec10c18c17环境；允许先记录干净Git/NPU，再fetch origin并detached CPU完整56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719；仅运行激活计划固定诊断程序，所有源码只Git部署，无安装/模型/NPU，cvteam14仅OBS
- `required_output`: 22项标准库targeted tests的真实exit/全日志、清理前实际组/descPID live-zombie记录、源SHA/命令/CPU环境/Git/NPU前后；完整证据本地及OBS逐SHA。失败先保留证据判根因，不自动重试
- `ownership`: 本轮只诊断/归档，不改源代码、配置或Kimi旧报告。Codex独立审核Linux原始证据后才更新正式pilot_authorized和完整SHA/固定预算
- `stop_conditions`: API失败/额度/上下文中断、必须改评分数学/环境/固定资产、无法确保共同deadline或full路径、未知旧文件、远端脏Git或资源冲突；保存证据停下交流，不擅自换模型、重试真实评分或扩大预算

未来正式合同：全3644×8046=29319624对，每片1822 chunk/14659812对/916466批；仅两次fresh scorer，无预热/重试/空查询，hard共同604800秒墙钟/336双卡合计卡小时。这里只给实现需求，不是当前NPU授权。完整SHA审核和ACTIVE正式授权前，真实评分额度始终0。

所有源码、配置、脚本、文档只本地编辑，经Git推送；远端仅干净detached批准完整SHA运行。重要原始证据留本地与OBS逐SHA核验，UID评价物理分离；100-query/短名单不能替代全量，不将双头label smoothing模型的分数称为CAD几何精确NLL。
