# 当前实验任务：Text2CAD G4本地启动实现（零前向）

- `state`: `implementation`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md`
- `phase`: G3完整结果已审通过；仅本地实现最小双卡全量启动控制与必要CPU验证，提交代码和新IMPLEMENTATION报告后停review_required。不得启动任何真实模型/BERT/NPU，不远端生成启动源码或改变checkout
- `latest_review`: `experiments/reviews/IDEA01_M1C_G3_CODEX_RESULTS_REVIEW_2026-10-02.md`
- `approved_sha`: `none`（尚未审G4启动实现，无任何新前向授权）
- `last_run_sha`: `487d02b9ac92f5e2338718e45b1f023c308dd332`；G3 run `idea01_text2cad_g3_probe_20261002`已结束，完整804600行/3219147独立检查/130本地及OBS SHA通过，真实G3 ETA28.8517907小时，G2保守38.2839355小时；正式估计取两者较大值
- `gpu_budget`: 0（仅本地CPU实现与测试）
- `max_scored_pairs`: 0（禁止任何新模型/BERT/NPU前向）
- `max_download_gb`: 0（资产全部已固定，不装包/换环境）
- `run_id`: `idea01_text2cad_g4_full_20261002`（未来正式run，当前不得创建远端运行产物或账本）
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；正式评价恢复7934原查询。worker本地 D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/。评分/启动控制不得读qrels/UID映射/原CSV/旧分数或使用probe选择；CPU测试使用临时目录/stub
- `model`: 现有原版Text2CAD固定checkpoint/BERT及已审原生score，FP32/full272/batch16尾14/无trim-cache；不改任何既有scorer/loader/math/ledger/worker/config/env
- `server`: 未来计算只用yunnan-share2两卡npu:0/1；checkout /home/ma-user/work/CADRetrieval，runtime /home/ma-user/work/CADRetrieval_runtime；既有环境spec10c18c17复用、不装包。当前仅必要只读preflight，不fetch/detach/执行真实评分；cvteam14仅OBS存储，不用/data/data_dev
- `required_output`: 本地新增最小控制器与CPU测试、实测full路径与固定argv/精确两分片/共同deadline/真实退出与拒绝未知旧run或ledger证据；TEXT2CAD_G4_IMPLEMENTATION_2026-10-02.md，完整实现SHA/CPU命令退出/文件清单与待审命令；必要原始CPU证据本地runs镜像
- `ownership`: Kimi可写新增scripts/src模块/tests及自己的新submission，不改既有src/config/CLAUDE/ACTIVE/plans/reviews/旧报告；exact stage自己的文件、commit后停、不push、不占卡。Codex审码后才另写pilot_authorized/完整SHA和正式预算
- `stop_conditions`: API失败/额度/上下文中断、必须改评分数学/环境/固定资产、无法确保共同deadline或full路径、未知旧文件、远端脏Git或资源冲突；保存证据停下交流，不擅自换模型、重试真实评分或扩大预算

未来正式合同：全3644×8046=29319624对，每片1822 chunk/14659812对/916466批；仅两次fresh scorer，无预热/重试/空查询，hard共同604800秒墙钟/336双卡合计卡小时。这里只给实现需求，不是当前NPU授权。完整SHA审核和ACTIVE正式授权前，真实评分额度始终0。

所有源码、配置、脚本、文档只本地编辑，经Git推送；远端仅干净detached批准完整SHA运行。重要原始证据留本地与OBS逐SHA核验，UID评价物理分离；100-query/短名单不能替代全量，不将双头label smoothing模型的分数称为CAD几何精确NLL。
