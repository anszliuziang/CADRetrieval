# 当前实验任务：Text2CAD G4 R2完成状态与截止收口（零前向）

- `state`: `implementation`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G4_R2_COMPLETION_FIX_2026-10-02.md`
- `phase`: R1代码226c410/报告6fd8407及回填0f66086已交接，session45579完成。原post-spawn故障已修，Root18及fresh23针对CPU通过；仍有后代信号晚于deadline和最后事件故障误报ok，正式G4不批准。仅本地最小R2及必要CPU证据，停review_required，无模型/BERT/NPU或远端checkout改动
- `latest_review`: `experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R2_2026-10-02.md`
- `approved_sha`: `none`（R1复审拒绝正式G4；模型/BERT/NPU预算0）
- `last_run_sha`: `487d02b9ac92f5e2338718e45b1f023c308dd332`；G3 run `idea01_text2cad_g3_probe_20261002`已结束，完整804600行/3219147独立检查/130本地及OBS SHA通过，真实G3 ETA28.8517907小时，G2保守38.2839355小时；正式估计取两者较大值
- `gpu_budget`: 0（仅本地CPU实现与测试）
- `max_scored_pairs`: 0（禁止任何新模型/BERT/NPU前向）
- `max_download_gb`: 0（资产全部已固定，不装包/换环境）
- `run_id`: `idea01_text2cad_g4_full_20261002`（未来正式run，当前不得创建远端运行产物或账本）
- `data`: 官方Text2CAD v1.0冻结worker四文件，8046 CAD/3644唯一strip文本；正式评价恢复7934原查询。worker本地 D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/。评分/启动控制不得读qrels/UID映射/原CSV/旧分数或使用probe选择；CPU测试使用临时目录/stub
- `model`: 现有原版Text2CAD固定checkpoint/BERT及已审原生score，FP32/full272/batch16尾14/无trim-cache；不改任何既有scorer/loader/math/ledger/worker/config/env
- `server`: 未来计算只用yunnan-share2两卡npu:0/1；checkout /home/ma-user/work/CADRetrieval，runtime /home/ma-user/work/CADRetrieval_runtime；既有环境spec10c18c17复用、不装包。当前仅必要只读preflight，不fetch/detach/执行真实评分；cvteam14仅OBS存储，不用/data/data_dev
- `required_output`: 后代组终止信号和宽限在原共同deadline内，末次controller_exit事件故障必须进入失败/一致JSON；实际组/后代活性断言先于测试finally清理、区分live与zombie。自己的新TEXT2CAD_G4_R2_IMPLEMENTATION_2026-10-02.md/完整SHA/必要CPU证据；审码后另授权Git部署零NPU Linux见证
- `ownership`: Kimi仅修改本轮新增g4控制模块/wrapper/test和自己的新R2 submission，不改旧scorer/config/env、Codex文档或旧报告；exact stage自己的文件，commit回填完整SHA后停review_required，不push、不占卡。正式完整SHA预算需Codex另授权
- `stop_conditions`: API失败/额度/上下文中断、必须改评分数学/环境/固定资产、无法确保共同deadline或full路径、未知旧文件、远端脏Git或资源冲突；保存证据停下交流，不擅自换模型、重试真实评分或扩大预算

未来正式合同：全3644×8046=29319624对，每片1822 chunk/14659812对/916466批；仅两次fresh scorer，无预热/重试/空查询，hard共同604800秒墙钟/336双卡合计卡小时。这里只给实现需求，不是当前NPU授权。完整SHA审核和ACTIVE正式授权前，真实评分额度始终0。

所有源码、配置、脚本、文档只本地编辑，经Git推送；远端仅干净detached批准完整SHA运行。重要原始证据留本地与OBS逐SHA核验，UID评价物理分离；100-query/短名单不能替代全量，不将双头label smoothing模型的分数称为CAD几何精确NLL。
