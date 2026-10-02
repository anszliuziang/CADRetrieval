# Text2CAD G4：完整精确检索矩阵与评价合同

2026-10-02。负责人Codex；本地Claude Code / 官方Kimi K3实现与运行。本合同具体化用户已授权的“G2/G3保守估计都在七天内，则完整运行”；自身不是NPU授权。G3完整原始结果审核及G4启动代码审核后，才由ACTIVE_TASK明确完整approved SHA、pilot_authorized与预算。

## 1. 问题、证据与边界

| 主张 | 本轮必要证据 | 不据此声称 |
| --- | --- | --- |
| C1：冻结原版Text2CAD的原生生成分数是否能排序登记配对CAD | 3644×8046稠密精确矩阵，恢复7934条原查询，登记UID R@1/5/10/20、MedR、覆盖与重复组分层 | 不把UID命中等同全部语义相关性；不与未恢复相同UID/gallery的IMMA发表数字直接并表 |
| C2：排名能否由候选长度/先验解释 | 同矩阵的预注册sum/mean诊断、CPU长度基线与文本重复上限；空查询校正另立NPU预算 | 本轮不证明几何边缘似然、纯MLE、因果机制或跨模型架构优势 |

数据是固定Text2CAD v1.0官方test衍生协议A：8046 CAD完整图库、7934非空L0原查询；只strip首尾空白，3644唯一文本仅减少相同查询的重复运算，评价逐条恢复7934原查询。既有M0的UID/训练交叉与缺失审计沿用。任何100-query或短名单均不替代本轮。

## 2. 前置门与执行顺序

1. G2已审核：两卡batch16、原始26254项/415个OBS哈希通过；保守137822.16767930926秒=38.283935466474794小时。
2. G3已真实完成804600对、两卡exit0/finalized；独立3219147项原始行检查及130项本地/OBS SHA通过。IDEA01_M1C_G3_CODEX_RESULTS_REVIEW_2026-10-02.md已通过准入；独立ETA103866.44651402211秒=28.851790698339475小时。
3. 门通过后，Kimi仅本地实现最小启动控制及CPU证据；不改评分器、资产、分数、worker、选择配置或环境。提交实现与完整SHA，Codex审源码、真实full路径及必要CPU拒绝路径。
4. 审核通过后另写ACTIVE、push；远端干净detached指定完整SHA，两卡fresh各一次正式全量。原G3运行目录/账本不复用。
5. 作业期间只读监控；CPU评价实现可在本地进行，remote运行中的checkout保持原SHA。完成后先矩阵/资产/账本/退出与归档审核，再评价及结果解释。

本轮不重复G1/G2/G3评分，不新做batch32/缓存/trim测试，不安装依赖。G1 batch32失败证据保留。

## 3. 固定评分与精确工作量

- 模型与资产：原版Text2CAD checkpoint、BERT五文件、cad_seq.zip及worker四文件逐SHA固定；源revision和完整路径同G3。权威清单为experiments/configs/text2cad_original_asset_sha256_20260930.json、text2cad_pair_manifest_expected_20260930.json、text2cad_checkpoint_expected_keys_20260930.json；不允许另造清单。
- 唯一主score：two_head_factorized_logprob_sum，spec text2cad-vector-score-v1；原始logits FP32 logsoftmax/现有双精度累加、已审shift与逐头mask/EOS，不用训练loss代替。训练含label_smoothing=0.1，不能称纯MLE/NLL或CAD几何概率。
- 两卡npu:0/shard0与npu:1/shard1；n_shards=2、cards_used=1、batch16/尾14、texts_per_chunk=1、完整272行、FP32、trim none、无文本缓存。
- **不用--probe-selection或--token-audit，不用--resume，不筛选/重排候选或查询。** 固定full路径遍历全部worker。
- shard0文本[0,1822)，shard1[1822,3644)；每片1822个chunk、14659812对、916466批；总3644 chunk、29319624对、1832932批。
- 每个chunk原子发布且携完整绑定和8046行。每行保存六项sum/mean分数与有效长度；CPU评价只消费审核后的原始产物。

## 4. 资源和停止条件

唯一计算ssh yunnan-share2的两张910C；checkout /home/ma-user/work/CADRetrieval；RT /home/ma-user/work/CADRetrieval_runtime；PY /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python。环境spec10c18c17 warm reuse、offline及OMP/MKL/OPENBLAS=4，不遮蔽设备编号。cvteam14只提供共享OBS存储；不用/data/data_dev。

保守计划时长取max(G2,G3)=38.28394小时，已含至少1.30余量。**正式硬上限：自两卡作业共同起点七天墙钟604800秒，双卡合计336卡小时=1209600卡秒，评分总数精确29319624；只有两次fresh调用，不含额外预热/重试/空查询。** 这沿用用户七天门，不擅自另设更短中止线。控制器应使用共同deadline，记录每卡剩余timeout和真实结束状态；终止宽限须在共同deadline内计入，不能以晚启动第二卡增加上限。

共享最终账本$RT/budgets/text2cad_m1c_<approved_sha>.json；启动前不存在。run_id idea01_text2cad_g4_full_20261002；运行产物$RT/runs/<run_id>/card0、card1。预计原始JSONL约20–25GB；启动前确认runtime/local/OBS至少100GB可用。源和旧run不清理/覆盖。

实际错误、资源占用冲突、非有限/拒绝候选、API中断、数据或代码指纹不符、超预算、未知旧输出或悬挂账本时保存证据停下交流；不扩大预算、不换模型、不重新评分。失败后续是否resume另由Codex/用户评估，不自动启动第二次invocation。

## 5. 最小本地启动控制实现合同（零NPU）

沿用现有experiments/scripts/run_text2cad_score_matrix.py和FrozenMatrixScorer。新增控制脚本放experiments/scripts/，任务测试放experiments/tests/；所有源码本地编辑，经Git远端运行。不要在remote生成启动源码。本地实现报告TEXT2CAD_G4_IMPLEMENTATION_2026-10-02.md记录确切文件与完整实现SHA。

控制器可用Python标准库或shell；职责限于：

1. 校验完整SHA/干净detached HEAD、ACTIVE授权来源与固定task/budget/run范围；不得修改Git或ACTIVE。源码tree缓存留外部。固定批准参数，拒绝任意extra scorer args。
2. 对完整run与两卡目录、ledger做fresh检查；已存在输出不得覆盖，第二次控制器调用不得重复启动。先检查两卡健康/空闲，由Kimi提供只读preflight，不把其他人的进程终止。
3. 生成两条固定full CLI与safe环境；记录执行命令/控制器Git/共同start/deadline/每卡真实PID和stdout/stderr/exit。允许两卡错开短时间避免Git fetch锁，但使用同一deadline。
4. 后台持久执行，SSH断开不杀掉正常作业；每卡实际退出码独立落盘，不由管道尾命令掩盖。任何单卡失败必须明确记录并及时交接；剩余卡可保留正常已授权运行，由Codex读证据决定，不自动重启失败卡。不能仅将控制器spawn成功写成实验成功。
5. 必要CPU验证：固定argv覆盖全3644/8046、两片并集/无交集及对数批数；无probe/resume/cache/trim/qrels访问；dirty/wrong SHA/非授权/旧run或ledger拒绝；假子进程真实退出/超时与同一deadline；真实full CLI计划到producer绑定。仅CPU stub/临时目录，不能载模型/BERT或NPU，不重跑无关全套。

如果已有控制入口已经满足上述要求，优先复用，提交具体证据；无需为满足文件数量增加实现。审查后完整approved SHA由Codex填写，脚本不得猜测或动态选HEAD作为批准值。

## 6. 监控、原始交付与成功判据

Heartbeat每30分钟只读PID/NPU健康、两卡完整chunk数、账本进度/卡时、stderr增量、磁盘、远端Git。正常运行不重启、不fetch/detach、不改参数；状态不变且无用户行动保持安静。有实质失败/完成/需要决策才通知。

两卡各1822完整chunk、各exit0/账本finalized，实际pair计数全29319624、零拒绝，方可收集final ledger/two summaries。收集所有chunks/命令/控制器与启动退出/环境/源资产/全过程日志/失败记录；LF SHA256SUMS，云脑、本地experiments/runs/<run_id>/与cvteam14:/data/obs_data_liuza/CADRetrieval/runs/<run_id>/逐SHA核验。Codex独立检查精确稠密索引、六项有限score/有效长度及算术、资产/SHA/实际预算/退出，不靠报告代替原件。

完成矩阵审核后单独激活CPU评价：UID映射/qrels只允许评价入口；验证与worker的text/cad连接、7934条恢复、固定按score降序并cad_index升序打破tie。主表只主sum，mean诊断单列，R@K/MedR/成功覆盖与唯一文本/重复组大小分层及理论上限，固定bootstrap按文本组取样的原查询加权CI，随机期望/长度基线在同一图库独立报告。任何新增超参数不看test后调优。评价实现、指标脚本及结论同样由Codex审核。

## 7. 本轮之外

空查询/CPN、TF-IDF及几何代理仅在各自数据/科学前提明确后单独立合同和预算；不夹进29319624评分账本。完整矩阵负结果仍交付，不按probe或test成绩调整方法。其他生成模型/训练检索头/IMMA最终协议重建不阻塞当前原版Text2CAD全量测量。
