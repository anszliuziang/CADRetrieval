# 当前任务：G5 同协议对照——统一评价入口 + 外部基线预检（2026-11-15 第一项交付完成，待审查）

- `state`: `results_submitted`（计划 `experiments/plans/IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md` §2：统一 CPU 评价入口 `experiments/src/text2cad_g5_eval_entry.py`（复用已审计 `text2cad_g5_cpu.py` 的 link_inputs/score_ranks/rank_metrics/bootstrap；身份/覆盖/数值门；子库模式；配对共享抽样 bootstrap；外部 JSONL 与矩阵双格式）+ 回放 CLI `experiments/scripts/run_g5_sum_replay_20261115.py`。**SUM 回放逐条对账 7,934/7,934 全等**（query_id/正例 UID/rank 三元组 vs 历史 G5 query_ranks.jsonl）；hit@1/5/10/20=16/73/130/207，R@K float64 全等历史 metrics.json，MedR=3411.5，长度切点 [27,40,70]；实际退出 0（403.8s 本地 CPU，29,319,624 行）。12 项针对性测试全过（重复文本/完全同分/行列错位/缺失与非有限/外部 JSONL 桩走同一入口/子库/配对 bootstrap）。资产预检：minimal_json 几何源头 8,046/8,046 覆盖实测（缺失 0），官方 CadSeqProc 提供 JSON→STEP/STL/点云转换，CLIP/OpenShape/MM-CAD 权重与依赖盘点见 `experiments/submissions/idea01_g5_external_baselines_20261115/ASSET_PRECHECK_2026-11-15.md`（全部未下载未安装，零前向零下载）。运行原件 `experiments/runs/idea01_g5_sum_replay_20261115/`（SHA256SUMS 6+3 项全 OK）。待 Codex 审码与对账复核 → reviewed；三方法实测需新授权：OCC 环境 + 权重下载 + 渲染脚本）

## 上一任务：E2 空文本背景小试（2026-11-15 Codex 终审通过、订正后 reviewed）

- `state`: `reviewed`（Codex 终审 2026-11-15：执行、归档、数值复核通过，verdict WARN→按订正清单收尾后 reviewed；审计 `CADRetrievalIdea1_runtime/e2_codex_final_review_20261115/EXPERIMENT_AUDIT.md` + `EXPERIMENT_AUDIT.json`（numerics all_passed=true：R0 7,934 查询逐条=G5、REPLAY 六分数字段零差、v3 复算逐条复现）。**2026-11-15 订正已全部执行**（`experiments/reviews/E2_CORRECTIONS_2026-11-15.md`）：①登记对照版本偏离——E2 分析实际计算的 R2 行是全测试文本候选均值（历史 v2，3.6047%），计划冻结的 v3（半样本）由审查者复算补入：R@20 **3.5669%** / MedR 669 / +0.958pp [+0.324, +1.618]，v3−v2 统计不可区分，全部结论不变；②末次结算订正 08:41:59Z；③归档普通文件合计订正 16,339,966 字节；④R1b 措辞"未观察到明确改善"；⑤双 SHA 登记：预检 `398dae63af5bea43601448e33d192ba0d505657a`（用户批准原文所针对版本）、实际执行 `71951046852302b411d45c60c5402ba1ab089e97`，AST 核验 `active_score_ast_unchanged_from_preflight=true`。**核心读出（订正后口径）**：R1(−S∅) R@20 1.4495% 显著低于 R0 2.6090%（CI 排除 0）；R1b 2.1553% 未观察到明确改善（区间跨零）；R1−R1b 差异 CI 排除 0 →"结果对背景文本选择敏感"；候选均值背景校正（v3 3.5669% / v2 3.6047%）显著高于 R0 且增益集中于单例组；S_B0 与候选均值 Pearson 0.999。探索性机制诊断，非方法确认。无待运行任务，零新增前向）
- `e2_execution`: 运行根 share4 `$RT/runs/idea01_text2cad_e2_background_20261112/`，本地镜像 `experiments/runs/idea01_text2cad_e2_background_20261112/`（11 chunk + 4×settlement/summary/environment + launch 日志 + ledger_final + gate record + analysis）；invocation `12f429c4`(smoke,1)/`e0b0f31e`(replay,16368)/`72421dc8`(b0,8046)/`f75fb4a1`(b1,8046)，结算 notes `e2_{smoke_b0,replay,b0,b1}_ok` 全部 settle_ok=true/breach=false/exit0，末次结算 2026-11-14T08:41:59Z
- `e2_execution`: 运行根 share4 `$RT/runs/idea01_text2cad_e2_background_20261112/`，本地镜像 `experiments/runs/idea01_text2cad_e2_background_20261112/`（11 chunk + 4×settlement/summary/environment + launch 日志 + ledger_final + gate record + analysis）；invocation `12f429c4`(smoke,1)/`e0b0f31e`(replay,16368)/`72421dc8`(b0,8046)/`f75fb4a1`(b1,8046)，结算 notes `e2_{smoke_b0,replay,b0,b1}_ok` 全部 settle_ok=true/breach=false/exit0
- `e2_approved_sha`: `71951046852302b411d45c60c5402ba1ab089e97`（运行 HEAD 必须 detached 到此完整 SHA 且工作树干净。**批准基线为用户预检的 `398dae63af5bea43601448e33d192ba0d505657a`**；其后差异全部披露：`495cbb7` 仅文档、`10b35d7` 删除从未被 Python 调用的重复 `start_deadline_supervision` 首定义（纯死代码删除，行为不变，POSIX/scorepath 回归针对现行定义）、`52990dd` 授权记录、`7195104` 授权分支 fetch 的 3 次有界重试（工程故障持续授权范围；失败仍 fail-closed，attempt 返回码入授权 meta）。评分语义、预算门、监督与执行顺序零改动）
- `ownership`: 用户 2026-11-11 指定 GLM 5.3 接手；**GLM 负责实施，NPU 正式前向由用户批准启动**（2026-11-12 二次验收订正）；计划先行、独立审码、SHA、预算纪律保留；评审按需参与
- `plan`: `experiments/plans/IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md`（v2，含验收四点修正）
- `acceptance`: 2026-11-12 验收（runtime/glm_night_acceptance_codex_20261112/EXPERIMENT_AUDIT.md）：E2 研究方向通过，进入本地实现与 CPU 预检；正式 NPU 前向待用户批准
- `e2_authorization`: 授权远端允许集合 = {CNB `https://cnb.cool/lzacode/CADRetrieval`, GitHub 镜像 `https://github.com/anszliuziang/CADRetrieval`}；授权 ref = `origin/glm-night-20261112`（runner 从 fetched ACTIVE_TASK 读 `state=pilot_authorized` + `e2_approved_sha`（==HEAD）+ `e2_max_pairs` + `e2_max_card_seconds` + `e2_deadline_utc`）；启动时本文件须已写入完整批准 SHA
- `e2_budget`: 总账精确 **32,461 对** = smoke 1 + REPLAY 16,368（8×2,046，尾批 14 覆盖）+ B0 8,046 + B1 8,046；**累计上限 900 卡秒**；**共同墙钟截止 `e2_deadline_utc` 于启动时写死**（覆盖模型加载与进行中评分；结算后复查，超支记失败退出非零）；先到即停非零
- `e2_prerequisites`: ①URL 允许集合 + 显式授权 ref + refspec 强制更新（防陈旧 tracking ref）②`load_background_texts` 专用准入 ③E2 单卡执行路径（dry-run/score 校验一致；四清单对账）④share4 CPU dry-run 预检 ⑤**强制执行顺序**：smoke → REPLAY → 容差门记录 → B0 → B1（runner 拒绝绕序）⑥detached HEAD 检查 ⑦caps 拒绝非有限值 ⑧结算后超支记失败
- `e2_positioning`: 已探索 test 图库上的机制诊断；非方法确认；B0/B1 差异明显时结论写作"结果对背景文本选择敏感"
- `night_tasks`: 用户 2026-11-12 授权三项夜间任务已完成并验收（share4 环境通过/失效分析接收/调研修订后通过路径以本轮修订为准）
- `gpu_budget`: 单张空闲 NPU；**累计上限 900 卡秒**（e2_max_card_seconds=900）
- `e2_max_pairs`: `32461`（smoke 1 + REPLAY 16,368 + B0 8,046 + B1 8,046，精确总数）
- `e2_max_card_seconds`: `900`（累计上限；先到即停非零）
- `e2_deadline_utc`: `2026-11-14T13:26:03Z`（共同墙钟窗口，自授权写入起 6 小时）
- `e2_run_root`: `$RT/runs/idea01_text2cad_e2_background_20261112/`（checkout 外；final_settlement 与预算台账为权威结算记录）
- `max_scored_pairs`: 0（E2 未启动；启动后精确 32,461 对新前向）
- `deliverables`: `experiments/submissions/idea01_next_phase_20261112/`（ENV_SETUP / TEXT2CAD_FAILURE_ANALYSIS / CAD_GENERATION_RETRIEVAL_BENCHMARK_SURVEY，均含 2026-11-12 验收订正）
- `share4_env`: `experiments/configs/idea01_yunnao_share4_env.json`（解释器 `/home/ma-user/anaconda3/envs/cadretrieval-glm-torch271/bin/python`，验收订正）；24/24 固定资产 SHA 复核通过（验收 witness）
- `failure_diag`: `experiments/scripts/analyze_text2cad_failure_factors_20261112.py`（执行记录已补全于失效分析 §4.1；全部 exploratory）
- `branch`: `glm-night-20261112`

## 以下为历史任务记录（已完成并 reviewed）

### Text2CAD 已有矩阵补充诊断（2026-10-04，Codex）

- `state`: `reviewed`
- `active_subplan`: `experiments/plans/IDEA01_TEXT2CAD_EXISTING_MATRIX_DIAGNOSTICS_2026-10-04.md`
- `phase`: 用户2026-10-04要求的已有矩阵分析已完成；一次本地actual exit0、全部15868正式positive ranks一致，5000共享置换及独立39548检查/60000null统计通过。报告边界审核完成，无待运行任务
- `approved_sha`: `none`
- `last_diagnostic_sha`: `00afcf73124edbb0056fc44cfc89d6755834efab`
- `diagnostic_authorization_ref`: `40639e4720c3524bf166f1223e633e35b7574fe2`
- `diagnostic_results`: `experiments/reviews/IDEA01_TEXT2CAD_MATRIX_DIAGNOSTICS_RESULTS_2026-10-04.json`
- `diagnostic_report`: `experiments/submissions/idea01_text2cad_benchmark_20260928/CODEX_TEXT2CAD_ALIGNMENT_DIAGNOSTICS_2026-10-04.md`
- `diagnostic_manifest_sha256`: `374d5df5406ec7ac3966adfdab0523a4703f568470a318b9e7ec0aa6f0be1ec7`（11payload；新增诊断仅本地，原G4/G5 OBS状态不变）
- `ownership`: 用户授权Codex实现/分析；不派Kimi。原正式G4/G5结果和证书保持，新增分析明确post-hoc exploratory

## 以下为已完成原全量研究的交付记录

- `state`: `reviewed`
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G5_CPU_EVALUATION_2026-10-03.md`
- `phase`: G4完整矩阵、G5正式评价及独立audit、主要原件和最终报告/trace证明包全部审核交付。最后62payload经备用共享OBS入口完整byte核验；closing receipt独占归档及SHA/byte通过。无待运行任务，不再评分/评价/raw audit/运输
- `archive_transport_authorized`: false（完整70payload已完成，不允许重复）
- `evaluation_approved`: true
- `evaluation_certificate`: `experiments/reviews/IDEA01_M1C_G5_EVALUATION_CERTIFICATE_2026-10-03.json`
- `evaluation_certificate_sha256`: `6421e331aa574a8337779982a39d812ddd340d94a8729deb5c533749684a097b`
- `g5_obs_manifest_sha256`: `0adec648178550648b36dfd6c372b9f1f3a1d08c3d6cfc373b0be9f446c09125`（70payload/24521948bytes；全部transport/SHA/byte/controller实际0）
- `finalproof_transport_authorized`: false（已完成；原读失败与备用挂载恢复证据分别保留，不允许重复运输）
- `closing_receipt_transport_authorized`: false（已完成一次独占普通系统资产上传，不允许重复）
- `approved_finalproof_sha`: `none`
- `last_finalproof_sha`: `078eb3f58be67ad894a72becb99f621d1d98e3c0`
- `finalproof_script_sha256`: `90073542265e5b804822e18c2173461a14d18c224a4e32df780c5e3497fde649`
- `finalproof_target`: `cvteam6:/data/obs_data_liuza/CADRetrieval/runs/m1c_g5_final_delivery_proof_20261003`（独占；本地runtime/g5_final_delivery_proof_bundle_20261003及control_20261003）
- `finalproof_launch`: 2026-10-03T16:11:17Z一次hidden controller PID40276，launcher runtime/g5_final_delivery_proof_launcher_20261003.json；授权refe0a32ba5d180727b96dc5145bc736f3302e90699
- `finalproof_manifest_sha256`: `b555cf47d3dad7ab465c616b2b19a3ca00d0735ac340c54dbf24c05c3bf21f1b`（62payload/108659bytes；本地全部SHA与报告/cert/原始trace绑定通过）
- `finalproof_progress`: 原8路双端actual0/remoteSHA0；cvteam6独立read后来ENOENT、原controller实际1保留。4-A6000同共享fuse.obsfs对象及manifest pins匹配，使用原a22d785只读witness一次恢复：reader/controller实际0、62payload/108659bytes/全部SHA和byte/无extra。未重复运输，不把原失败改称0；内部obsfs原因未确定
- `closing_receipt_target`: `4-A6000:/data/obs_data_liuza/CADRetrieval/runs/m1c_g5_final_closing_receipt_20261003`
- `closing_receipt_sha256`: `11dfa50b87ed01c19cfdea3fcba4e046091dc45d3f8a609a609fb1aef4e8be76`（73977bytes，包含88份原退出/命令/launcher/失败/manifest原字节记录及SHA）
- `closing_receipt_manifest_sha256`: `d8d307bcc47eaf6a39db51de0bf5fdb75beb145c4f667a1cecf0ed944d2c7640`（prepare/sender/receiver/remoteSHA/独立byte/controller全部实际0；1payload，无extra）
- `approved_archive_sha`: `none`
- `last_g5_archive_execution_sha`: `a6525797506ab7d4e9012bdb623d4c52931b8351`
- `archive_script_sha256`: `4210ebebef08ce7a5a253f1202925d78a84b49a4d06dc2edd895bb11dedfb1e0`
- `archive_recovery`: 首次仅本地准备因review目录中已知pytest_toy未列入proof清单而拒绝，actual exit1且未SSH/建远端目录/运输；失败现场保留。新代码只复制固定10份proof与manifest，显式排除且保留pytest_toy，其他extra仍拒绝。11针对CPU及唯一archive reviewer通过；授权新独占本地v2 bundle/control一次归档
- `archive_review`: `experiments/reviews/IDEA01_M1C_G5_ARCHIVE_APPROVAL_2026-10-03.md`
- `cpu_authorization_ref`: `4ad109025ea28557b47f8f467bfd8962882ccf34`
- `cpu_launch_report`: `experiments/submissions/idea01_text2cad_benchmark_20260928/CODEX_G5_CPU_LAUNCH_2026-10-03.md`
- `latest_review`: `experiments/reviews/IDEA01_M1C_TEXT2CAD_FINAL_DELIVERY_2026-10-03.md`
- `approved_diagnostic_plan`: `experiments/plans/IDEA01_M1C_G4_OBS_RECEIVER_DIAGNOSTIC_PROPOSAL_2026-10-03.md`
- `approved_proposal`: `experiments/plans/IDEA01_M1C_G4_ARCHIVE_RECOVERY_PROPOSAL_2026-10-03.md`（用户授权继续；执行者改为Codex）
- `approved_sha`: `none`（正式CPU评价已结束，无新增评价授权）
- `last_cpu_sha`: `e2d823aa284c800906ed37c0a2d0272e052de96b`
- `cpu_result_manifest_sha256`: `c98e205626a9d95d4830f3c628518f5f5e34f215c80b2079948dc0e54b19be95`（15payload/15660013bytes；实际child exit0/545.908秒）
- `independent_result_witness_sha256`: `007aa82bdc682a35e2b8c0e25bc2b153ca0f2a86786e4d9b9554578fcd6c3fc9`（只读独立审码通过；941toy方向/tie/median通过，无production import）
- `last_archive_sha`: `8bc4eeabea57f416982a286add9b65ba35f57ed1`
- `full_matrix_approved`: true
- `matrix_certificate`: `experiments/reviews/IDEA01_M1C_G4_MATRIX_CERTIFICATE_2026-10-03.json`
- `matrix_certificate_sha256`: `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6`
- `readonly_audit_witness_sha`: `43db1b90be46f346cdf6a9bddf83717a7b5d777814631e9d70172f1b654a1323`
- `archive_mode`: `closed`
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
- `server`: yunnan-share2保持原干净detached56ac0f8，不fetch/detach/改源码/装包；RT /home/ma-user/work/CADRetrieval_runtime，既有spec10c18c17。cvteam6与4-A6000仅共享OBS存储/只读核验入口，未使用其GPU；避/data/data_dev
- `required_output`: G5独立CPU模块/CLI、固定toy证据/完整实现SHA与独立审码；批准后一次正式评价，全7934ranks/3644top20/固定基线/重复上限/5000文本组query加权CI/分层/真实命令退出与输入代码指纹；独立复算和本地OBS逐SHA报告。G4原件/失败现场与Kimi草稿保持
- `ownership`: 用户已授权Codex接手本地实现、运输、报告及CPU评价；保留Kimi草稿，以CODEX_新报告交接。新增代码仍独立审查。零NPU、不重启/重复评分/续跑
- `stop_conditions`: 工程故障保留真实证据，依据根因修复、审查后继续，无需重复许可；源资产/实验数值/研究协议或预算问题保留证据讨论，不换模型/放宽门/扩NPU预算/删除覆盖未知资产，不重试已完成评分

full无G3逐query计时sidecar/cold_start/peak字段；mtime仅进度，不补造测量。两summary是真实时点快照，card1合法peer pending保持原样；稳定final ledger另取副本。仅从本run归档，禁止父目录递归混入CPU fixtures。源码/config/script/doc始终本地Git，服务器不编辑源码；当前仅原始资产运输。
