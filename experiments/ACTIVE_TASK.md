# 当前任务 · 论文表 1–4 正式执行（2026-11-24）

- `state: in_progress`。以 Astra 的交接提交 `5616599a943de1fc85ea661e767d7c8c3a51eafc` 为准。[正式执行交接单](plans/IDEA01_FOUR_TABLES_HANDOFF_2026-11-24.md) · [表格底稿](plans/IDEA01_FOUR_TABLES_WORKSHEET_2026-11-24.md)
- 在 `yunnao-910a` 完成 T／CT 冻结轴、两套完整训练岭回归的 2,560 条验证名次回放、Fusion 原文／UID／项目清单及 40 对开发题的 320 展示、12 种方法逐项对照。正式题的几何准备、Fusion 全源转换和原生分成本核验分别留证。[执行记录](submissions/idea01_four_tables_20261124/EXECUTION_PROGRESS.md)
- 12 家族共 960 对的原始几何与盲核公开包已核验，240 对原始坐标干预齐全。待两位独立真人评阅者回执。Text2CAD 的有效几何在绑定、易题上各覆盖 410／480，坐标干预 0／240 达到预登记等价门槛。详情和现有辨认逐条工件见执行记录。
- `yunnao-910a` 命令／SpectraBRep 共有已核 170,361 个 DeepCAD 候选输入。CT 命令双塔 seed 17 单卡一整轮、T 冻结 BERT 成本探针按[dry-run 核验的配方](submissions/idea01_four_tables_20261124/COMMAND_TOWER_TRAINING.md)启动。表 1 潜在精确工作量 166,503,712 对。大规模生成器打分等待实测资源与整库预算。在机器间分别执行 SSH 连接启动间隔 30 秒。

## 40 对开发辨认的成品几何对照（2026-11-24）

- `state: reviewed`。MM-CAD v2、OpenShape 适配版及等轴测 CLIP 已补齐同一批 40 对题。固定 CAD 双正确为 8、14、2 /320，固定句双正确均为 2/320，八种展示全对均为 0/40。
- 加入成像诊断。CLIP 正俯视的 160 个孔开口中心全部可见，固定 CAD 双正确 14/320，固定句双正确 2/320。MM-CAD 的正四格差 202/320，形状与尺寸分层分别为 122/160、80/160。
- 全部实际编码使用 `yunnao-share4` CPU。实际退出、逐文件归档、四格分数、MM-CAD 曲面提取与俯视可见性均已核验。结果限两类开发几何题。[结果、版本和原件](submissions/idea01_dev_geometry_external_20261124/RESULTS.md) · [复核](submissions/idea01_dev_geometry_external_20261124/EXPERIMENT_AUDIT.md)

## 910A 准入和 40 对开发辨认（北京时间 2026-11-24）

- `state: reviewed`。Text2CAD 在 `yunnao-910a` 以保留 FP32 精度的配置通过历史单件回放、16/14 批形状复算、完整开发辨认及两个来源共六步训练更新。小型 HCCL 双卡见证通过，BF16 能力查询返回不支持。HierCAD 继续使用 `yunnao-share4` 的 BF16 环境。
- 两生成器各完成 320 个展示、1,280 个主分数。固定 CAD 时两件都选对描述的展示数为 Text2CAD 20/320、HierCAD 122/320；全部八种展示都答对分别为 0/40、2/40。两支双塔、槽位与局部坐标规则及表示恒等性对照随报告。40 对结果属于两个开发家族。
- 两来源的验证清单已备齐；L3 的 95/512 条超过 Text2CAD 的 512 token 窗口。对齐的生成器重训、验证面板评分、正式封存家族和自由生成对照保留待填。
- 本轮作业与复算均已结算。最终检查时 910A 八张卡处于健康、空闲状态。[本轮执行计划](plans/IDEA01_910A_AND_DEV_GEOMETRY_RECOGNITION_2026-11-23.md) · [运行、指标与原件](submissions/idea01_yunnao_910a_qualification_20261123/RUNTIME_AND_BINDING_RESULTS.md) · [复核](submissions/idea01_yunnao_910a_qualification_20261123/EXPERIMENT_AUDIT.md)

## 两来源同域检索与几何读出先导（2026-11-23）

- `state`：验证先导已执行并复核。八个检索矩阵、四个训练曝光账本、40 对几何题和两种生成器的已登记读出均有实际证据。Text2CAD L3 的 8 个配对超长。HierCAD 首轮跨批行分布超出容差，补做固定批行与掩码的真实后缀对照 10/10 全词表一致。
- `plan`：[先导交接单](plans/IDEA01_IN_DOMAIN_PILOT_HANDOFF_2026-11-23.md)与 Astra 所列的三类输入碰撞、完整前缀分布、直接文字–UID 正例检查。
- `run_root`：`yunnao-share4:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_in_domain_pilot_20261123`。本地进行编辑和 Git 管理，SSH 相邻启动至少间隔 60 秒。
- `scope`：两来源分别使用 4,096 训练 UID、512 验证查询和 2,048 验证候选。全训练集无序命令计数岭回归单列。测试侧仅封存清单和重合统计。四套正式权重训练、测试排名、正式几何题与自由生成另议。
- `report`：[同源先导结果](submissions/idea01_in_domain_pilot_20261123/PILOT_RESULTS.md)。HierCAD-14B 两套验证的长输入准入、原生分与 NLL 对账、21 个计时单元和 6 个实步均完成，原始前缀异常与对照分列。当前无待运行作业。正式评估规模与推理布局待研究决策。

## 既往任务记录

# 当前任务：G5 同协议对照——三外部方法全量评测完成，MM-CAD v2 复审通过（2026-11-18 收官，待终审）

- `state`: `results_submitted`（用户批准序列 CLIP → MM-CAD → OpenShape 全部完成；**MM-CAD v2 复审通过**（`g5_mmcad_v2_codex_review_20261118`：独立复算逐查询名次/R@K/MedR/5,000 次配对抽样全部一致；**修复增益配对 R@20 +0.619pp CI [+0.237, +1.013]，命中 69 → 118**）。公共子库 8,032 件 / 7,920 保留查询，配对 bootstrap seed 20261112 ×5,000。**四方法最终对照（R@20 / MedR）**：SUM 2.6136% / 3405.5；**OpenShape（本轮适配版）2.4369% / 1374.0**（R@20 未观测到明确差异，MedR 全场最优）；CLIP 1.2374% / 2057.5（显著更差）；**MM-CAD v2 1.4899% / 1735.0**（显著更差；旧 0.8712% 为混合实现**历史诊断作废**——`auto_model`/`model` 键失配 + 邻接静默失败，两修复合计配对增益 +0.619pp [+0.237, +1.013]，命中 69→118）。**分层（正例命令长度四分位 Top-20 命中）**：最短组 SUM 144 / OpenShape 51 / CLIP 37 / MM-CAD v2 33；最长组 SUM 6 / OpenShape 63 / CLIP 29 / MM-CAD v2 27——**SUM 头部命中优势集中于短命令组，三种外部配置在最长组都超过 SUM**。归因限制：四配置同时改变表示/模型/训练，排序不做归因。报告 `experiments/submissions/idea01_g5_external_baselines_20261115/`（CLIP/MMCAD/OPENSHAPE 三份 G5_RESULTS + 总表在 OPENSHAPE_G5_RESULTS §2-3 + ASSET_PRECHECK + READINESS）；本地镜像 `experiments/runs/idea01_{clip_g5_20261115, mmcad_g5_v2_20261118, openshape_g5_20261116}/`。执行 2026-11-16~18 全程 CPU（零 NPU），连接遵守 yunnao 安全策略。**遗留订正已完成**：计时口径（v2 编码 8,724.4s / 旧版 8,410.2s 分列）、MM-CAD v2 行标注（公开联合检查点严格加载 + G5 重建 STEP + 适配邻接提取，"恢复有效邻接，保留格式上限"）、ACTIVE_TASK 旧成绩同步。待 Codex 终审 → 收官；§4 分层归因分析需新授权）
- `g5_baselines_artifacts`: 首次回放（已验收原件）`experiments/runs/idea01_g5_sum_replay_20261115/`；返修后对账 `experiments/runs/idea01_g5_sum_replay_20261115_v2/`；报告 `experiments/submissions/idea01_g5_external_baselines_20261115/`（IMPLEMENTATION / RESULTS / ASSET_PRECHECK）；CLIP 探针产物 `runtime/clip_probe_20261115/`（32 UID 全链路）+ `runtime/clip_probe_smoke_20261115/`；环境规格 `experiments/configs/clip_probe_env_20261115.json`；回执 `runtime/g5_replay_20261115_exit_receipt.json`

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
