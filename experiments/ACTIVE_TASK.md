# 实验交接

## 当前任务 · 三张 CAD 检索对比表

- `task_id: idea01_retrieval_benchmark_fill_20261120`，`state: review_required`。首批已验收，CadTranslator–DeepCAD 和 Text2CAD L2 的五种方法已评价、核验并回填。原 256 件 E1 编码和评分已结案。[执行计划](plans/IDEA01_RETRIEVAL_BENCHMARK_FILL_2026-11-20.md)
- 分支 `codex/retrieval-benchmark-fill`。使用主仓已冻结的 8,032 件图库，同一批 7,614 个四级查询目标，另有 7,513 条 CadTranslator–DeepCAD 查询。冻结清单 SHA-256 为 `bdd3bc14d3b585bb7ccaf06100d2266d46e5ccea71450b87182212e0a24c4785`，L0 的五个方法共 38,070 个正例名次已从原矩阵逐条对账。
- 生成器保持官方权重、原有效字段掩码、FP32、完整 272 行和 batch size 16。预检实际退出 0，用 2,560 对及 62.18 卡秒核验，512 对历史分数逐位一致，同批原始 BERT 输出复用对预检分数也逐位一致。评分入口只读候选程序与去重原文，UID 对照和正例链接在远端评价时核对。[独立代码审查和预检](reviews/IDEA01_RETRIEVAL_BENCHMARK_FILL_CODE_REVIEW_2026-11-20.md)
- 前向实际使用 `6cd791312eefa5e575b0e56a39553fa9f92fbd3b`，历史授权 ref `fc71f871a8da2aa78bfcff3714ff3b5afa9d546d`。北京时间 11 月 20 日 19:39:46 启动，11 月 21 日 07:55:47 主进程实际退出 0。CT–DeepCAD 使用三卡产出 2,029 行，L2 使用四卡产出 7,604 行，共 77,372,256 对、162,808.99 卡秒。七个计费调用均已结算并退出 0。历史上限为 77,404,384 对和 224,000 卡秒，授权在 11 月 22 日 01:38:43 到期。
- 首批正式验收在 share4 完成，验收入口 `62378587f2195220312ecb314a5f42470b6ff9d0`，独立复算两条件两种生成分的全部登记查询名次。11 月 21 日 07:56:31 验收实际退出 0，`first_batch_accepted.json` SHA-256 为 `03c1731f0142a99adaeaa49e11675d0533c796de12b66835cc97ae07ed71443c`，绑定最终账本和评分退出原件。服务器证据在运行根的 `remote_acceptance_20261120`。
- 11 月 22 日在 share4 核对排名及原始来源指纹并生成三张表，D 盘 Markdown、LaTeX 和证据 JSON 已回填。L2 的 S₄₈ R@20 为 40.806%，S₀ 为 15.537%，配对增益 +25.269 个百分点，95% 区间 [+24.166, +26.411]。CT–DeepCAD 的 S₄₈ 和 S₀ 分别为 0.679% 与 0.719%，配对差 −0.040 个百分点，95% 区间 [−0.563, +0.395]。同件配对使用 7,122 个共同目标。远端缺 LaTeX 编译器，PDF 链接保留并标明 11 月 20 日版。交付指纹和报告原件在运行根的 `publication_20261122T233256/published`。
- 外部方法沿用 G5 固定权重与候选预处理。OpenShape 和 MM-CAD 复用经历史 L0 重放核对的既有候选向量。CLIP 用原 8,032 张渲染重编一次。Fusion 的最终图库取决于 HDF5 转码、几何和量化后排重。
- 2026-11-20 20:23（北京时间）已回填 CLIP、OpenShape、MM-CAD v2 的 CadTranslator–DeepCAD 和 L2 六行。独立排序复算全部 45,381 条登记正例名次。两套文本的 R@20 命中数分别为 196、360、224（7,513 条短句）和 54、87、56（7,614 条 L2）。短句未截断，L2 分别有 5,115、5,115、1 条截断。三种方法均无查询和候选编码失败。MM-CAD 的首次入参错误及修复后的真实退出均保留。
- 首批外部方法证据已归档至共享 OBS 的 `runs/m1c_retrieval_benchmark_first_external_20261120`。89 项、394,049,970 字节逐 SHA 及独立逐字节覆盖核验实际退出 0，清单 `184507f19e7ed5be8526847abf2ddb380591eb98360226c3bdeef30023416071`。4-A6000 仅作存储入口。
- L1、L3 的 CLIP、OpenShape、MM-CAD v2 已在 share4 编码结束，沿用原图库向量。全部六行的独立名次复算通过，已回填表 1。首个部署尝试因所指 Git 对象库缺少父提交而在启动前退出 1，第二次用已核对的 Duoduo 工作树做同一 bundle 的基底并实际成功，保留两份记录。L1/L3 的两种生成分待正式评分，旧小试窗口已经到期。
- 在三种外部方法均可完整读入两种描述的 2,335 个共同 UID 上，短句/L2 的 R@20 命中各为 CLIP 55/15、OpenShape 96/26、MM-CAD 67/19，图库仍为 8,032 件。结果文件 `evaluation/first_batch_external_full_read_panel.json` 绑定六个逐查询文件和共同 UID 清单，SHA-256 为 `4176ae38c0f4de0821adecef41e560be0d0112ff447ce9fa2ffb9ff659d40536`。
- 用户要求所有实验计算、复算和验收在远端执行。本机相关计算和等候脚本已停止，部分产物保留。首批验收与表格汇总均在 share4 完成。
- DuoduoCLIP 官方示例在 share4 CPU 实际退出 0。全图库八视图渲染亦实际退出 0，7,980/8,032 件有效，44 件 B-Rep 无效、7 件有面无三角网、1 件空渲染，均保留在图库。编码入口 `9fccd51841cee59f7c2c030f4e034aef2519b946` 只读评分轴。全量渲染包已备，待远端传输和正式评分。
- Fusion 的本机回放已停止并保留部分结果，最终图库和查询数量待远端核实后冻结。已测试的准备入口位于分支 `codex/retrieval-benchmark-fusion-eval`，预检和逐行引擎提交 `777a938f4404261e46673e19238a1b59afb8a3ef`。NPU 实测和正式评分按最终矩阵估算。
- 两种源监督训练共用的源端登记清单已就绪。157,215 个训练 UID 和 8,838 个验证 UID 至少有一级原文，均有官方程序。与 CadTranslator 短句的候选交集分别为 146,215、8,134 件。清单排除全部 8,046 个官方测试 UID，描述符资格和训练选点仍由正式训练入口核对。清单 SHA-256 为 `8a734cd2364a7c24a763a358bbe411c535edc7266be4c1566167456be889c132`。
- SpectraBRep 作者代码及 0.6B 基座已核验。四个官方程序的描述符与固定作者实现逐项相同，缓存实现为 `14a3757a742e36c808d73f5e24416ec4e4b56d00`。全量训练与验证缓存待远端执行，按实际覆盖冻结训练 UID 和参数。
- 本地运行根为 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_retrieval_benchmark_fill_20261120`。share4 使用干净 detached checkout、既有隔离环境和 checkout 外运行目录，全部 SSH 连接间隔至少 60 秒。原始分数、覆盖日志和汇总逐阶段留存。

## E1 补测 · 冻结 MM-CAD 基线

- `state: reviewed`。冻结 MM-CAD v2 在 E1-A/B/C 的完整补测已完成。A/B 共用原 E1 的 256 件 DeepCAD 图库，C 使用 256 件 Fusion 图库，512 件量化重放实体全部提取与编码成功。share4 CPU 编码、本地评价均实际退出 0。
- C 原生窗口下的 MM-CAD/S0/S48 R@20 为 62/256、115/256、252/256，R@1 为 8/256、33/256、207/256。两模型完整读取的 245 条原文上，R@20 为 59/245、115/245、241/245。A/B 的 DeepCAD 图库有 57 件与 MM-CAD:B 公共 train 名单共享原始 ABC UID，另附移除后的共同图库诊断。
- 运行标识 `idea01_e1_mmcad_20261120`，执行 SHA `17293e1c6bb7b840b40718203181abedf1343582`，分支 `codex/e1-mmcad-baseline`。已独立复算 196,096 个分数与 7,570 条正例名次。OBS 清单 `d153d32a080cebc6dedad8fd7cb91aad6763a2843b295aa6e166dd0a57992dbe`，43 项逐字节核回，当前 CPU 运行已结束。共同短描述和双人几何评审仍是后续条件。[补测计划](plans/IDEA01_E1_MMCAD_BASELINE_2026-11-20.md) · [结果报告](submissions/idea01_e1_fusion360_retrieval_20261120/E1_MMCAD_BASELINE_RESULTS_2026-11-20.md) · [审码记录](reviews/IDEA01_E1_MMCAD_CODE_REVIEW_2026-11-20.md) · [结果审计](reviews/IDEA01_E1_MMCAD_RESULT_AUDIT_2026-11-20.md)

## 实验一 · Fusion 360 跨来源检索

本地分支 `codex/e1-fusion360-protocol` 接续已结案的 Text2CAD 提精度实验。v1.1 的自动清点、几何核验、两域评分、正式评价和独立复算已完成。[执行文档](plans/IDEA01_E1_FUSION360_RETRIEVAL_2026-11-19.md) · [最终结果](submissions/idea01_e1_fusion360_retrieval_20261120/CODEX_E1_FUSION360_RESULTS_2026-11-20.md) · [48 条冻结背景索引](configs/idea01_e1_fusion360_background_48_20261119.json)

- E1 状态为 `reviewed`。正式前向授权已关闭，运行进程均已退出。E1d 有 48 条匿名查询和 321 个已通过读取核验的实体，双人语义盲评待标注。当前结果衡量登记 UID 找回。
- 本地运行标识 `idea01_e1_fusion360_20261120`。本地资产和输出使用 `CADRetrievalIdea1_runtime`。share4 使用独立 Git checkout、现有 Ascend 环境和外部运行目录，校验设备逻辑号、健康状态和进程占用后调度。
- 源验证 64 件中 62 件通过量化后的几何核验。公开训练序列 159,049 件逐件核对。第一层筛查的 5,555 条 DeepCAD 与 1,341 条 Fusion 专家文字经实际 BERT 分词器复核，token ID 无差异。两域几何作业实际退出 0。Fusion 529 件通过独立 STEP 核验，去重及留出后有 318 个可抽样组，两域按方案冻结为 256 件。
- 历史授权 ref 为 `9044ad292eb4baaf40687c3ab1c3e98b86aebc12`，实际实现 `cf959f3bc611bbbffe006525ba3843930a9e3755`，冻结清单 SHA `3df2a07cdd71317b9236c028f18725cd024f680cbf3f7337be0e5ed844e3f9e5`。两张空闲 Ascend 共完成 307,056 对、1,853.12 卡秒、1,047.32 秒控制器墙钟。G4 重放 256 对零差。B 编码总体未决，原生分保守区间跨过容限，校正分通过，冻结执行条件放行 C 全量评分。五个评分子进程、两个评价子进程和控制器实际退出均为 0。
- C 的 R@20 为 115/256 → 252/256，配对增益 +53.52 个百分点 [47.27, 59.77]，两者置换 Holm p 均为 0.00039992。校正后 R@1 为 207/256。字面数字规则命中 219/256，14 条隐去精确量值的合格改写使校正命中从 14 降至 9。结果适用于该筛选图库及程序派生描述。
- 401 件完整运行原件在本地及 share4 逐字节核验，清单 SHA `9ddcc49bea01a72c0e054d31f11c4950bc4964bd16ee1d6ead2bd90d6dc1f7d2`。共享 OBS 的 125 件评分和执行证据清单 SHA `20b3a9bfce099e821e13d488b8b407f3993004a5af79af69dd80c44e0603ce2b`，上传与独立复核实际退出 0。盲评材料的匿名目录和密封映射留在完整运行原件。
- 执行入口依次为 `prepare_text2cad_e1_inventory.py`、`prepare_text2cad_e1_geometry.py`、`prepare_text2cad_e1_freeze.py`、`run_text2cad_e1_score.py`、`eval_text2cad_e1.py`，均位于 `experiments/scripts`。

## 已完成 · Text2CAD 冻结模型检索提精度（2026-11-19）

- `state`: `reviewed`。八组方案共有 37 个验证配置，25 个预注册冻结读出完成留出测试。GLM‑5.3 审方案、复审代码和只读汇报，Codex 复核原始矩阵、拟合、名次和区间。
- 正式运行使用 `yunnao-share4` 四卡，源码 SHA `0a9e8606da2418cd1151c88cc275deff94c28e0c`。验证/测试 4,998,000/8,625,312 对，另有 60 对回放。两个控制器及各评分、评价端实际退出 0。当前两阶段均已结束。
- 冻结配置 SHA-256 `58ec5b53cf75021384cfbe6fbc070bc23b19c192880578bf494560f82980513f`。在完整 8,046 候选库、预注册 2,368 条测试查询上，`logmix_l1` R@20 为 5.194%，SUM 3.167%。配对差 +2.027 个百分点，95% CI [+0.821,+3.575]。MedR 3,046 → 602。字段加权代表低于对数混合背景校正。
- [实验计划](plans/IDEA01_TEXT2CAD_RETRIEVAL_ACCURACY_2026-11-18.md) · [验证结果](submissions/idea01_text2cad_accuracy_night_20261119/VALIDATION_RESULTS_2026-11-19.md) · [留出测试结果](submissions/idea01_text2cad_accuracy_night_20261119/TEST_RESULTS_2026-11-19.md) · [独立复算脚本](scripts/audit_text2cad_accuracy_night.py)。原始证据与阶段实退位于本地 `CADRetrievalIdea1_runtime/text2cad_accuracy_night_20261118`。

## 前一任务：G5 同协议对照——三外部方法全量评测完成，MM-CAD v2 复审通过（2026-11-18 收官，待终审）

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
