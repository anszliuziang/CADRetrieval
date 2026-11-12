# Idea 01 / E2：空文本背景分数小试（机制诊断）——v2（2026-11-12 验收意见修订）

状态：**研究方向通过；本地实现 + CPU 预检进行中；正式 NPU 前向待用户批准启动**。授权模型（验收订正）：**GLM 负责实施，用户批准启动**。v1 的四点验收修正全部纳入本版（启动授权措辞、跨机分数可比门、单卡 E2 执行路径、总账预算）。

## 0. 授权与绑定（验收第 1 点）

- **授权模型**：GLM 负责实施；NPU 正式前向由**用户批准启动**（`ACTIVE_TASK.md` ownership 行已同步订正）。
- **可信授权远端**：允许集合 = {CNB `https://cnb.cool/lzacode/CADRetrieval`（share2 历史），GitHub 镜像 `https://github.com/anszliuziang/CADRetrieval`（share4）}；runner 校验 `origin` URL 属于允许集合（精确匹配 + 大小写/`.git` 归一）。
- **授权引用 ref**：runner 不再固定读 `origin/master`——E2 授权从**计划钉住的显式 ref** 读取：`origin/glm-night-20261112`（本分支已推送；后续所有 E2 代码提交都在该分支）。fetched ACTIVE_TASK 必须包含：`state`=`pilot_authorized`、**`e2_approved_sha`**（完整 40 hex == 运行 HEAD）、`e2_max_pairs`、`e2_max_card_seconds`、`e2_deadline_utc`。任一缺失/不符即拒绝。授权 ref 及其来源（远端 URL、fetched ref SHA、blob sha256）写入 environment 记录。
- **完整批准 SHA**：启动前由用户批准的执行提交 = 本分支上含全部 E2 代码与计划的提交；运行 HEAD 必须 detached 到该 SHA 且工作树干净。

## 1. 本轮只回答什么

同候选差分读出：给每个候选一次空文本背景分数 `S(∅,c)`，检验 `S(q,c) − S(∅,c)` 是否优于裸 SUM、与 v3（候选均值扣除）相比如何。**定位：已探索 test 图库上的机制诊断，非方法确认**；空文本可能偏离训练分布，B0/B1 敏度对照用于检验这一点。查询侧零新前向（复用 G4 已认证分数）。

**跨机可比前提（验收第 2 点，新增门）**：G4 条件分数来自 share2，背景分数将在 share4 生成。**差分读出之前必须先通过跨机复算门**：冻结少量历史 G4 非空文本与对应候选批次，在 share4 以原协议（FP32、完整 272、batch 16/尾批 14、无 trim/cache）复算，逐对与 G4 原始分数比较，容差**事先定义**：每对 `|Δscore_total| ≤ max(1e-3, 1e-6·|score|)` 且 `n_valid` 完全一致；任一对超差即停止，不做差分。通过后全量查询矩阵继续复用。

## 2. 冻结定义

**对象**：checkpoint `235b137e…`、BERT `6da4b6a2…`、图库 = `cad_seq.zip` 8,046 候选，**候选索引顺序与原 pair_manifest_v4 的 cad_store 完全一致**（构建时逐字节复制 + 运行时 SHA 对账）。

**背景清单（三份，均为独立 manifest 目录）**：
- **B0**：texts 恰 1 行 `text=""`（BERT 编码预期 `[CLS] [SEP]`，运行时以 tokenizer 证据固定 ids）；
- **B1**：texts 恰 1 行 `text="a CAD model."`（敏度；仅报告不选主）；
- **REPLAY（跨机复算）**：**预先注册**的 G4 历史非空文本 8 条（唯一文本 index = {0, 1, 2, 3, 1000, 2000, 3000, 3643}，取自 pair_manifest_v4 texts 表原文）× **候选集合 = G4 的前 127 个完整 16 批（原始 index 0..2031）∪ G4 真尾批（原始 index 8,032..8,045，14 条；8,046 = 502×16+14）**，dense 重排为 0..2,045，映射 `replay_dense_to_original_cad_index` 钉入指纹配置；批形状与 G4 历史批**逐批同构**（含真尾批 14，而非尺寸相同的另一批——复核 4 订正）= **16,368 对**。

**准入实现**：`load_texts` 通用合同不变（仍拒绝空串）；新增专用 `load_background_texts(dir, expected_text)`：恰 1 条记录、text 与期望逐字相等、`text_sha256`/`n_chars` 一致、text_index=0 连续；REPLAY 清单用通用 `load_texts`（全部非空）。三份清单的 fingerprints 在构建时写入**提交的** `configs/text2cad_e2_background_expected_20261112.json`，运行时严格对账。

**执行路径（验收第 3 点）**：新增**单卡 E2 专用入口** `run_text2cad_e2_background.py`（模块 `idea01/text2cad_e2_runner.py`），与现有 G4 双卡正式入口并存、互不改语义：
- E2 入口：单卡（`--device npu:X`，cards_used=1，n_shards=1）、FP32/无 trim/无 cache/**batch 16（尾批按实际）**、读 E2 三份清单、授权按 §0、E2 专用预算台账（per-invocation environment + ledger）；
- **dry-run（CPU、零前向）实际读取 E2 三份清单**：B0 验证空串准入与 tokenizer ids、B1 验证固定句、REPLAY 验证 8 条文本与候选子集、三者共用的 8,046 候选与索引顺序对账（cad_store SHA 与原 v4 相等）；**dry-run 与 score 两个入口的校验逻辑一致**（同一模块的同一组校验函数）。

**读出（冻结）**：R0=SUM（已有）；**R1=S(q,c)−S_B0(∅,c)**；R1b=S(q,c)−S_B1(∅,c)（敏度）；R2=v3（已有）。指标：R@1/5/10/20、MedR、单例（3,005 查询）/重复（4,929 查询，**对应 639 个文本组**——口径订正）分层、长度四分位（27/40/70）；配对文本组 bootstrap（3,644 组、5,000 次、seed 20261112）。**B0 与 B1 若结论方向或幅度差异明显，结论一律写成"结果对背景文本选择敏感"**。机制统计：corr(S_B0, col_effect)、corr(S_B0, length)（对照 −0.757）、S_B0 与候选均值差分布、R1 vs R2 排名 Spearman。float64 口径、tie 按 cad_index。

## 3. 总账预算（验收第 4 点）

| 项 | 对数 | 说明 |
|---|---:|---|
| B0 smoke | **1** | 1×1 对真实前向，**最先执行**，完成记录（台账 note + chunk 文件）是后续一切批量的前置门 |
| REPLAY 跨机复算 | **16,368** | share4 复算 G4 历史批（8×2,046，含尾批 14），本地与 G4 记录逐对比对过容差门后出 `replay_gate_record.json` |
| B0 背景 | 8,046 | **前置门：smoke 完成记录（真实 pairs_charged 字段 + environment/summary 证据核验）+ replay_gate_record(pass=true, 认证 G4 清单绑定, 8 chunk SHA 全验, 逐对完整唯一索引)** |
| B1 背景 | 8,046 | 同 B0 前置门，**另加 B0 自身成功记录（e2_b0_ok + 证据核验）** |
| **总计（启动前锁定）** | **32,461 对** | 1 + 16,368 + 8,046 + 8,046（精确数，无区间） |

- **累计上限 900 卡秒**（外推需求 ≈133 卡秒，余量 ~7×）；**共同墙钟截止 `e2_deadline_utc`** 的强制覆盖：scorer 每 batch heartbeat（DeadlineLedgerProxy）+ 每 chunk 前 + 模型加载后 + 结算判定，另配 G4 式**独立监督进程**（POSIX；独立解释器每 5 s 读共享台账，以「墙钟截止 −30 s 宽限」与「投影卡秒达上限 −30 s 宽限」**较早者**发 SIGTERM，再 30 s SIGKILL——宽限计入上限，卡死的加载/前向被按期杀死，台账留 pending、拒绝自动续跑）。
- **最终台账在所有证据核验与写入之后、以唯一退出判定一次性写入**：结算（投影卡秒/墙钟）→ environment+summary 落盘（失败即 `evidence_write_failed` 结算失败）→ 成功 note 仅在全链成功后写入 → finalize；过程退出与台账永远一致，不允许"程序 2 / 台账 0"。
- **G4 对照强制认证**：比较器先核验 `SHA256SUMS` 文件 SHA256 == 指纹配置钉住的认证值（`97f9cbb6…`），再逐块核对其余 8 个 chunk 的清单条目与磁盘文件，任一不符即拒绝比较；批量门同时核对 record 中的认证清单绑定与候选映射。
- 授权 caps 拒绝非有限值（inf/nan）；所有调用（smoke/批量/replay）共享同一 canonical E2 ledger；台账未收尾拒绝续跑。

## 4. 运行前置门（顺序）

1. 本地实现 + 单测：URL 允许集合、显式授权 ref（refspec 强制更新 + 从捕获提交读取，防陈旧 tracking ref）、`load_background_texts`、E2 清单构建器、E2 runner（dry-run/score 一致校验）、预算台账 caps（拒绝非有限值）、smoke/REPLAY 门前置、超限记失败、独立监督进程、证据后置统一落账、G4 认证清单强制。
2. **三条验收回归全部本机通过**：真实台账完整顺序（32,461 对 + 四成功 note）／真实台账证据写失败 → 下一阶段拒绝／G4 清单 pin 不符与 chunk 漂移 → REPLAY 拒绝；POSIX 卡死子进程按期退出回归已写就（本机 Windows skip，share4 Linux 预检执行）。
3. 独立 fresh review 通过 + 用户批准启动（写 `e2_approved_sha` 等 ACTIVE_TASK 字段）。
4. share4 CPU dry-run 预检（四清单 + 候选对账，零前向）+ POSIX 监督回归实际执行。
5. 执行顺序（强制门控）：**B0 smoke 1 对 → 证据审查 → REPLAY 16,368 对 → 本地逐对比对出 gate record（绑定认证 G4 清单）→ gate 记录放回 run root 并核 SHA → B0 8,046 → B1 8,046（需 B0 成功记录+证据核验）**；任何绕序尝试被 runner 拒绝。
6. 本地差分分析 → E2 结果报告。

## 4.1 第五轮复核修复（2026-11-12）

1. **独立监督进程**：`_supervise_loop`（POSIX fork）替代进程内 Python 定时器——独立解释器每 5 s 读共享台账，投影卡秒达「上限−30 s 宽限」或墙钟达「截止−30 s」即向评分进程发 SIGTERM，30 s 后 SIGKILL；宽限计入上限（KILL 不晚于截止）。评分进程被杀时台账留 pending。
2. **统一落账**：结算判定 → environment/summary 证据写入（失败即 `evidence_write_failed`）→ 成功 note → finalize（最后一步唯一写入）；下一阶段的完成记录门同时核验 note、pairs_charged 与证据文件存在。
3. **G4 认证强制**：`g4_sha256sums_sha256`（认证清单 `97f9cbb6…`）钉入指纹配置；比较前核验清单文件 SHA 与 8 个 chunk 的清单条目逐一相符；gate record 携带认证绑定，批量门复核。

三条验收回归（真实台账/真实门代码，仅替 forward）全部本机通过；POSIX 卡死子进程回归已在测试集中就绪（`test_posix_supervisor_kills_stuck_child_by_deadline`），将在 share4 Linux 预检执行。

## 5. 停止条件与产物

哈希失败、tokenizer 证据不符、replay 超容差、非有限分数、超 900 卡秒或墙钟截止、远端 Git 变脏、准入拒绝——任一即停、留证、非零。产物：`runs/idea01_text2cad_e2_background_20261112/`（checkout 外）→ 本地镜像 + SHA256SUMS → 共享 OBS；分析输出 `results.json` + 轻量 summary 进 `experiments/reviews/`；报告回填 `submissions/idea01_next_phase_20261112/E2_RESULTS_*.md`。

## 6. 与 E1/E3 衔接（不变）

E1：开发文本估计目标图库逐候选背景必须在同一批目标 CAD 上估计；按开发集 Recall 选方法则开发图库须含开发正例；最终评价用事前保留的查询与图库协议。E3：L1/L3 同库对照。
