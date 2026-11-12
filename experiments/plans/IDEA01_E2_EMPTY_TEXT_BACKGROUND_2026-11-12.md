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
- **REPLAY（跨机复算）**：**预先注册**的 G4 历史非空文本 8 条（唯一文本 index = {0, 1, 2, 3, 1000, 2000, 3000, 3643}，取自 pair_manifest_v4 texts 表原文）× 候选 index [0, 2,046)（与 G4 chunk 对齐；**2,046 = 127×16+14，与 G4 的 8,046 = 502×16+14 同构尾批 14**，覆盖尾批路径）= **16,368 对**。

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
| B0 背景 | 8,046 | **前置门：smoke 完成记录 + replay_gate_record(pass=true, SHA 对账)** |
| B1 背景 | 8,046 | 同 B0 前置门 |
| **总计（启动前锁定）** | **32,461 对** | 1 + 16,368 + 8,046 + 8,046（精确数，无区间） |

- **累计上限 900 卡秒**（外推需求 ≈133 卡秒，余量 ~7×）；**共同墙钟截止 `e2_deadline_utc`** 覆盖模型加载与进行中的评分（scorer 每 batch heartbeat 处强制），任一上限先到即停；**结算后复查累计卡秒，任何超支记为失败退出非零**（不允许"900.2 卡秒 + exit 0"）。
- 授权 caps 拒绝非有限值（inf/nan）；所有调用（smoke/批量/replay）共享同一 canonical E2 ledger；台账未收尾拒绝续跑。

## 4. 运行前置门（顺序）

1. 本地实现 + 单测：URL 允许集合、显式授权 ref（refspec 强制更新 + 从捕获提交读取，防陈旧 tracking ref）、`load_background_texts`、E2 清单构建器、E2 runner（dry-run/score 一致校验）、预算台账 caps（拒绝非有限值）、smoke/REPLAY 门前置、超限记失败。
2. 独立 fresh review 通过 + 用户批准启动（写 `e2_approved_sha` 等 ACTIVE_TASK 字段）。
3. share4 CPU dry-run 预检（四清单 + 候选对账，零前向）。
4. 执行顺序（强制门控）：**B0 smoke 1 对 → 证据审查 → REPLAY 16,368 对 → 本地逐对比对出 gate record → gate 记录放回 run root 并核 SHA → B0 8,046 → B1 8,046**；任何绕序尝试被 runner 拒绝。
5. 本地差分分析 → E2 结果报告。

## 5. 停止条件与产物

哈希失败、tokenizer 证据不符、replay 超容差、非有限分数、超 900 卡秒或墙钟截止、远端 Git 变脏、准入拒绝——任一即停、留证、非零。产物：`runs/idea01_text2cad_e2_background_20261112/`（checkout 外）→ 本地镜像 + SHA256SUMS → 共享 OBS；分析输出 `results.json` + 轻量 summary 进 `experiments/reviews/`；报告回填 `submissions/idea01_next_phase_20261112/E2_RESULTS_*.md`。

## 6. 与 E1/E3 衔接（不变）

E1：开发文本估计目标图库逐候选背景必须在同一批目标 CAD 上估计；按开发集 Recall 选方法则开发图库须含开发正例；最终评价用事前保留的查询与图库协议。E3：L1/L3 同库对照。
