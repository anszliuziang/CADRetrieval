# Idea 01 / E2：空文本背景分数小试（机制诊断）

状态：**计划提交，待 GLM 授权执行**（按 2026-11-11 接手治理与 2026-11-12 验收意见，新实验启动授权由 GLM 决定；本计划由 GLM 起草并自审，执行代码须另过独立 fresh review）。上位背景：失效分析 `experiments/submissions/idea01_next_phase_20261112/TEXT2CAD_FAILURE_ANALYSIS_2026-11-12.md`（验收意见：先 E2 机制诊断 → E1 事前保留评价集确认 → E3 描述具体度对照）。

## 1. 本轮只回答什么

最初 diffusion 类比的同候选差分读出在 CAD 自回归线上是否成立：**给每个候选 CAD 一次空文本背景分数 `S(∅,c)`，检验 `S(q,c) − S(∅,c)` 是否优于裸 SUM、与 v3（候选均值扣除）相比如何**。同时测量空文本背景与"候选平均条件分数背景"（v2/v3 的 col 效应）的接近程度。

**定位边界**：这是在**已探索过的 test 图库（8,046 CAD / 7,934 L0 查询）上的机制诊断**，不是方法确认。空文本可能偏离 Text2CAD 训练分布（训练时 instruction 从不为空），因此结果只用于机制解读；任何方法声明必须走 E1 的确认流程。查询侧**零新前向**（复用 G4 已认证的 S(q,c) 分数）。

## 2. 冻结定义（先于任何结果）

**对象（与 G4 完全一致，不换任何资产）**：checkpoint `235b137e…`、BERT `6da4b6a2…`、图库 = `cad_seq.zip` 8,046 候选 + `pair_manifest_v4` 冻结清单、协议 = FP32 / 完整 272 位置 / `two_head_factorized_logprob_sum`。

**背景文本定义（两个，预先注册；B0 主、B1 敏度，不据结果挑选）**：
- **B0（主）**：空串 `""`。编码 = 官方 `BertTokenizer` 对空串的分词结果（预期 `[CLS] [SEP]` 两个 token；运行时以 tokenizer 证据固定 ids 并写入环境记录）。**不改 `load_texts` 的通用校验**（其拒绝空串的合同保持）——新增专用加载函数 `load_background_texts`：严格要求恰 1 条记录、`text == ""`、`text_sha256 == sha256("")`、`n_chars == 0`，其余 schema 校验同 texts 表；新专用 reason code；任何偏离即拒绝。背景清单单独成目录（`background_manifest/`），sha256 入运行环境。
- **B1（敏度）**：`"a CAD model."`（最小内容、更接近训练分布的自然语句）。作用：检验 B0 的 off-distribution 效应方向；仅报告，不进入主结论。

**读出（全部冻结）**：
- R0 = 裸 SUM（已有 G4 分数）；
- **R1 = S(q,c) − S_B0(∅,c)**（主读出；同候选差分，即 PMI/DiffusionITM 式读出的自回归对应物）；
- R1b = S(q,c) − S_B1(∅,c)（敏度）；
- R2 = v3 半样本候选均值扣除（复用失效分析产物，作加性背景参照）。

**指标**：R@1/5/10/20、MedR、MeanR（7,934 查询等权）；单例（3,005）/重复（4,929）文本组分层；长度四分位（切点 27/40/70）；**配对文本组 bootstrap**（3,644 组、5,000 次、seed 20261112、共享抽样，报告 R1−R0 与 R1−R2 的 R@20/R@5 差值区间及 R@1 区间）。
**机制统计**：`corr(S_B0(c), col_effect(c))` 与 `corr(S_B0(c), length(c))`（对照 −0.757）；每候选 `S_B0(c) − mean_t S(t,c)` 的分布；R1 与 R2 排名的 Spearman。
**实现口径**：候选侧排名比较在 float64 下进行（失效分析的 float32 教训）；tie 按 cad_index 升序（与 G5 一致）。

## 3. 预算

全库新前向 **8,046（B0）+ 8,046（B1）= 16,092 对**。按 G4 实测速率（119,885.30 卡秒 / 29,319,624 对 ≈ 4.09 ms·卡/对）外推 ≈ **66 卡秒**；预算上限设 **0.25 卡时 / 单次墙钟 30 分钟 / 单张 NPU**，先到即停非零退出。share4 四卡空闲，使用任务指定单卡。

## 4. 运行前置门（share4 首次正式前向，按验收要求逐项）

1. **可信远端迁移审核**：`p1_runner.EXPECTED_ORIGIN_URL` 从单一 CNB 改为**显式允许集合**{CNB `https://cnb.cool/lzacode/CADRetrieval`，GitHub 镜像 `https://github.com/anszliuziang/CADRetrieval`}，其余授权门（state/approved_sha/approved_selection、fetch rc、problems 空）不变；本地提交 + 单测（含伪造 bare remote 的双 URL 用例与非法 URL 拒绝用例）+ 独立 fresh review。注意：Text2CAD frozen runner（`text2cad_frozen_runner.py`）同样引用该常量，一并生效。
2. **空文本准入代码**：`load_background_texts` + B0/B1 清单构造脚本 + 单测（空表/多条/非空文本/哈希不符全拒绝）；fresh review。
3. **share4 零 NPU 预检**：`run_text2cad_score_matrix.py --mode dry-run --scan-candidates --device cpu`（B2 已有模式）在新机通过，产出与 B2 同构证据。
4. **1 对真实 smoke**：新 SHA 上单卡 1 对真实前向（B0 背景 × 1 候选），审逐 token/有限值/显存/耗时后，再放行批量。
5. **评分输入构造**：E2 专用 pair manifest（1 个背景文本 × 8,046 候选 × 2 个背景定义），UID 映射与 qrels 不进入评分目录（沿用 v4 纪律）。

## 5. 执行与产物

- `run_id`: `idea01_text2cad_e2_background_20261112`；输出 `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_e2_background_20261112/`（checkout 外），台账延续 G4 风格（per-invocation environment + budget ledger）。
- 完成后：原始文件取回本地 Git 忽略镜像 `experiments/runs/<run_id>/`，不自引用 SHA256SUMS 双端核验，再按计划转存共享 OBS。
- 分析：本地 CPU 复算 R1/R1b/R2 对比与 bootstrap，输出 `results.json` + 轻量 summary 进 `experiments/reviews/`；报告回填 `experiments/submissions/idea01_next_phase_20261112/E2_RESULTS_*.md`（事实/解释/建议分节）。

## 6. 停止条件

哈希失败、tokenizer 证据与预期不符（B0 ids ≠ [CLS][SEP]）、非有限分数、超预算、远端 Git 变脏、准入代码拒绝合法输入——任一发生即停、留证、非零退出，不得放宽。结果若显示 B0 与 B1 结论方向相反且幅度大，只如实报告并降级机制解读（off-distribution 主导的信号）。

## 7. 与 E1/E3 的衔接（摘自验收意见，E1/E3 另立计划）

- **E1**：若用开发文本估计目标图库逐候选背景，必须在**同一批目标 CAD** 上估计；若按开发集 Recall 选方法，开发图库必须含开发正例；最终评价用事前保留的查询与图库协议。E2 的 B0/B1 若要进入确认流程，须在 E1 的开发协议下重新冻结并只保留一个背景定义。
- **E3**：L1/L3 同库对照，检验描述具体度对登记 UID 检索的影响（规模另行估算）。
