# 实验一完整性审计

2026-11-20。审查由未参与实现的独立 Codex 代理完成，属于同模型家族的暂定语义审查。原始输入、退出、文件完整性和数值另外做了确定性核验。总体结论为 **WARN**。登记实例检索的结果通过独立复算，范围限制来自筛选图库、专家描述对程序的依赖、原生分转换对照未决和语义标签待采集。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| A · 正例来源 | PASS | 256 条 B 和 256 条 C 描述逐条与固定公开 Parquet 一致。254 条 A 和 48 条背景与原始 Text2CAD CSV 一致。256 个官方 B 张量和 256 个 C 的最终 Autodesk STEP 源成员逐字节核对。C 的匿名实体来源是由最终 STEP 规范化得到的参考 B-Rep。训练张量精确重合排查覆盖 159,049 个公开训练 UID。[数据清点](../src/idea01/text2cad_e1_inventory.py#L258) · [源几何绑定](../scripts/prepare_text2cad_e1_geometry.py#L217) |
| B · 分数和统计 | PASS | 原生有效字段 log 分之和始终保存。S48 扣的是预先选定的 48 条源验证文字在候选列上的 log-mean-exp。独立的两种实现重算 2,044 个 A/B/B 官方/C 正例名次，Top-20 一致。B 保守区间、C 置换和 Holm、三臂配对抽样、替代列和改写排名均一致。[评分内核](../src/idea01/text2cad_vector_scorer.py#L390) · [读出和排名](../src/idea01/text2cad_e1_eval.py#L52) |
| C · 真实运行与工件 | PASS | 401 个运行原件及 406 个传输普通成员核回；307,056 个原始分片单元与合并矩阵全等。五个评分进程、两个 CPU 评价进程和控制器退出均为 0，合计 1,853.12 卡秒。OBS 独立复核 125 件。 |
| D · 实际评价路径 | PASS | 两次评价均执行冻结 CLI，实际记录的脚本与矩阵 SHA 一致。G4 重放 256 对精确一致。独立核查程序确认主评分和评价已完成。B 总体未决后，冻结控制器放行全量 C。门前另有已登记的 32 对目标域测速。[执行顺序](../scripts/run_text2cad_e1_score.py#L302) · [评价入口](../src/idea01/text2cad_e1_eval.py#L1225) |
| E · 适用范围 | WARN | C 是冻结模型对 256 个已通过筛查的项目、程序派生描述和已知正例的检索。无需模型的字面数字规则也命中 219/256。B 的原生编码对照保守区间为 [−2.79, +4.28] 个百分点。两版合格改写只有 10/48 和 14/48，七条两版均合格。A 有 208/254 行 L0 在公开训练文字中找到相同描述。[预注册样本与门](../plans/IDEA01_E1_FUSION360_RETRIEVAL_2026-11-19.md#L98) |
| F · 评价类型 | PASS | 实测是公开配对登记 UID 的真实正例评价。另有待双人人类标注的相关性端点，321 份匿名几何齐备、标注行数为零。[盲评包生成](../src/idea01/text2cad_e1_eval.py#L993) |

正式 C 的 115/256、252/256，两个 Holm p 值 2/5001，配对增益 +53.52 个百分点 [47.27, 59.77]，独立重算全等。附加核验涵盖字面数字、长度、按事前规则排除疑似近重合项目、两段背景、反向检索和 91 对严格匹配。单次建库并非独立来源重复实验。当前证据支持本协议下冻结 Text2CAD 的跨来源实例信号和背景校正收益。面向自然工程需求的几何相关性须完成两名标注者的独立评审。

逐项可复算证据位于 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_e1_fusion360_20261120/control/independent_integrity_audit_fnh6del0`，包含 `frozen_evidence.json`、`execution_evidence.json`、`score_evidence.json`、`diagnostic_evidence.json` 和只读核验脚本。输入与原件位置见[研究结果](../submissions/idea01_e1_fusion360_retrieval_20261120/CODEX_E1_FUSION360_RESULTS_2026-11-20.md)。
