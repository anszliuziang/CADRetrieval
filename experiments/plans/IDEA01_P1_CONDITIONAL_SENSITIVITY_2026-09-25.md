# Idea 01 / P1：冻结 CAD 程序生成器的条件敏感性小矩阵

状态：**计划草案；只有 `ACTIVE_TASK.md` 指向本文件并明确阶段后才执行。** 计划负责人 Codex；实现与运行者本地 Claude Code / Kimi K3；代码和结果由 Codex 审核。上位计划为 `IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`，P0 结论见 `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`。

## 1. 本轮只回答什么

P0 在 20 个固定 UID 上发现正确描述相对错配/空描述的正边际，但没有候选排序，也没有独立语义真值。P1 只测试：**同一个候选程序在其配对文本与其他 UID 文本下，原生条件序列 log-prob 是否稳定变化；在一个小型固定候选集合中，配对 UID 能否排在其他 UID 前。** 这是官方配对数据上的仪器检查，不能称为 CAD 语义理解或开放库检索性能。

模型、数据、官方提示模板、slow tokenizer、BF16 和 `response + EOS` 的 FP32 log-softmax sum 继承 P0 pinned 版本。`S(q,x)` 为完整原样程序加 EOS 的条件序列 log-prob；`S(∅,x)` 为同程序空 query 分数。`S(q,x)-S(∅,x)` 是已知的候选先验校正诊断，不作为新方法。mean 只是长度诊断，不称完整概率。

## 2. 固定数据与候选设计（先于任何模型分数）

只读验证集 `data_val.jsonl`，从 P0 的唯一 `uid` 映射与 token 覆盖审计出发。按 UID 排序后，定义合格池：唯一 UID、ZIP 程序与 JSONL `output` 逐字相同、前缀 token 边界正确、输出内无 EOS、`n_prefix_tokens <= 250`、`n_response_tokens <= 700`（含 EOS）。Codex 按 P0 原始覆盖表只读预核算为 **4,949/5,513（89.77%）**，是代码实现后须独立复算的参考值。记录合格池占验证集比例和与全集的长度差异；不按 P0/P1 分数筛行。若合格池不足 32 条则停止并报告。

用 `random.Random(20260925)` 从合格 UID 列表抽取 **32 个不同 query/配对程序**，保持抽样后的固定顺序。对第 `i` 个 query 构造 8 个不同候选：

1. 自身配对程序 1 个；
2. 从 1…31 中用同一 RNG 抽取 **3 个不重复循环偏移**，对所有 `i` 使用相同偏移，取 `(i + offset) mod 32` 的三个其他配对程序。这保证每个选中程序恰作为 3 个其他 query 的候选，允许同候选内比较正确/错配 query；
3. 从合格池中排除上述 32 个 UID，按程序 CadQuery 方法名集合的 Jaccard 相似度降序选 **2 个结构近邻**，同分依次按 response token 长度差升序、UID 升序。方法名由原样程序中 `\.name(` 的 `name` 提取，忽略 `export`/`exporters`；规则、正则和排除词写入选择清单；
4. 从剩余合格池按与正例的 response token 长度绝对差升序、UID 升序选 **2 个长度近邻**。

每个 query 8 个候选，总条件评分 **32×8=256 对**；对所有不同候选各算一次 `S(∅,x)`，至多 32+4×32=160 对。总前向 **≤416 对**，小于 512 上限。若任何 query 不能凑满 8 个不同候选，停止并报告，不临时放宽规则。先在 CPU 上将所有计划配对完整 tokenize，断言前缀边界、EOS 和 `len(input_ids)≤1024`；尽管长度上限按 250+700 给出余量，也要实测每一对，失败则停在 CPU 门。

选择清单必须存：源行号、UID、候选来源（self / selected-offset / method-neighbor / length-neighbor）、方法名集合/相似度、前缀与 response token 长度、seed、全部排除和回退情况。配对 UID 只供选择/分析层使用，评分函数仍只接收 `query` 与候选 `output`。不把其他 UID 自动视为语义负例：它们可能几何等价。

## 3. 预先确定的分析

**主诊断（同候选比较）**：对 32 个被选中配对程序，每个有自身正确 query 和三个其他 query。逐候选求 `S(own,x) - mean_j S(other_j,x)`，报告 32 个边际的均值/中位数、正值数与以候选 UID 为重抽样单位的 95% bootstrap CI（固定 seed、≥2,000 次）。同一候选的 response token 完全相同，这个边际不会由候选程序长度单独产生；它仍可能来自文本模板或训练数据的配对规律。

**次诊断（8 候选配对 UID sanity）**：对每个 query，按 `S(q,x)` 给 8 个候选排序，报告配对 UID 的名次、top-1 比例和对 7 个其他 UID 的逐对胜率；分别对 raw sum、mean 和固定的 `S(q,x)-S(∅,x)` 计算。以 query 为重抽样单位给 95% bootstrap CI。它们全部标为 **exact-UID 配对 sanity**，不是独立语义检索 R@K。

**必要对照与诊断**：随机候选排序（固定 seed）、查询无关 `S(∅,x)`、按 response token 长度从短到长的排序、无监督 query↔原样程序字符 3-gram TF-IDF cosine（IDF 只由本轮候选程序拟合，不用 qrels）在**同一 8 候选集合**上重算；报告各自配对 UID 名次/逐对胜率及耗时。分层列出正例长度、方法集合相似度、注释比例、导出样板、EOS logprob，尤其检查优势是否仅来自短程序或复制的数字/标识符。分数臂和基线不得看结果后挑选或改阈值。

对 32 个正例与其近邻另做分数无关的程序 token Jaccard 粗筛，给高相似对的 UID 和相似度清单；这是近重复提示，不是几何家族证明。不删除或改写难例。若发现 exact 相同程序/UID 或错误源映射，停止并修正数据准入。

## 4. 实现、审核、运行门

**第一门仅实现**：Kimi 在 Windows 本地写 P1 选择、评分调度、分析与必要测试，提交并推送 Git；用本地缓存或远端已检出的提交做 **CPU-only** 候选/配对草案和计划计数，远端输出仍须在 checkout 外，摘要写入 `experiments/submissions/idea01_p1_conditional_sensitivity_20260925/IMPLEMENTATION.md`。此门不占 NPU、不跑 416 对。Codex 检查选择与评分隔离、P0 修复后完整 input ID/环境记录、预算上限、EOS/shift 与基线实现。只有 Codex 在 `ACTIVE_TASK.md` 写入 `pilot_authorized` 和**完整已审核 SHA** 后才可远端 NPU 运行。

**第二门受限真实运行**：远端 `ssh yunnan-share2` 只 `git fetch`、`checkout --detach <approved_sha>`，确认 checkout 干净；对模型、数据的 910C OBS 与本机运行副本各自 `sha256sum -c SHA256SUMS`。先在新 SHA 上运行 1 对并审逐 token、有限值、HBM、耗时；再 10 对估算总时间；两门通过后才完成全部矩阵。单张 NPU，**最多 416 次前向、1 卡小时、45 分钟墙钟**，三者先到即停。不得在服务器源码树修改脚本或结果；缓存、日志、原始矩阵写 `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01/` 并复制到 OBS，本地镜像置于 Git 忽略的 `experiments/runs/`。

**2026-09-25 资产核验替代路径（仅在 `/data/data_dev` 的 `fuse.obsfs` 挂载仍卡住期间适用）**：P0 补证复制 PID 2602998 已在该挂载的 `request_wait_answer` 处保持 `D` 状态近五小时，因此本阶段不在 910C OBS 上再发起读取、校验或写入。上述“910C OBS 与本机副本各自校验”的 910C OBS 一项由以下证据链替代：共享权威 OBS `/data/obs_data_liuza/CADRetrieval/` 上的模型与数据 `SHA256SUMS` 文件 SHA256 分别为 `98c1c44b9052218a79f48e2c5f7879b579031981048612c23304bfdc9fe2c785`、`f75a7a48c9c97fe6933f8b10dbe13e779da95fbeeab88bb1a612104a5f57fd34`；910C checkout 外运行副本中的两个清单与其逐字节哈希相同。2026-09-25 对运行副本已重新执行逐项校验，模型 **11/11**、数据 **6/6** 均通过；共享 OBS 资产此前按官方 pinned revision 逐文件核过，记录见 `experiments/assets/IDEA01_ASSET_STATUS.md`。正式评分前仍须由 runner 再验运行副本，并记录这两个清单哈希及结果。运行产物先写 checkout 外，随后取回本地、转存共享 OBS，并以不自引用的 SHA256 清单在本地和共享 OBS 逐项复核；910C OBS 归档留待挂载恢复。这个替代路径只修改存储核验，不改变模型、数据、选择清单、评分或预算。

第二门按 **1 对 → 累计 10 对 → 全部 414 对**分三次审查，每阶段需要 `ACTIVE_TASK.md` 另行明确批准；`--resume` 只跳过已核实的已有配对，整个 run 的前向与墙钟预算累计。阶段 1 仅 `--pair-ids 0`，在 Codex 复核完整输入 ID、逐 token、有限值、环境、成本和 Git/清单绑定之前不得运行下一阶段。阶段 1 的小样本只能证明修复后代码可实际评分，不能得出排序结论。

任何哈希失败、输入边界错位、非有限分数、OOM、超长计划配对、实际 pair 超预算、重复/污染源映射、远端 Git 变脏都记录该失败并停止，退出非零；不能以跳过失败样本后只报成功集替代。保留每对完整 input IDs、目标位置/ID/logprob、sum/mean/EOS、query/candidate UID（簿记）、Git SHA、模型/数据 revision 与完整路径、环境双哈希、精度、设备、耗时和失败原因；环境按调用分文件。汇总表必须从原始逐对文件复算。

## 5. 裁决与下一步

P1 若同候选正确文本边际不稳定，先诊断数值/模板/数据，不进入更大库。即使 P1 的 exact-UID 排名很好，也只说明官方配对链可检索；**P2 的独立语义 qrels 和设计家族/几何等价核验仍未解决**。在这些真值未闭合前，不以 P1 推出 CAD 理解主张，也不启动 P2/P3。Kimi 分开回填事实、解释和建议；Codex 审核代码与原始结果后再制定下一轮。
