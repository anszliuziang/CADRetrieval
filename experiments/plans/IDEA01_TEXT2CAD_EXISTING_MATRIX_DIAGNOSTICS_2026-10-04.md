# Text2CAD 已有矩阵诊断：文本对齐与候选共同偏好

2026-10-04，用户要求获取已有数据并分析为何检索弱。本方案在新诊断结果计算前冻结；属于看过 G5 后提出的探索性分析，不是预注册确认实验，不替换 G5 主结果。

## 授权与边界

- Codex 本地 CPU/NumPy/stdlib 实现与执行，新增执行代码独立审查。零模型/BERT/NPU 前向，零 GPU，零下载，既有分数/UID/文本/长度不修改。
- cvteam6/cvteam14 因用户说明不可达，4-A6000 仅用于共享 OBS 只读存储核验；本地完整矩阵可读。
- 原正式 G4/G5 报告、证书和 Kimi 草稿保持原件。新独占诊断输出、失败/退出和代码输入指纹持久保存。

## 固定计算

1. 输入沿用 G5 八项 SHA pins、G4 证书 afad9371…、3666 条原清单 97f9cbb6…；所有 3644 chunk 在读取时核 SHA。复用已审纯 CPU 输入连接和读取函数，不重新跑评分或原数值审计。
2. 提取原 SUM 和诊断 MEAN 的完整 3644×8046 排名，exact score 降序/cad_index 升序 tie。恢复 7934 查询，与已保存 G5 每个 positive rank 逐项比较，必须全部一致。
3. seed=20261004，5000 次均匀排列 3644 个文本行。每组内所有原 query 使用相同置换行，CAD/qrels 保持原样，按原 query 等权；SUM/MEAN 共用同一排列。保存排列与 null R@1/5/10/20、MedR、MeanR。排列允许 fixed points，是标准 row permutation；不冒称每条都错配。
4. 另算全体其他 3643 文本行的精确错配期望 R@K/MeanR（排除原行）。置换 p=(1+不少于观察到表现的 replicate 数)/(5000+1)，rank 越小越好。只解释为标签对齐的随机化对照，依赖 exchangeability，不证明因果或语义等价。
5. 查询无关候选共同分数=各 unique 文本行的算术平均，不用 qrels 拟合。报告其 exact UID 指标、每行与该固定排序的 Spearman（固定 tie ordinal 排名）、Top20 重合、top20 候选覆盖/频次。文本组等权和原 query 加权均明确标注。
6. 两向加性平方和分解：文本行 offset、候选 column effect、交互 residual；额外报告去掉行 offset 后候选共同效应比例。该均值是经验共同偏好，不是无条件模型概率；方差占比不直接等于检索贡献。
7. 每个真实 query 的正确文本 score 相对所有其他文本该 CAD score 的 strict win fraction、tie fraction；按原 query 加权汇总。无需新前向，不能称干预机制证明。
8. 按 G5 固定 gallery 有效长度四分位（27/40/70）报告 SUM/MEAN 的正确文本与错配期望 R20/MeanR；unique vs repeated 同样报告。不据结果选择归一化或修正方法，不新增 relevance。

## 关口与交付

toy 覆盖重复组复用行/原 query 权重、完整 ties、正确排名方向、置换双臂共享、精确排除正确行、加性分解、候选排序和非有限值拒绝。独立审码通过后指定完整 Git SHA 本地一次分析。结果表、解释与局限写新 CODEX 报告。原正式评价不会重复；后续盲审多正例/同 gallery 强检索基线/新 checkpoint 或前向仅列建议，当前不运行。

## 论文对实验设计的约束

- [Text2CAD 原论文](https://arxiv.org/html/2409.17106v1) 第3节将 L0 定义为抽象外形、L3 定义为详细几何相对数值；第4节建模 CAD 构造序列的文本条件分布。L0 exact UID 与语义相关性应分别评价，不能将未命中 UID 自动判作语义错误，也不能未经盲审将所有未命中解释为语义正确。
- [Diffusion Classifier](https://arxiv.org/html/2303.16203v3) 第3.2节固定图像 x、比较条件 c 的 p(x|c)，并按类别先验得后验。本任务固定文本 t、比较不同 CAD c 的 s(c,t)，比较轴相反。推论：候选共同偏好在后者不会自动抵消；分类成功的文献不能直接担保本检索形式成功。
- [Likelihood Ratios for OOD Detection](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html) 在另一个任务中发现 likelihood 被背景统计混淆，并提出 likelihood ratio。这里仅借其提出假设，不能把跨领域结果当 CAD 根因证明。
- [Text-to-CAD Retrieval: a Strong Baseline](https://arxiv.org/html/2605.05572v1) 第IV-A节也是 L0，但预处理后的 test 为8023而本库为8046；已有不同模态训练。应在同一 UID gallery/同一 qrels 重算强基线，不能直接抄其发表数字与本实验并表。

下一轮若获授权：先在验证集冻结候选共同项修正规则，再在未用于选规则的测试集确认；对 query/Gallery 几何进行独立盲审多正例、带长度匹配的硬负例，检查语义条件区分能力；明确泛化单位为设计家族而非仅 UID。原 R20 2.609% 与重复组 UID ceiling 80.401% 的巨大差距不能仅用重复文本免责。所有这些是新研究方案，不在本次分析中偷偷改变正例或主方法。

## 执行结果（原方案冻结后追加）

代码00afcf73124edbb0056fc44cfc89d6755834efab通过独立Ready及31针对CPU测试，授权ref40639e4720c3524bf166f1223e633e35b7574fe2。仅一次本地CPU分析actual exit0/557.870秒；15868正式positive ranks全部一致。独立结果见证39548检查、5000完整共享排列及60000null统计通过；未重复模型前向或原G4全行审核。报告和完整统计见ACTIVE。新增诊断只在本地保存，原OBS归档结果保持。此方案已完成，不授权后续新方法、评分或运输。
