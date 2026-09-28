# Idea 01：Text2CAD 数据上的冻结生成模型检索评测

状态：执行合同。负责人 Codex；本地 Claude Code / Kimi 负责实现与运行，Codex 在每次扩大计算前审代码与原始结果。直接用户指令已授权开展评测；每阶段资源和有效 SHA 仍由 `experiments/ACTIVE_TASK.md` 指定。

## 1. 核心问题与可声称结果

给定文本查询 `q`、不含配对描述的 CAD 图库 `G`，冻结文本条件 CAD 生成模型，对**图库中已有 CAD 的模型原生表示**计算条件分数并排序；不用配对数据训练新的检索头。主指标为配对 UID 的 R@1/5/10/20 和 MedR；这衡量登记实例的名次，不能自动称为全部语义正确 CAD 的检索率。

主要主张只留两条：(C1) 生成模型原生分数是否能把登记 CAD 排到前列；(C2) 候选先验、长度、表示与训练集交叉对 C1 有多大影响。负结果同样记录。不同表示/训练语料间的差异**不能**解释成某种生成架构的单因素优势。

## 2. 评测协议，先于跑分冻结

**A. Text2CAD-v1.0 官方 test 协议**。固定 `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` 的 `text2cad_v1.0/train_test_val.json`：159,049 train、8,925 validation、8,046 test UID。以官方 v1.0 CSV 的 L0 `abstract` 作主查询，**只去掉文本首尾空白（`str.strip()`），不改大小写、不合并内部空白**；每个唯一 test UID 的 CAD 作图库候选，配对 UID 为登记正例。query 清单必须包含供评分器读取的实际文本与不指向正例的 query ID，配对 CAD UID 仅放 qrels；gallery 清单不得含配对文本或由配对文本导出的标记。重复组大小等评价分层信息只放 audit，不交评分器。评分器不得读取 UID 映射、原始 CSV、audit 或 qrels。Codex 只读核查：8,046 个 test CAD 在 `cad_seq.zip` 与 v1.0 minimal JSON ZIP 均有源文件；CSV 只覆盖 7,946 个 test UID，其中 12 个 L0 为空。因此主协议暂定**完整 8,046 CAD 图库、7,934 个非空 L0 查询**，以正式清单独立复核为准。缺失查询不能伪造成检索失败；缺失可评分 CAD 则须保留在图库覆盖报告与不可评分账本中，不能静默缩小图库。L1–L3 仅作事前指定的文本具体度分层。

**B. IMMA 已发表协议**。IMMA 从 DeepCAD 同源对象剔除不能转点云者，报告 8,023 个 test 样本，只用 Text2CAD L0；主表 R@1 9.71%、R@5 25.54%、R@10 36.05%。若拿不到作者最终 UID、图库和查询清单，或不能完全复现其过滤规则，A 的成绩只能标为 Text2CAD-v1.0 衍生检索成绩，**不与 IMMA 数字直接并表**。若将来重建 B，须额外存 8,046→8,023 的全部剔除 UID 与理由、点云生成设置及 test gallery 证据。论文：[IMMA](https://arxiv.org/html/2605.05572)。

**版本歧义必须核查**：IMMA 论文只写 Text2CAD、L0，未明确 v1.0/v1.1。官方 v1.0 CSV 缺 100 个 test UID 且 12 条 L0 为空；已校验的官方 v1.1 CSV (`sha256=cded1b8048f432c192127a5d177da59dc926149ac02fb9ae31e1015ee6219136`) 覆盖 8,046 个 test UID 且仅 1 条 L0 为空。不能仅因 IMMA 报 8,023 个 test CAD 就推断它使用哪个文本版本。复现前须取得版本证据或分别报告两版。

**登记 UID 的可识别上限**：v1.0 主协议的 7,934 个非空 L0 查询在上述 `strip()` 后有 3,644 个逐字不同的文本；639 个重复文本组共影响 4,929 条查询，最大组有 371 个 CAD。若直接比较未去空白的原始 CSV 字符串，则是 4,010 种、693 个重复组、最大组 320；该口径会把仅首尾空白不同的描述视为不同查询，不用于主指标。同一规范化文本面对同一图库给出相同排序，故以单配对 UID 为唯一正例时，不考虑其他错误的理论 R@1 上限也仅为 `3644/7934=45.93%`；K=2/5/10/20 的相应上限由 `sum_text min(K, group_size)/7934` 给出。指标表必须同时列唯一文本数、重复率、按重复组大小分层和上限；不得将低 UID R@1 直接解释为模型不懂文本。

**C. 跨模型干净共享子集**。先对每个权重的实际训练 UID 做交叉审计，再由预先声明的集合运算确定所有被比较模型均未训练的官方 test UID；训练交叉者绝不作为该模型的 held-out 正例。所有模型在同一固定 query/gallery/qrels 上重算才能并列。已从本地 P0 原始 UID 清单独立计算：Text-to-CadQuery Qwen2.5-3B-SFT 的 99,236 训练 UID 中有 **4,581** 个落在 A 的 8,046 test UID 内；其现有 val/test 与 A 的交集分别为 259/247。因此该模型**不得**在 A 全集上报告无泄漏成绩。还须审计它的 `CadQuery.zip` 对干净 UID 的覆盖。原生 Text-to-CadQuery test 5,514 项可作补充协议，但不能冒充 A/B。另一方面，`chandar-lab/CADmium-ds@ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4` 的 118,299 train / 8,925 val / 8,046 test UID 与 v1.0 划分逐项同属对应 split，三组相互无交叉；CADmium-3B 暂具备 A 的 held-out UID 条件，仍要核对权重训练配置。

## 3. 已发表检索方法的定位

| 工作 | 与本协议的关系 | 已核实数字 | 直接比较条件 |
|---|---|---|---|
| [IMMA](https://arxiv.org/html/2605.05572) | Text2CAD 来源、L0、序列+点云判别式检索 | 主表 Text2CAD R@1/5/10 = 9.71/25.54/36.05% | 须恢复其过滤后 test UID、图库和 query；代码在论文/作者页仍写将发布 |
| [CADIR](https://arxiv.org/html/2608.00891) | 从 8,000 个 Text2CAD 来源对象转换为构造图，4,000 测试，在 N=200/1,000 随机候选池评测 | text→graph N=1,000 的 R@5/10 = 74.95/80.54% | 其随机池与表示不同，不能对照 A/B |
| [CAD-RAG](https://wscg.zcu.cz/WSCG2025/papers/C41.pdf) | DeepCAD 来源的文本→渲染图/点云索引，取回 CAD 命令 | 见 `reports/TEXT_CAD_RETRIEVAL_BENCHMARK_AUDIT_2026-09-26.md`，表格标签有歧义 | 不在同一 test UID/gallery 上，先只列相关工作 |

不把生成论文的 Chamfer、F1、程序 exact match 当作检索 R@K；也不将小候选池 R@K 与 8,046 全库 R@K 混报。

## 4. 生成模型路线和评分定义

| 路线 | 代表模型与资产 | 候选表示与分数 | 准入门 |
|---|---|---|---|
| DeepCAD 向量/命令自回归 | [Text2CAD 原作者权重 `Text2CAD_1.0.pth`](https://huggingface.co/datasets/SadilKhan/Text2CAD) | 官方 `cad_seq.zip` 的命令向量；从原始输出 logits 对有效命令/参数及停止位计算条件分数 | 必须按官方代码审计多头分解、mask、padding、EOS 与 `label_smoothing`；**诊断臂**，不称纯 MLE 训练模型 |
| minimal JSON 自回归 | [CADmium 1.5B/3B](https://huggingface.co/collections/chandar-lab/cadmium)，先选 3B | 官方 minimal JSON 固定序列化；teacher-forced 原始 token log-prob 和 EOS，sum 为主，mean 仅诊断；可加预先固定的空查询校正 | 权重训练 UID、LoRA base、prompt/template、输出 JSON 版本、许可、目标完整长度均核实；若训练集与 A test 重叠，转 C |
| CadQuery 代码自回归 | [Text-to-CadQuery Qwen2.5-3B-SFT](https://arxiv.org/html/2505.06507)，本地 P1 已核过 scorer | 同 UID 的 CadQuery 程序 `response+EOS` teacher-forced log-prob；raw sum 与固定空查询校正 | A 与其训练交叉 4,581 UID；只能在 C 或原生 held-out test 报无泄漏成绩。逐 UID 核对源码覆盖、可执行几何和文本污染 |
| 文本条件 B-rep 扩散/STEP 序列 | [HoLa 论文](https://arxiv.org/abs/2504.14257) / [STEP-LLM](https://github.com/JasonShiii/STEP-LLM) | HoLa 的固定噪声表去噪误差；STEP-LLM 的固定 STEP 序列 log-prob | **可行性附录，不阻塞前三臂**。须先证明 CAD→B-rep/STEP、模型 AE/预处理、文本条件权重、长度/面数覆盖、训练重叠；失败时记录覆盖率，不强行给 R@K |

CADmium 与 CadQuery 的 JSON/代码评分只针对各自固定的程序代表元，不把其 log-prob 称作 CAD 几何的精确边缘似然。HoLa 去噪 MSE 不称精确 NLL。纯 MLE 主表仅纳入训练损失与目标定义经代码核实满足要求的仪器。原始模型使用训练文本本身的复述/数字线索可能形成捷径，需保存 L0–L3 分层和词面基线。

## 5. 执行顺序和自动停止门

| 阶段 | 工作与可交付证据 | 计算上限与停止门 |
|---|---|---|
| M0 数据/模型审计 | 固定官方原件的 revision、size、SHA256；UID/文本/命令/JSON/CadQuery 连接、训练交叉、全图库覆盖、重复和几何代表元小样本；写 `query/gallery/qrels` 清单；检索论文原协议对照表 | CPU/网络与 OBS；不得 NPU 评分。没有来源证明或训练集不可审计的模型暂不进入主表 |
| M1 评分仪器 | 本地实现 Text2CAD/CADmium 原生 scorer，复用已审 Qwen scorer；单元测试覆盖 shift/mask/EOS、batch=1 与 batch>1 等价、截断失败、token 精度；提交完整 SHA | Kimi 提交后 Codex 审码；每模型最多 1 对 NPU 烟测，再 10 对吞吐测试。任何非有限、错位、越长、漏标签即停 |
| M2 固定先导 | 在 M0 冻结的 100 query × 100 候选池先做条件分数、空查询校正、随机/长度/字符 TF-IDF 基线；每 query 保证自身目标在池中并记录构造种子。再抽 100 query × **完整可评图库**测吞吐与 R@K | 小池不是全库指标。先导代码与原始分数须再审；运行时间以 10 对测得吞吐重估，超预算时停并保留完整账本 |
| M3 正式评测 | 对通过门槛的模型，优先 A 或 C 的**全图库**与全部合格查询；计算 raw、预注册校正及分层指标。若穷举耗时过大，可单独报告固定无监督短名单 + 生成重排的全 query 结果，并记录一阶段召回上界 | 不把短名单结果标为穷举生成检索；不把 A/C 与 B 发表数字直接比较。长作业必须可断点、幂等、按块 SHA 校验并由定时任务检查；规模扩张前 Codex 再批准具体 SHA、卡数与墙钟 |

现有 Qwen P1 的 414 对用时 217.19 秒，约 0.52 秒/对。8,046² ≈ 6,474 万对；按原实现单卡外推为约 392 天，仅为成本警报，**不得**当优化后吞吐。M1 必须实测批处理、候选分组等能否缩短；未达到可接受 ETA 时优先 M2 的完整图库查询子集与可审计两阶段 M3，不让无限期任务占卡。

## 6. 结果包与审核

每一臂存 UID/split/训练交叉清单、query/gallery/qrels、模型与数据完整 revision 和文件哈希、Git 完整 SHA、环境、评分定义、每对原始分数及 token 诊断、失败/覆盖、吞吐与总成本、指标脚本、随机/长度/词面基线和分层统计。结果事实、解释和下一步建议分开回填。Codex 同时复核代码、映射与原始结果，再决定可声称的结论。

源码、配置、计划只在本地编辑提交，经 Git 到服务器指定 SHA 运行；OBS 保存数据/权重/结果。当前用户要求长实验自动监控，但定时任务只在有已审核并启动的长作业后创建，避免空转。M0/M1/M2/M3 每次通过 `ACTIVE_TASK.md` 切换有效阶段与预算。
