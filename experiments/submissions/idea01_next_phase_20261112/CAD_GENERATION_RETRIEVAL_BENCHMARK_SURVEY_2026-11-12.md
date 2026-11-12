# CAD 生成 × 检索：表示、数据集与 benchmark 对齐调研

日期：2026-11-12。执行：GLM 5.3（三个并行网络调研子代理 + 交叉综合）。目的：回答"生成命令序列的方法必然对应命令序列检索的数据集/benchmark；生成 BRep 对应 BRep 检索"这一原则在现实中成立到什么程度，为本项目选定评估基准。**标注规则：所有关键数字附来源链接；未能核实处明确标「未核验」；本轮不引入任何本地实验数字之外的新前向。**

---

## 1. 一句话结论（验收后修订）

**"生成表示 ↔ 同表示检索 benchmark"的原则目前只在两条线上有实例**：命令序列（IMMA，arXiv 2605.05572，Text2CAD L0 + 过滤后 DeepCAD test 8,023 库，R@1=9.71）与 **B-Rep（验收订正：MM-CAD:B 已发布公开协议——数据/代码/权重可得，验证图库 19,263 个 B-Rep，纯文本查询 R@1=15.42%；BRepCLIP 亦在 ABC 91K/CADParser 40K/AutoMate 65K 全图库报告文本检索）**。**text↔CadQuery 代码表示的检索 benchmark 本轮检索未找到**——代码线有大量文本对训练数据却无现成检索协议，且**自建协议的主要约束是训练隔离**（现有 Qwen 微调训练集与本 8,046 图库精确 UID 相交 4,581 个）。这意味着：(a) 我们在 Text2CAD 上的 UID 检索数字有可对照的外部锚点；(b) text↔CadQuery 协议可自组，但必须先冻结图库、正例、代码覆盖与训练隔离。

---

## 2. CAD 表示族谱与生成方法版图

| 表示族 | 代表生成方法（text 条件除注明外） | 现状 |
|---|---|---|
| **命令/参数序列**（DeepCAD sketch-extrude JSON，272 token、8-bit 量化） | **Text2CAD**（NeurIPS 2024 Spotlight，[2409.17106](https://arxiv.org/abs/2409.17106)，BERT 编码 + cross-attention decoder；本项目主线仪器）、CAD-GPT（[2412.19663](https://arxiv.org/abs/2412.19663)）、**CADmium**（[2507.09792](https://arxiv.org/abs/2507.09792)，Qwen2.5-Coder 1.5B–14B LoRA，MIT/Apache-2.0，**生成 JSON 序列而非 CadQuery 代码**——与我们 9-30 审计结论一致）、CAD-Llama（[2505.04481](https://arxiv.org/abs/2505.04481)）、GenCAD（[2409.16294](https://arxiv.org/abs/2409.16294)，验收订正：图像条件生成，非文本条件）；编辑类：FlexCAD（[2411.05823](https://arxiv.org/abs/2411.05823)）、CAD-Editor（[2502.03997](https://arxiv.org/abs/2502.03997)） | 最成熟；评测全部 generation-only（F1/Chamfer/Invalidity）；本轮检索未发现用生成分数做检索的实例 |
| **程序代码** | **Text-to-CadQuery**（[2505.06507](https://arxiv.org/abs/2505.06507)，文本继承 Text2CAD、CadQuery 代码由 Gemini 2.0 Flash 翻译 + 执行/渲染反馈修正，6 个开源 LLM 微调，Top-1 exact match 69.3%；HF `ricemonster/IEEE-T-ASE`，即本项目 P0/P1 仪器）、CAD-Recode（[2412.14042](https://arxiv.org/abs/2412.14042)，point cloud 条件）、CAD-Coder（同名两篇：[2505.14646](https://arxiv.org/abs/2505.14646) MIT/GenCAD-Code 147K 对；[2505.19713](https://arxiv.org/abs/2505.19713) CoT+RL）、Query2CAD（[2406.00144](https://arxiv.org/abs/2406.00144)，OpenSCAD，纯 prompt 工程）、CADCodeVerify（ICLR 2025） | 数据充足（17 万 + 百万级新数据集，见 §3）；评测 generation-only；**检索 benchmark 不存在** |
| **BRep** | SolidGen（[2203.13944](https://arxiv.org/abs/2203.13944)，无条件）、BrepGen（SIGGRAPH 2024，[2401.15563](https://arxiv.org/abs/2401.15563)，无条件/补全；**注意：2410.01551 是无关论文，勿引**）、ComplexGen（[2205.14573](https://arxiv.org/abs/2205.14573)，点云条件）、BrepDiff（SIGGRAPH 2025）、DTGBrepGen（[2503.13110](https://arxiv.org/abs/2503.13110)）、**HoLa（[2504.14257](https://arxiv.org/abs/2504.14257)，验收订正：§5.3 明确支持文本条件生成）**、BrepARG（CVPR 2026）；编辑类 B-repLer（[2508.10201](https://arxiv.org/abs/2508.10201)） | 文本条件 BRep 生成已有实例（HoLa）；检索评测见 §4（MM-CAD:B、BRepCLIP） |
| **相邻**：text-to-3D | CLIP-Forge（CVPR 2022，zero-shot 检索式生成，ShapeNet）、Point-E、Shap-E；image-to-CAD：Img2CAD（同名两篇 [2408.01437](https://arxiv.org/abs/2408.01437)/[2410.03417](https://arxiv.org/abs/2410.03417)） | 方法论参照系（对比对齐/生成先验） |

**新颖性表述（验收后采用审慎口径）**：本轮检索未找到「冻结生成器对固定 CAD 候选作条件生成分数排序」的直接先例（图像域有 Diffusion Classifier [2303.16203](https://arxiv.org/abs/2303.16203) / DiffusionITM [2305.16397](https://arxiv.org/abs/2305.16397) 先例；CAD 域已知检索工作 IMMA/MM-CAD:B/BRepCLIP 均为判别式双塔打分）。此为可保留的新颖性空间，同时是风险来源（裸似然读出的背景混淆）；结论保留检索边界，不做全称断言。

---

## 3. 数据集版图

| 数据集 | 表示 | 规模 | 文本 | 官方 split | 许可 |
|---|---|---|---|---|---|
| **DeepCAD**（ICCV 2021，[2105.09492](https://arxiv.org/abs/2105.09492)） | sketch-extrude 序列 | 178,238 | 无 | 随机 90/5/5；下游引用口径不一（CADmium：118,299/8,925/8,046；IMMA：119,482/8,904/8,023，过滤后） | MIT（CADmium 论文口径，原始 LICENSE 未核验） |
| **ABC**（CVPR 2019，[官网](https://deep-geometry.github.io/abc-dataset)） | STEP(BRep)+网格+分割+FeatureScript | 1,000,000 | 无 | — | 版权归创作者，无标准开源许可（商用前须核验） |
| **Fusion 360 Gallery**（[2010.02392](https://arxiv.org/abs/2010.02392)） | 序列/BRep/装配 | reconstruction 8,625；segmentation 35,680 part；assembly 8,251 | 无 | 各子集自定义 | 非商用研究 |
| **Text2CAD v1.0/v1.1**（[HF](https://huggingface.co/datasets/SadilKhan/Text2CAD)） | 同 DeepCAD | ~170K 模型 / ~660K 文本（4 层 × ~165K） | **LLaVA-NeXT**（L0）+ **Mistral-7x8B**（L1–L3）两阶段自动标注，v1.1 换 Qwen2-VL-14B + Qwen2.5-72B；**无人工逐条校验**；8,046 test CAD / 7,934 非空 L0 查询（本项目已核验的文件级计数；论文层无对应表） | 159,049/8,925/8,046 UID，零交集 | **CC BY-NC-SA 4.0**，gated |
| **Text-to-CadQuery**（[2505.06507](https://arxiv.org/abs/2505.06507)；HF `ricemonster/IEEE-T-ASE`） | CadQuery Python | ~170,000 对 | **直接继承 Text2CAD 文本**；代码 = Gemini 2.0 Flash 翻译 DeepCAD JSON + 执行/渲染反馈（成功率 53%→85%） | 90/5/5 | 未核验（arXiv 页未见 IEEE T-ASE 发表信息，org 名仅暗示投稿目标） |
| **CADmium-ds**（[2507.09792](https://arxiv.org/abs/2507.09792)） | DeepCAD minimal JSON | 176,017 DeepCAD + 8,307 Fusion360 | GPT-4.1（minimal JSON + ≤10 渲染图，>6 亿 token） | **与 Text2CAD 相同 split（118,299/8,925/8,046）** | MIT（模型 Apache-2.0） |
| **Zero-To-CAD-1m**（[ADSKAILab/HF](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m)，[2604.24479](https://arxiv.org/abs/2604.24479)） | CadQuery+BRep+STL+STEP+8 渲染 | **999,633 全合成**（LLM 反馈环境生成） | LLM 生成环境含描述 | 979,633/10,000/10,000 | **Apache 2.0**（许可最友好） |
| SolidLetters（随 UV-Net [2006.10211](https://arxiv.org/abs/2006.10211)） | BRep | ~96K 字母拉伸体 | 类别/风格标签，非描述 | — | — |
| SldprtNet（ICRA 2026，[2603.13098](https://arxiv.org/abs/2603.13098)） | STEP/sldprt + 13 类命令结构化文本 | 242K+ | Qwen2.5-VL 生成，声称人工校验 | 未核验 | CC BY-NC-SA 4.0 |
| Text2CAD-Bench（[2605.18430](https://arxiv.org/abs/2605.18430)） | CadQuery | 600 例人工精编（4 难度级） | 设计师手写 1,200 prompt | — | 纯 generation benchmark |
| ABC-NECAD | — | — | — | — | **未核验/可能不存在**（多源检索无命中；勿引） |

---

## 4. 检索 benchmark 版图（文本→已有 CAD）

### 4.1 存在的

**IMMA / "Text-to-CAD Retrieval: a Strong Baseline"（[2605.05572](https://arxiv.org/abs/2605.05572)，2026-05 preprint，venue 未核验）——命令序列表示上唯一基于 DeepCAD/Text2CAD 数据的检索 benchmark（验收定位：可作外部参照，按原协议引用）：**
- 协议：Text2CAD **L0** 查询（明确排除 L1–L3，因含显式尺寸会泄漏）+ DeepCAD test，过滤点云不可转换后 **119,482/8,904/8,023**；候选侧双表示 = 命令序列（272 token）+ 1024 点点云。
- 模型：CLIP 文本塔 + 序列塔 + PointNet 塔，双向 InfoNCE + masked feature decoder；推理 cosine 排序。
- 数字（Text2CAD L0，全部 baseline 在 119,482 对上重训）：

| 方法 | R@1 | R@5 | R@10 | R@20 | MedR |
|---|---:|---:|---:|---:|---:|
| **IMMA** | **9.71** | 25.54 | 36.05 | 46.85 | 25 |
| SCA3D | 6.33 | — | — | — | — |
| Parts2Words | 4.81 | — | — | — | — |
| RoMa | 3.80 | — | — | — | — |
| TriColo | 2.12 | — | — | — | — |
| 消融：sequence-only | 5.00 | — | — | — | — |

**BRepCLIP**（[2606.05515](https://arxiv.org/abs/2606.05515)，验收订正）：BRep 原语↔语言/图像对比对齐（CADCap-1M 的 ABC 子集 **400K 训练配对 + 10K 验证**，frozen OpenCLIP bigG）；§4.1 在 **ABC 91K / CADParser 40K / AutoMate 65K 三个全图库**报告文本检索：Top-1/5/10/20 = 8.59/24.52/35.08/47.89（ABC）、5.00/15.08/22.12/30.60（CADParser）、9.42/24.18/32.86/42.83（AutoMate）。gallery 与我们完全不同，数字不可并表；其 split/qrels 的可获取性未核验。
**MM-CAD:B（验收补充，重要）**：MM-CAD 基准的 B-Rep 检索子协议（[官方项目](https://exanos.github.io/MMCAD/)，CGF/SGP 2026）：**已发布数据、代码与权重**，验证图库 **19,263 个 B-Rep**，纯文本查询 **R@1=15.42%**——这是 B-Rep 表示上现成的公开文本检索协议。
**其他**：CAD-RAG（WSCG 2025，检索增强生成）、UV-Net 自监督检索（仅定性图无 R@K）。

### 4.2 本轮未找到现成协议的（验收后重新表述）

- **text↔CadQuery 代码检索 benchmark**：本轮检索未找到。Text-to-CadQuery、Text2CAD-Bench、Zero-To-CAD-1m 全部 generation-only。**自建的主要约束不是数据而是训练隔离**：现有 Qwen 微调（Text-to-CadQuery 训练集）与本 8,046 图库精确 UID 相交 **4,581 个**（该权重在本图库上不能声称 zero-shot；259/247 只是 val/test 的交集口径），重组评测前必须冻结真实图库、正例、代码覆盖与训练隔离。
- **text↔BRep 检索**：验收订正——已有公开实例（MM-CAD:B 19,263 库 R@1 15.42%；BRepCLIP 三库全图库评价），不再是空白；其 split/qrels 可获取性分别标注为待核验。
- DeepCAD/Fusion360 社区无公开维护的检索 leaderboard（Papers with Code 2025-07 下线后无承接）。
- **CADPrompt**（[2410.05340](https://arxiv.org/abs/2410.05340)，ICLR 2025）经核实是 **200 例代码生成评测**（IoGT/Hausdorff 等），没有检索协议、没有 R@K。
- Text2CAD 原论文无检索实验（两次全文抓取确认）；其"CAD vector"即 DeepCAD 量化序列，非对齐 embedding。
- **协议设计注记（验收）**：检索协议以查询、图库实体和 qrels 固定；同一实体若有经验证的程序、序列和 STEP，各模型可读各自的原生表示并共用同一评价——这是"生成表示↔同表示检索"原则的可操作形式。

---

## 5. 表示 ↔ benchmark 配对矩阵（用户原则的现实成立度）

| 生成表示 | 有文本对训练数据？ | 有同表示检索 benchmark？ | 缺口结论 |
|---|---|---|---|
| 命令序列（DeepCAD JSON） | ✅ Text2CAD 660K / CADmium-ds 176K | ✅ **IMMA**（唯一，L0，8,023 库，R@1=9.71） | 只此一家；只用 L0；库小 |
| CadQuery 代码 | ✅ Text-to-CadQuery 170K / Zero-To-CAD-1m 1M（Apache-2.0，验收已核发行卡） | ❌ **本轮检索未找到** | 可自组协议（Text2CAD UID ↔ `CQ/{uid}.py` 映射本项目已逐字核验），**主要约束是训练隔离**（现有微调训练集与本图库相交 4,581 UID）；IEEE-T-ASE 发布页缺许可元数据，派生资产许可须单独核对 |
| OpenSCAD/FreeCAD | ❌（仅 Query2CAD prompt 工程） | ❌ | 双缺 |
| BRep | **有实例**（验收订正：BRepCLIP 训练用 CADCap-1M ABC 子集 400K 配对；HoLa 文本条件生成） | **有公开协议**（MM-CAD:B 19,263 库 R@1 15.42%；BRepCLIP 三库评价） | 数据与协议都有实例；与本项目的可复用性待核 |
| mesh/渲染+文本 | ✅（Text2CAD 渲染包、Zero-To-CAD-1m 8 视角） | ❌（CAD 专属缺；通用 3D 检索不属 CAD） | 可作为几何盲审的标注素材源 |

---

## 6. 与本项目的对齐

1. **可同库比较的外部锚点只有一个**：IMMA 在过滤后 DeepCAD test（8,023）+ Text2CAD L0 上 R@1/5/10/20 = 9.71/25.54/36.05/46.85、MedR 25；本项目 G5 SUM = 0.2017/0.9201/1.6385/2.6090%、MedR 3411.5。**口径差异必须声明**：(a) 他们过滤 23 个不可转点云 UID（8,046→8,023，逐 UID 清单未公开，未核验）；(b) 他们全部经 119,482 对判别训练，我们是冻结生成模型零训练评分——这是"同库不同训练条件"，只能作参照上界而非公平对照；(c) gallery 构成论文未明示（推定 8,023 test CAD，未核验）。
2. **外部基线弱这一事实本身有信息量**：最强判别模型 R@1 也只有 9.71%（随机 0.0125% 的 ~780 倍，但绝对值低）；sequence-only 消融 R@1=5.00。说明"L0 文本 → 8K CAD 库精确 UID 检索"对**所有现有方法**都难，我们 2.6%@20 的裸生成读出并非孤立失败，但它显著低于训练过的判别基线——差值包含"判别训练收益"与"读出失配"两部分，正是后续实验要分离的对象。
3. **P0/P1 的 CadQuery 线（验收订正）**：现有 Qwen 微调的**训练集**与本 8,046 官方 test 图库精确 UID 相交 **4,581 个**（247/259 只是其 test/val split 与官方 test 的交集口径）——该权重在本图库上不能声称 zero-shot/held-out；做 CadQuery 检索必须重组协议并冻结训练隔离（如用 Zero-To-CAD-1m 等独立源训练或只评与所有训练源零交叉的子集）。
4. **可自建的 benchmark**（论文级贡献候选）：text↔CadQuery 检索——gallery = Text-to-CadQuery test split 的 `CQ/{uid}.py`（零前向可组装、映射已核验），查询 = Text2CAD 同 UID 文本（注意同 UID 多层级文本与近重复设计问题，需引入 T2 的多正例/盲审规则）；评测对象 = 生成式读出（qwen2.5-3B-SFT teacher-forced）vs 判别式基线（嵌入）vs 词面基线。这直接填 §5 的最大空白，且与"生成表示对应同表示检索"的项目原则自洽。

---

## 7. 建议（供验收讨论，未授权执行）

1. **主线对照**：把 IMMA 四个重训 baseline 数字列入我们报告的"同库参照"栏（标注训练条件差异），不再说"没有任何可比较数字"。
2. **表示策略**：命令序列线继续用 Text2CAD 官方 test（唯一现成协议）；若论文需要"基准贡献"，优先自建 text↔CadQuery 检索协议（资产、映射、split 全部现成，缺口真实存在）。
3. **数据来源注意**：Text2CAD/IEEE-T-ASE 为 CC BY-NC-SA 4.0（gated），公开基准衍生需核许可；Zero-To-CAD-1m（Apache-2.0）是许可最干净的大规模替代/扩充源。
4. **引用卫生**：BrepGen 用 2401.15563；CADmium 生成的是 JSON 序列不是 CadQuery；CADPrompt 不是检索基准；ABC-NECAD 未找到出处勿引；"CAD-GRAM/BRepDiffuser"未检索到，疑为记忆混淆，勿引。

## 8. 未核验事项清单

IMMA gallery 构成与 23 个过滤 UID 清单；2605.05572 与 2505.06507 的正式发表状态；Text2CAD 每层文本精确计数；DeepCAD 原始 LICENSE；ABC 许可表述冲突；SldprtNet split；Zero-To-CAD-1m 与 Text2CAD 文本是否同源（其描述为 LLM 生成环境产物，推定独立）。以上在正式引用前须逐项回原文核验。

---

*子报告来源：三个并行调研代理（生成方法版图 / 检索 benchmark / 数据集版图），本文件为其交叉综合；原始链接均已附上，综合时修正了代理发现的 ID/归属错误。*
