# M1c：原版 Text2CAD 两路命令向量评分与两卡测速

状态：用户已选择原版 Text2CAD 作为下一生成模型。此合同的第一阶段仅本地实现、CPU 审计和资产准备；`ACTIVE_TASK.md` 在 Codex 审码前保持 `implementation`、NPU 预算 0。上位协议为 `IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md` 的 A，以及 `IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md` 的完整矩阵与七天门。

## 固定对象与主张

- 模型：原作者 `Text2CAD_1.0.pth`，SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`；冻结的 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`，其权重 SHA256 `28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c`。原作者源码固定本地只读镜像 Git SHA `669a81472ede2269837c22a4d00070eaac360c82`；许可 `CC BY-NC-SA 4.0`，如将必要源码改编后纳入 Git，保留来源、许可与改动说明。
- 候选：官方 v1.0 `cad_seq.zip` 中 UID 对应 `.../seq/<chunk>.pth` 的 `vec`/`mask_cad_dict`；ZIP SHA256 `9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa`。query/gallery/qrels 复用已审 M0：图库 8,046、非空查询 7,934、唯一 `strip()` 文本 3,644。评分进程仅见文本与候选向量，不得读取 UID 对应关系、CSV、audit 或 qrels。
- 主张是冻结生成模型的**原始两路 logits 构成的条件排序分数**能否排序登记 UID。训练损失是 `CrossEntropyLoss(label_smoothing=0.1)`，不得直接当 NLL；两路 x/y 的原始 `log_softmax` 有效项相加，报告为“two-head factorized score”，不宣称 CAD 几何边缘似然或纯 MLE 模型。

## G0：本地实现和零前向核验

1. 首先审官方代码的模型定义、训练 teacher forcing、`Text2CAD_1.0.pth` 的 state dict、BERT 冻结方式、双头目标、mask、开始/终止/填充、`flag_vec/index_vec` 的确定性构造、文本 tokenizer 与 272 最大 CAD 序列长度。官方 `train.py` 以 `cad_vec[:, :-1]` 预测 `cad_vec[:, 1:]`，loss 在 `~key_padding_mask[:, 1:]` 上分别计 x/y。文档必须写出每一类 token 哪一路有效，以及任意不一致；不能猜测最后一个有效 token 是 EOS。
2. 在本地 `experiments/src/` 内实现 910 兼容的最小模型适配和评分器，严禁在服务器修改源码。官方 `TextEmbedder.get_embedding` 硬编码 `.to("cuda")`，须在本地修成显式 device 并核对输出语义；不得将文本重新清洗成官方训练脚本未实际返回的字符串。模型 checkpoint 只保存可训练参数，BERT 从固定基础权重加载；严格确认非 BERT 参数的加载键与数值，意外缺键/多键阻止前向。资产在远端 checkout 外，运行时 `local_files_only`。
3. 评分数学：对每对 `(q,C)` 做 teacher forcing；从 `pred[B,t,head,vocab]` 中取 next-step 有效 x/y 标签，FP32 `log_softmax` 后求 sum；保存每对两个头的 sum、有效项数量、合计分、有效长度、拒绝原因。使用官方 mask，不为更快改命令语义；验证前缀、padding、终止、flag/index、文本 query 与图库表示一致。若分数依赖固定候选 vector 中的 padding 或未定义 token，停止。
4. 本地 CPU 单元/小样例测试包括 shift、双头 mask、EOS 与 padding、长度边界、batch=1/B 数值一致、重复文本去重映射、输入字段隔离、资产和 checkpoint 失败路径、预算与结果不覆盖。零前向的真实 checkpoint 加载审计必须检查 SHA、权重键和统计，不能仅以 `strict=False` 无报错放行。
5. 实现可复现的固定 pair 清单生成、外层 UID 映射和 benchmark runner。固定输入只从 query 文本与候选向量生成；评分程序不能 import qrels。测速先覆盖短/中/长向量、长文本；batch `1,2,4,8`，HBM 允许再测 `16/32`，每档同一批 pair、至少两次 warmup、重复计时，记录 forward+score、端到端、pairs/s、有效命令项/s、HBM 峰值、OOM 与拒绝率。两卡仅处理互斥分片，绝不分摊同一 pair 的两路输出。由于 BERT 文本编码器冻结、`eval()` 且唯一文本仅 3,644 种，先核验按唯一 query 缓存 BERT 表示能否与逐对前向逐项等价；若等价再把缓存纳入正式吞吐。任何更激进的候选侧缓存须另做同 pair 数值一致性门，不能在 baseline 正确性前启用。
6. Claude Code / Kimi 将实现、CPU 测试、官方代码差异、资产 SHA、精确远端命令和缺口写入 `experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_IMPLEMENTATION.md`，结尾标 `review_required`。Codex 审码并在 `ACTIVE_TASK.md` 填完整批准 SHA、单次对数与卡时后，才准任何真实模型前向。CUDA/CPU 参考对照如需计算也须另获预算。

## G1–G4：审查后分门执行

- G1：单卡固定 1 对，核验真实加载、两路 logits/CPU gather、终止与 padding；再审后 10 对跨长度，比较 batch=1 与 batch>1；任一差异超预设容差、非有限或资产不符即停。
- G2：测代表性长度分层、batch 和双卡稳态吞吐；对正式 3,644×8,046＝29,319,624 对的真实文本/向量长度分布加权外推，计冷启动、预处理、I/O、失败重试、输出写盘，并乘至少 1.30 安全系数。
- G3：只有 G2 保守 ETA ≤168 小时才对事前固定的 100 种 query 文本评分**完整 8,046 图库**，复核双卡长稳态和 ETA。若 G2/G3 任一步估计 >168 小时，停下来与用户讨论，不启动全量，也不改为短名单。
- G4：G2/G3 均通过且独立审查代码、数据和预算后，才授权两卡完整精确矩阵、断点/分片校验与独立 qrels 评价。全量不可用查询子集或 rerank 替代。

所有代码/配置/文档仅本地编辑并 Git 推送；`yunnan-share2:/home/ma-user/work/CADRetrieval` 只 fetch/detach 审核 SHA 运行。模型、数据、日志放 `/home/ma-user/work/CADRetrieval_runtime/`，重要结果归档共享 OBS；NPU 预算只认 `ACTIVE_TASK.md`。
