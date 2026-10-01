# Text2CAD 固定64对扩展 G1 与 CPU 资产核验

状态：**CPU资产预处理已授权，真实704对前向尚未授权。** R3代码已审，见 `reviews/IDEA01_M1C_G2_R3_CODEX_REVIEW_2026-10-01.md`；上位为 M1c 与 G2 protocol repair。最终前向授权须由 Codex 在 ACTIVE_TASK 另写完整 SHA 和预算。

## 目的与固定范围

此前十对门没有形成真正32对批，也没有验证 trim+cache 组合。现在使用事前按实际 token/CAD 长度选出的64对，核对 reference、16/32及组合优化是否在原固定容差内等价。这里只决定实现能否用于后续测速，不产出 R@K 或语义有效性结论。

- 8,046 CAD、3,644唯一文本与 pair_manifest_v4 不变；评价 UID 映射不进入选择或评分进程。
- token 审计使用固定 BERT tokenizer，包括 special tokens，不清洗/截断文本；不加载 BertModel。
- 64对为8个文本 token 长度分位 × 8个 CAD 长度分位：0/10/25/50/75/90/99/100，包含真实最长文本和 CAD。重复分位索引必须如实记录，不能按评分挑样本。
- 默认模型/双头 score/shift/mask/FP32与原容差不变；不得用含label smoothing的训练loss代替分数。

## 顺序与责任

1. Codex 审核 R2 代码及受影响 CPU 测试。仅通过后 push，远端检查干净并 Git fetch/detach 审定完整 SHA。
2. 远端只运行 `audit-text-lengths`，输出 checkout 外的派生 token 长度资产；设置离线缓存/临时目录在 runtime，零模型前向。Codex 取回原文件并重算 SHA，核3,644条索引、token范围、texts/BERT文件指纹。
3. **本地**从取回的审计文件及本地 pair_manifest_v4 运行 `select-g2`，生成新配置并审查、Git提交。远端不生成或改配置。核每档权重合计29,319,624、固定两个文本分片、真实批/tail、样本与64对选择；资产和配置分别保留SHA。
   - 本地使用 `D:\projects\IdeaSpark\CADRetrievalIdea1_runtime\pair_manifest_v4_worker\`，仅含四份已审 worker文件；2026-10-01复制后重核SHA一致：manifest `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`、texts `c59b4aef8137b1e71c1581291a4c1d2bcbafe36f22d81d01431a5cc02dad7a21`、cadstore `2f441110f013bb614b0bd2fdb7a3b49569a231bbf3f8eaa2c03ac1d725cb1380`、cadlength `daac66a102e34fe45c9cb474922076b0d62345ab0c9796f917a0f8085c6ea0c3`。旧本地原始目录保留，不把其中评价映射复制进此worker目录。
4. Codex 更新 ACTIVE_TASK：允许单卡 npu:0 的一个 profile-consistency 调用，固定704评分对、上限1卡小时；明确批准完整代码/配置 SHA，另提交授权文档。远端只按批准SHA运行，不沿用旧账本。
5. Claude Code/Kimi 保存11份有效行 logits、逐对分数与计数、真实批尺寸、环境、资产/配置/代码指纹、账本、stdout/stderr、失败记录，取回本地并归档共享OBS。Codex 独立复算，不只采信 passed 字段。

## 固定变体与预算

| 变体 | batch | trim | text cache | 对数 |
|---|---:|---|---|---:|
| reference | 1 | none | false | 64 |
| baseline16 / baseline32 | 16 / 32 | none | false | 128 |
| trim32 | 32 | 有效最长 CAD | false | 64 |
| cache32 | 32 | none | true | 64 |
| trim+cache | 1/2/4/8/16/32 | 有效最长 CAD | true | 384 |

合计704对。没有额外真实模型 warmup 或试跑；失败立即保留已完成与导致失败的原始证据，非零退出，正确最终化账本，不自动重试或放宽容差。

## 独立验收

- 核实际运行 SHA = approved SHA、工作树干净、完整704预算与账本一致；原始资产/config/token audit SHA 在运行端与本地一致。
- 固定64 pair顺序与各variant实际batch/trim/cache一致，无候选丢失或截断。
- `weights_only`加载11份原始有效行 FP32 logits；核每pair真实有效行数、两个头与vocab形状、所有值有限。
- 从原candidate vec/mask构造targets，独立FP32 log_softmax/gather核逐头/总分、有效计数；比较各variant与reference的逐项固定logit及score容差。可在CPU使用相同FP32公式，但计数/shift/mask须独立核，不直接信report分数。
- 不要求不同batch位级相等，容差固定；失败不追加新测试来冲淡失败。
- 归档原始结果、审核脚本/结果及SHA清单至共享OBS；保留本地可审计副本。

## 通过后的去向

此门通过后，Codex另审并激活真实生产测速profile（允许的trim/cache只来自通过的具体组合），保持相同正式批界/文本分片/固定资产。G2两卡实测后按较慢卡完成时间加冷启动/I/O/故障余量、至少1.30安全系数判七天门；G3稳态复核后才启动完整矩阵。此文不授权G2、G3或全量。
