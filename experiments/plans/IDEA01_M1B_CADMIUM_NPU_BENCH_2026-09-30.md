# M1b：CADmium 真实评分 runner 与两卡吞吐验证

状态：`ACTIVE_TASK.md` 激活后，**先仅本地实现与 CPU 测试，NPU 预算仍为 0**。用户已授权后续验证性 NPU 实验，但须 Codex 审查实现完整 SHA、资产/清单/预算后，另把 `pilot_authorized` 写入唯一入口。科学协议见 [Text2CAD 总计划](IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md)，两卡阈值见 [自动执行路线](IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md)。本合同只针对 CADmium-3B；其它模型另立子合同。

## 本地实现交付

1. 复用已审核的 `experiments/src/idea01/cadmium_scorer.py` 输入/目标定义：官方 adapter tokenizer、固定 system prompt、`query.text`、CADmium-ds `json_desc`，完整 assistant 目标含 `<|im_end|>` 和换行。评分核心不接收 `cad_id` 对应 qrels、CADmium `annotation` 或原 CSV。所有配置/源码/脚本写在本地 `experiments/` 并经 Git；不得在云脑编辑。
2. 实现真实 `Qwen/Qwen2.5-Coder-3B-Instruct@488639f1ff808d1d3d0ba301aef8c11461451ec5` + `chandar-lab/CADmium-3B@6c6a79d160595207804171be56a037883b369abd` LoRA 加载，运行时优先从已 SHA 核验的云脑 checkout 外副本读取，`local_files_only`。真正前向之前校验 adapter/base/tokenizer/小型配置文件、query/gallery 清单的 SHA 与模型 revision，任何缺失非零退出。
3. 在 NPU 上按 causal shift 对**目标位置**求 FP32 `log_softmax` 目标 token 分数；只传回逐对标量、长度和抽查所需的目标 token 分数，不把整批完整 vocab logits 复制到 CPU。若为节省 HBM 做 chunk、动态 padding、`use_cache=False`、按长度分桶或其它优化，必须保持与已审 CPU 数学核心在同一真实 logits 上一致。无截断；`len(full_ids)>4096` 必须记录 unsupported。batch=1 与 batch>1 的同一对分数按事前写定的数值容差比较，超差时停止。
4. 以 M0 已审 `query.jsonl`、`gallery.jsonl` 为评分输入，并从 pinned test parquet 只读取 `uid,json_desc` 构造候选文件。外层 logger 可保存 UID，评分函数不能看 `qrels.jsonl`；评价脚本另开进程读取 qrels。独立复算每个模型上下文可评分覆盖及拒绝原因。固定 query 去重只依据规范化文本，查询映射到 7,934 条原始 query 的过程做可逆断言。
5. 实现可重入的 **benchmark 模式**：给定预先落盘的、只由 query 文本与候选 JSON 构成的固定 pair 清单，测 batch size `1,2,4,8`，若 HBM 允许再测 `16`；每个配置使用相同 pair 集、固定输入次序、长度分桶和至少两次 warmup，记录纯前向+评分时间、端到端含 tokenization/I/O 时间、pairs/s、target tokens/s、HBM 峰值、失败率和累计卡时。记录短/中/长目标与长 query 的分位覆盖。OOM 时记录失败并停止该 batch size 的进一步放大，不自行改变模型目标或精度。
6. 审核通过的 pilot 顺序预定为：一张卡 1 对真实分数与完整 token 对照 → 10 对跨长度正确性与初步内存 → 每个 batch size 最多 512 对固定样本的两次计时（预计最多 6,000 对含热身）→ 两张卡各跑不同固定 shard 的稳态吞吐复核。每步有单独上限和完整 SHA，具体批准数写入 `ACTIVE_TASK.md`；本次实现阶段不得先跑其中任何一步。基准中不同 batch size 的分数必须逐 pair 对齐，不允许用更快但分数改变的方案估算 G4。
7. 估计全量时用 3,644 个唯一文本×8,046 CAD 的**真实长度/可评分分布**加权实测吞吐；外推包含两卡并发效率、冷启动、预处理、失败/断点与输出写盘，在 G3 的全图库代表性子集上验证，再乘至少 1.30 安全系数。若保守 ETA>168 小时，停止全量放大并向用户讨论；不启动 G4，也不以短名单代替。

## 必要测试及交接

- CPU 合成测试：真实 tokenizer 指纹、LoRA/base 路径检验的失败路径、shift/mask/结束 token、batch padding、长序列拒绝、分数与 qrels 的隔离、pair 清单固定性、预算累加和重启幂等；可以用 stub 模型测试 runner 输入与故障处理。NPU 真实数值验收留待审核 SHA 后。
- `experiments/submissions/idea01_text2cad_benchmark_20260928/IMPLEMENTATION.md` 必须写变更文件、官方来源/revision/SHA、清单传递方式、模型加载和 score 数学、全部测试命令/结果、准备的精确远端命令、HBM 风险和运行预算、已知阻塞，并以 `review_required` 结束。Codex 审查后才允许 `ACTIVE_TASK.md` 变为 `pilot_authorized`。
- 若 Kimi 官方 API 不稳定，先保存已完成文件与错误摘要到 `BLOCKERS.md`，不猜测成功、不改用其他模型。若云脑资产、环境或 Git 状态不符，保留原始错误并停止受影响步骤。
