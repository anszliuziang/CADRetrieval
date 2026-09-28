# M1a CADmium 评分器代码审查（CPU 阶段）

审查对象：本地提交 `d23dd463d7a9fea2781bdcffcddd964b65ed8fda`。这不是模型分数或检索结果，也不是 NPU pilot 批准。`ACTIVE_TASK.md` 仍为 `implementation`，`approved_sha` 仍为无，GPU/NPU 评分预算仍为 0。

## 审查结论

**CPU 评分定义与测试可留存；远端模型前向暂不准入。** Claude Code/GLM-5.2 两次调用均到达轮数上限，只留下评分器与测试初稿，未写约定的 `experiments/submissions/idea01_text2cad_benchmark_20260928/IMPLEMENTATION.md`。Codex 在本地核对原作者源码、运行失败测试、修复并提交了目前版本。此处对代码的独立审查不能替代执行者的运行命令、显存/吞吐预算和结果交接。

## 已核的关键行为

- 固定 system prompt 的实际 SHA256 与计划相同；配置正文及配置中的 SHA 字段都必须与代码固定值一致。真实 adapter tokenizer 的 chat template SHA256 为 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`，`len(tokenizer)=151665`、`vocab_size=151643`、`im_start=151644`、`im_end=eos=151645`、`pad=151643`。正式 `score_pair_from_logits` 会拒绝指纹不符的 tokenizer；测试用假 tokenizer 只调用明确命名的内部数学核心。
- prefix 是 `apply_chat_template([system,user], tokenize=True, add_generation_prompt=True)`；full 是加入原始 `json_desc` assistant 后、`add_generation_prompt=False`。要求 prefix 为 token 级真前缀。目标含模板自带 `<|im_end|>` 及后续换行 token，不再添加 EOS。模板接口缺失或出错会直接失败，不能静默换用手写模板。
- 模型传入的 `input_ids` 必须逐 token 等于构造的 full 序列；检查了 causal shift、目标 mask、非有限值、sum/mean、4096 边界和 batch 分数比较。`len(full)>4096` 拒绝，不截断。评分器函数参数不含 qrels、CSV、annotation 或 UID。
- 原作者 `cadmium/src/tokenize_dataset.py::process_batch`（Git `2b2a5e7346e17a239ffe6f4a932610bfdb9766c6`）确实把 `annotation` 放在 user、`json_desc` 放在 assistant，最大长度 4096，并 mask assistant marker 之前的标签。用真实 adapter tokenizer 对一个 CPU 样例核对：`apply_chat_template(..., tokenize=True)` 与作者“先渲染模板再 tokenizer”的 token 序列相同，末尾为 `[...,151645,198]`。
- 新评分器测试 `31 passed`；完整 `experiments/tests` 在 Windows 本地 `167 passed`（2026-09-28）。旧 `data_audit.py` 原先使用 `csv.field_size_limit(sys.maxsize)`，Windows C long 溢出；本提交改为在溢出时逐步降到可接受上限，不改变 Linux 成功路径。

## pilot 前必须补齐

1. 在本地实现并提交真实权重加载、LoRA 组合、单对模型前向、CPU/NPU 原始 logits 提取与评分的启动脚本；服务器只 `fetch`/检出获批完整 SHA 后运行。脚本必须在任何前向之前校验 tokenizer 指纹、`tokenizer.json` 文件 SHA 和模型/adapter 文件 SHA，并把实际 `input_ids` 传入公开评分入口。不能用这个内部测试核心运行正式实验。
2. 写 `IMPLEMENTATION.md`：精确版本、测试与可复现命令、独立的 7,934×8,046 上下文覆盖统计、预计最大 RAM/HBM 与 10 对吞吐试验设计。现有全库长度普查是 Codex 的初测，不可冒称 GLM 已独立复算。
3. 两片 Qwen base 权重、CADmium adapter/tokenizer 与 test parquet 已放到云脑 checkout 外运行目录且重算 SHA 通过；正式 runner 仍须校验其它小型模型配置文件、确认 M0 查询清单的独立传递，并记录所有使用文件的实际路径及 SHA。2026-09-28 `/data/data_dev` 读路径曾卡住；不可依赖它运行。
4. Codex 对新增 runner、依赖、命令和预算再次审码，在 `ACTIVE_TASK.md` 写入新的 `pilot_authorized` 与完整 `approved_sha`。在此之前，**不运行 CADmium 模型前向或检索矩阵**。

科学口径仍受 M0 审核限制：CADmium 的 8,046 个 test UID 与其 train/validation 没有精确 UID 交叉，但 Text2CAD v1.0 的 7,934 条查询有大量逐字重复；固定单正 UID 的 R@1 仅是登记配对命中率，不可直接宣称独立语义检索能力，也不可直接与 IMMA 不同过滤集的论文数字并榜。
