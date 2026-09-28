# M1a：CADmium 原生 JSON 条件评分器（本地 CPU 实现合同）

本合同从 [Text2CAD 总计划](IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md)细化 M1 的第一支。当前 `ACTIVE_TASK.md` 的 GPU/NPU 预算为 0；这次只允许本地代码、tokenizer CPU 核验和合成 logits 测试。Codex 审查源码并批准完整 SHA 后，才能另写 1 对烟测及 10 对吞吐预算。

## 固定资产和输入

- 模型：`chandar-lab/CADmium-3B@6c6a79d160595207804171be56a037883b369abd` 的 LoRA adapter，base `Qwen/Qwen2.5-Coder-3B-Instruct@488639f1ff808d1d3d0ba301aef8c11461451ec5`。候选目标取 `chandar-lab/CADmium-ds@ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4` 的 test parquet `json_desc`，不要用其 `annotation` 作为 query。实际 query 只取已审 `.cache/m0_real_run_v3/query.jsonl` 的 `text`。
- 原作者 `cadmium/src/tokenize_dataset.py::process_batch` 构建 system/user/assistant chat；目标为原始 `json_desc` 字符串，官方训练最大长度 4,096。system 内容的准确副本在 `experiments/configs/cadmium_system_message.json`，来自原作者 Git `2b2a5e7346e17a239ffe6f4a932610bfdb9766c6`，文本 SHA256 `59d623d0c7ca29810eb80ab265891023c749402516701caade11e96a62a18ab3`。不能手工简写 schema 或用候选配对 annotation 充当 query。
- 输入隔离：构造一对 token 序列的函数仅接收 `query_text`、`candidate_json_desc`、固定 system prompt 与 tokenizer。它不能打开 qrels、audit、原始 CSV 或 CADmium annotation；UID 只能由外层日志/评价模块保存。

## 评分定义及拒绝条件

1. 用发布的 adapter tokenizer（已核验与 base 的 chat template、词表长度和特殊 token ID 一致）分别构造 `prefix_ids = apply_chat_template([system,user], tokenize=True, add_generation_prompt=True)` 和 `full_ids = apply_chat_template([system,user,assistant=json_desc], tokenize=True, add_generation_prompt=False)`，要求前者严格等于后者的 token 前缀。正式目标是 `full_ids[len(prefix_ids):]`，包含作者模板的 `<|im_end|>` **及其后换行 token**；不得另附一个 EOS。缺失/空目标、前缀不匹配、tokenizer 版本不符则拒绝并记账。
2. `len(full_ids) > 4096` 时标为 `out_of_training_context`，不得截断后声称完整条件分数。Codex 对全部 7,934 个 query 前缀与 8,046 个候选目标做 CPU 长度普查：前缀 665–761 tokens；目标尾部 182–3446；22,585/63,836,964 对超过 4096，涉及 3 个 CAD、其中 1 个对所有 query 超长。实现者须独立复核确切组合总数和不可评分对数，不要硬编码普查数字。
3. 用冻结模型原始 logits 的 causal shift 取目标每个 token 的 `log_softmax`；`S_sum = Σ log p(y_i | q,y_<i)` 为主分数，`S_mean` 与有效 token 数仅诊断。可复用 `experiments/src/idea01/scoring.py::gather_response_logprobs`，但**不能**调用该文件为 CadQuery 设计的 `build_scored_sequence`（它会额外加 EOS）。运行时 padding 只用于批处理，padding token 不计分；batch=1 与 batch>1 对同一 pair 的分数应相同。
4. CPU 测试覆盖模板前缀、结束 token/换行、shift、mask、非有限值、4096 边界、批量一致性、目标 JSON 原样保存、不读取 qrels。真实 tokenizer 可用本地 `.cache/cadmium_model/` 与 `transformers 4.53.1`，无须加载权重或执行模型前向。外部模型源码与本地 `.cache` 不能作为服务器运行时唯一代码依赖；项目源码/配置须进 Git。

## 交付和停止

在 `experiments/src/idea01/`、`experiments/tests/` 编写评分器与测试，在 `experiments/submissions/idea01_text2cad_benchmark_20260928/IMPLEMENTATION.md` 写清输入输出、prompt/tokenizer hash、覆盖、测试命令、未决训练配置与预计运行成本。写 `review_required` 并停止。不得运行任何新模型前向、NPU/CUDA 烟测、检索矩阵或长作业。
