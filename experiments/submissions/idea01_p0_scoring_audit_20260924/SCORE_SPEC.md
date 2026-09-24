# P0 Score Specification (`idea01_p0_scoring_audit_20260924`)

实现：`experiments/src/idea01/scoring.py` + `experiments/scripts/run_p0_smoke.py`。
本文件随运行复制到 `experiments/runs/<run_id>/score_spec.md`。

## 模型与模板（固定，不随样本改变）

- 冻结模型：`ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，BF16 权重，单张 Ascend 910 NPU 推理，`torch.no_grad()`。
- 前缀（官方 Text-to-CadQuery@4f7f5017 训练/推理格式）：
  `### Instruction:\n{input}\n\n### Response:\n`，`{input}` 为查询原文。
- tokenizer：checkpoint 自带，`use_fast=False`，`add_special_tokens=False`；EOS 以 token id 追加（不作为文本拼接）。

## 输入构造与边界

- `prefix_ids = tok(prefix)`，`full_ids = tok(prefix + output)`。
- 断言 `full_ids[:len(prefix_ids)] == prefix_ids`（前缀在 token 级是完整串的严格前缀）。不满足 → 该样本记 `boundary_error`，不评分、不截断。
- `input_ids = full_ids + [eos_id]`；`target_start = len(prefix_ids)`。
- `len(input_ids) > 1024` → 记 `unsupported_full_sequence`，不评分（官方训练也只保留 ≤1024 的完整串）。

## 分数定义

对完整串一次前向得 `logits ∈ R^{L×V}`，log-softmax 在 FP32 计算
（`logits.float()` → numpy float32 → 稳定 log-softmax）。

目标集合 = 位置 `target_start .. L-1`（response token + EOS）。shift 约定：
logits 第 j 行预测第 j+1 个 token，故取行 `target_start-1 .. L-2`。

- **主分数 `sum_logprob`** = Σ log pθ(t_i | q, t_{<i})，i 遍历目标集合。
  这是程序 token 串（含 EOS）在给定 query 下的原始条件序列 log-prob。
- `mean_logprob = sum / n_target`：仅为长度诊断，**不是**完整条件概率。
- 不剥离注释、不去 `cq.exporters` 样板：分数始终对原样 `output + EOS`。

## 掩码核对

- prompt/prefix token 不是目标（位置 0..target_start-1 只作上下文）。
- EOS 是目标集合的最后一个 token。
- padding：batch=1，无 padding。
- 手构造 logits 的 shift/mask/EOS 单测：`experiments/tests/test_scoring.py`（26 项全过）。

## 隔离

评分路径只接收 query 文本与候选 `output` 文本；UID、CSV 其他 caption、
qrels 仅出现在输出记录的簿记字段与数据审计层。
