# RESULTS — idea01_p0_scoring_audit_20260924

run_id：`idea01_p0_20260924_r01`；执行代码 SHA `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`；单张 Ascend 910（npu:0），BF16，log-softmax FP32。
路径：远端 `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p0_20260924_r01/`；OBS `/data/data_dev/CADRetrieval/runs/idea01_p0_20260924_r01/`；本地镜像 `experiments/runs/idea01_p0_20260924_r01/`（含 `MANIFEST_SHA256.txt`，本地 73/73 文件与远端一致）。

## A. 事实（可直接复核）

### A1. 来源/泄漏审计（`audit/data_audit.json`，本地彩排结果完全相同）

- CSV 176,017 行，`intermediate` 键 176,017 个、**无歧义键**。
- 映射（verbatim join）：train 99,236/99,236、val 5,513/5,513、test 5,514/5,514 唯一映射；缺失/歧义均为 0。
- 程序一致性（`CQ/{uid}.py` vs `output`）：val 5,513/5,513 逐字一致；test 5,514/5,514 一致；train 99,234/99,236 一致，例外 2 条：`0000/00000102`、`0000/00000144`（JSONL 版本多于 ZIP 版本，含额外注释）。
- split 内 exact duplicate input/output：三个 split 均为 0；split 间 UID 交集：train–val、train–test、val–test 全为 0。
- 独立复算与 Codex 只读初测完全一致（映射数、2 条例外 UID 均吻合）。

### A2. 目标覆盖（`coverage/coverage_summary.json`，checkpoint tokenizer use_fast=False）

| split | 行数 | ≤1024 覆盖 | 边界不一致 | EOS 混入 output | 长度 min/p50/p90/max | 含导出样板 | 平均注释行比 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| val | 5,513 | 5,186（94.07%） | 0 | 0 | 228/402/856/2892 | 99.87% | 0.251 |
| test | 5,514 | 5,190（94.12%） | 0 | 0 | 223/398/856/3092 | 99.93% | 0.251 |
| train | 99,236 | 93,168（93.89%） | 0 | 0 | 208/398/866/3236 | 99.90% | 0.252 |

- tokenizer 对超长样本发出 model_max_length=1024 的信息性警告（与官方训练 ≤1024 过滤一致）；本实现不截断，>1024 样本只标记 `unsupported_full_sequence`。
- 逐行数据在 `coverage/coverage_<split>.jsonl`。

### A3. P0 选择与烟雾评分（`smoke/pairs.jsonl`、`smoke/tokens/*.jsonl`）

- 选择：val 中 uid 唯一映射 + ≤1024 + 边界 OK 的行按 uid 升序取前 20（uid `0000/00000257`…`0000/00006466`，token 长 272–745）；错配 query 由 `random.Random(20260924)` 从其余 19 个候选中抽取（配对表见 `coverage/selection.json`）。
- 60/60 对评分成功，`status=ok`、全部有限；0 失败、0 边界错误、0 超长（选择阶段已排除）。
- 逐对 sum（response+EOS 条件 log-prob）：

| 条件 | mean sum | 说明 |
| --- | ---: | --- |
| correct | -61.19 | 同候选三条件的 response token 完全相同，只有 query 不同 |
| empty | -89.72 | |
| mismatched | -119.43 | |

- **20/20 候选同时满足 correct > mismatched 且 correct > empty**。边际（correct−mismatched）mean 58.24（min 25.60，max 84.07）；（correct−empty）mean 28.53（min 14.36，max 47.52）。
- 复核：`verify_p0_run.py` 从 60 个逐 token 文件重算 sum，与 `pairs.jsonl` 最大偏差 5.7e-14；所有 pair 首目标位置 = prefix 长度、末目标 = EOS(151645)、`pred_row=position-1`。
- EOS 诊断：`<|im_end|>` 单 token logprob 在 correct/empty/mismatched 下均值分别 -21.34 / -20.94 / -20.85（范围约 -22.5…-19.6）。
- 环境/耗时：torch 2.7.1+cpu、torch_npu 2.7.1.post1、Transformers 4.53.1；单对前向 0.18–0.87 s，峰值分配 ≤6.99 GB；NPU 总占用 ≈96 s（含两次模型加载各约 19 s），约为预算 1 卡小时的 2.7%。
- 停止线：未触及任何停止条件（预算、非有限值、边界歧义、哈希失败、源码树污染均未发生；运行前后远端 `git status --porcelain` 均为空）。

## B. 分析（我的解释，非事实本身）

1. **条件敏感性方向正确且相当强**：同一候选、同一 response token 序列下，正确 query 的 sum 系统地高于错配（平均高 58 nats）和空 query（高 29 nats）。由于三条件共享 response，长度/模板/注释等 response 侧捷径在候选内完全抵消，边际只能来自 query 条件——这正是 P0 想确认的仪器性质。按合同，这不是检索结论（无 R@K、20 个 uid 升序候选非随机抽样、错配来自不同 UID 而非受控难负例）。
2. **EOS 近乎恒定地被强烈惩罚**（约 -21 nats，等价于模型几乎不预测程序结束）。三种条件下 EOS logprob 相近（correct 反而略低 0.4–0.5 nats），所以它轻微压缩而非放大 correct 的边际；但任一固定约 -21 的加性常数都占 correct 平均 sum 的 1/3。若 P1+ 使用 sum 作分数，建议把 EOS 项单列诊断（当前逐 token 证据已支持直接重算）；我不建议现在改主分数定义。
3. **empty 介于 correct 与 mismatched 之间**符合直觉：空 query 退化为无条件程序先验，错配 query 则提供了主动不符的条件。empty−mismatched 的间隔（约 30 nats）提示 query 内容本身（而非"有没有 query"）贡献了大部分边际。
4. **覆盖与样板**：≤1024 覆盖约 94%，三 split 一致；>99.8% 程序含导出样板、注释行比约 0.25。高样板比例意味着 sum 中有相当部分来自模板化 token（逐 token 表显示大量近 0 logprob 的固定写法），P1 的基线比较必须包含查询无关的长度/模板项。
5. **吞吐外推**：实测 BF16 单对约 0.2–0.9 s（随 token 数增长），P1 的 512 对规模预计 <10 分钟单卡（含一次加载），P2 的 256×256=65,536 对按 p50 长度粗估约 5–8 卡小时，需另行预算审核。

## C. 建议（需 Codex 裁决，非授权）

1. P0 停止线全部通过，建议 Codex 审核本代码（重点：`scoring.py` 的 shift/mask、`run_p0_smoke.py` 的隔离与记录）与原始逐 token 证据后，按上位计划裁决是否进入 P1（32 query × 8–16 候选，含长度/模板/词面基线）。
2. P1 设计建议：错配 query 改为同 split 内受控难负例；把 EOS logprob、近 0 logprob 样板 token 占比作为标配诊断列；随机抽样候选（固定种子）而非 uid 升序。
3. 数据准入口径：UID 链已闭合可做配对 sanity；family/近重复分组仍是开口项，建议 P1 前至少用程序文本近重复（如 token 级 Jaccard 或归一化编辑距离粗筛）量化 val 内候选间相似度。
4. 数据仓库 license 字段为空：继续内部研究可用、禁止公开再分发的口径维持不变。

## D. 本轮新增提交

- `4e19259`：全部执行代码、测试、配置、SCORE_SPEC（远端实际运行 SHA）。
- 结果提交提交（本文件及 `verify_p0_run.py`）：只读复核脚本与报告，SHA 见最终 push 记录；不影响已执行的 4e19259。
