# Idea 01 P0：Codex 代码与原始结果审核

审核对象：Kimi 提交 `4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`（实际运行代码）及 `663097011977da0377269192a249f3d672faef61`（交接报告）。运行 `idea01_p0_20260924_r01`，本地原始镜像 `experiments/runs/idea01_p0_20260924_r01/`，910C 原件 `/data/data_dev/CADRetrieval/runs/idea01_p0_20260924_r01/`。

## 裁决

**P0 数值证据可信；交接尚需补正，不能据此启动 P1 或正式检索。** 本次只证明固定 20 个验证候选上的条件评分器可运行、有有限值且对配对文本敏感；没有 gallery 排序、R@K、语义有效性或跨设计家族泛化结论。

## 我独立复核的事实

1. 在本地执行 `python -m pytest experiments/tests/ -q`：26 passed。手构造 logits 的 shift、mask、EOS 测试均通过。`scoring.py` 对目标位置 `[target_start,L)` 取预测行 `[target_start-1,L-1)`，数学口径正确。
2. 远端执行 SHA 为 `4e19259...`，`git status --porcelain` 为空；原始结果与本地镜像的旧清单逐文件比对 73/73 一致（清单自身不计）。
3. 我在远端只读脚本中重新加载 pinned 验证集、`coverage/selection.json` 和 checkpoint slow tokenizer，逐对重建实际 query、`prefix + output + EOS` 的完整输入。60/60 记录唯一且与选择表完全对应；每个目标 token ID 等于重建输入的对应位置，`pred_row=position-1`，逐 token logprob 求和与 `pairs.jsonl` 一致；错误数 0。另抽 3 条验证样本确认「完整文本编码后追加 EOS ID」与官方「完整文本拼接 EOS 字符串再编码」逐 token 相同。
4. 我独立复算 `pairs.jsonl`：60/60 状态 `ok`；correct / mismatched / empty 的平均 sum 为 -61.1928 / -119.4285 / -89.7208；20/20 候选同时满足 correct 高于另外两项；correct−mismatched 平均 58.2358，correct−empty 平均 28.5280。最长单对 0.874 秒，单对最高分配显存 6,990,295,040 bytes。只把它们作为 P0 仪器检查。
5. `audit/data_audit.json` 的唯一 UID 映射 110,263/110,263、train 两条程序差异、split 间 UID 交集 0 与我先前独立普查一致。我另用 pinned JSONL 逐字集合核查 train/val/test 两两相同 `input` 和相同 `output` 的交集，也全部为 0。此结果仍不能证明近重复或几何家族隔离。
6. `coverage/coverage_summary.json`：≤1024 完整序列 train 93,168/99,236、val 5,186/5,513、test 5,190/5,514；三组边界错位和输出内 EOS 都为 0。超长样本没有进入 P0 评分。

## 必须在 P1 前补正

1. **原始输入与环境信息不完整。** 任务合同要求保存完整模型输入 token ID，且逐对原始记录包含模型/数据完整路径与环境 spec hash。`run_p0_smoke.py` 目前只保存目标 token ID；`pairs.jsonl` 没有上述路径/hash，需结合已被第二次调用覆盖的 `environment.json` 才能推断。保留原始文件不变，用只读 CPU 派生脚本从 pinned 输入重建 60 对完整 input ID，逐 token 与原始证据交叉校验，并在独立 `review_supplement/` 输出每对补充记录、派生规则和校验汇总。修改未来运行的评分器，使新记录直接保存这些字段；多次调用的环境记录须分别保留。
2. **异常停点未严格执行。** `run_p0_smoke.py` 遇 `ScoringError`、普通异常或非有限值时会写记录却继续或以退出码 0 结束；墙钟预算到达也返回 0。实际 60 对没有触发这些路径，因此本次分数不受影响。改成记录失败样本后立即停并返回非零；`unsupported_full_sequence` 可继续计数。限制并去重 `--pair-ids`，确保不超过本任务 60 对上限。增加有意义的轻量测试。
3. **跨 split exact duplicate 审计缺项。** `data_audit.py:cross_split_overlap` 的文档声称检查 UID/input/output，实际只算 UID。补算并输出两两相同输入与相同程序的交集，保持原 P0 审计文件不变，新的 CPU 审计结果作为补充证据。当前我独立计算的两类交集均为 0。
4. **SHA 清单自引用。** 旧 `MANIFEST_SHA256.txt` 把自身列为 SHA256 空文件 `e3b0...`，因此不能对 74 个条目直接 `sha256sum -c`。旧清单保留作交付原貌；新增不自引用的修订清单，覆盖原始文件和补充证据，在本地与 910C OBS 两处逐项验证。不要更改原始 `pairs.jsonl` 或逐 token 分数。

## 解释已核清的哈希差异

准备期 `10c18c17...` 是对环境 JSON 解析后按 `sort_keys=True,separators=(',',':')` 序列化所得的 **canonical JSON SHA256**；运行记录 `b6cdd884...` 是远端 Git 文件原始字节的 **file SHA256**。二者来自同一配置内容，并非环境版本漂移。修订报告应同时标明完整哈希与计算口径，不应仅以包版本取代 spec hash。

## 下一关

上述补正经再次代码/清单复核后，P0 可以记为 `reviewed`。P1 需单独制定样本、难负例、长度/模板/词面基线、近重复筛查、预算和停止条件；本审核不授权 P1。
