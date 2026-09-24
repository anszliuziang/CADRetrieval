# IMPLEMENTATION — idea01_p0_scoring_audit_20260924

状态：**review_required**。P0 实现 + 数据准入审计 + 60 对烟雾评分已完成，等待 Codex 审核代码与原始结果。

## 1. 任务与版本

- 任务合同：`experiments/plans/IDEA01_P0_IMPLEMENTATION_2026-09-24.md`（上位计划 `IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`），入口 `experiments/ACTIVE_TASK.md` state=`implementation`。
- **执行代码 Git SHA（远端实际检出运行）**：`4e19259fd7f05cdddbb95ab11ea36a723a46d2aa`（分支 `worktree-idea01-p0`，已推送 origin）。后续提交只新增报告与 `verify_p0_run.py` 只读复核脚本，不改变已执行代码。
- 模型：`ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，BF16，本机运行副本（checkout 外）。
- 数据：`ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，本机运行副本。
- 环境：`/home/ma-user/work/envs/cadretrieval-idea01-torch271`（Python 3.11.10、torch 2.7.1+cpu、torch_npu 2.7.1.post1、Transformers 4.53.1）；env spec hash 前缀 `b6cdd884`（注意：与准备期记录的 `10c18c17` 不同——本次哈希取自远端 checkout 内 `experiments/configs/idea01_yunnan_share2_env.json` 文件本身，两次哈希口径/文件版本可能不同，环境包版本已在 environment.json 实测记录）。

## 2. 改动文件（全部本地编写，Git 同步，未在服务器编辑）

| 文件 | 用途 |
| --- | --- |
| `experiments/src/idea01/prompts.py` | 官方前缀 `### Instruction:\n{input}\n\n### Response:\n`，EOS 在 token 级追加 |
| `experiments/src/idea01/scoring.py` | numpy FP32 log-softmax + shift 后 response+EOS gather；边界断言；手构造 logits 可测 |
| `experiments/src/idea01/data_audit.py` | JSONL↔CSV(`intermediate`)↔`CQ/{uid}.py` 来源/泄漏审计 |
| `experiments/src/idea01/coverage.py` | token 长度/≤1024 覆盖/边界可分/EOS 可分/样板与注释统计 |
| `experiments/src/idea01/selection.py` | 固定选择规则（uid 升序取前 20）+ 种子错配分配 |
| `experiments/scripts/run_data_audit.py` | 审计 CLI（CPU） |
| `experiments/scripts/run_coverage.py` | 覆盖 + 选择 CLI（CPU，tokenizer `use_fast=False`） |
| `experiments/scripts/run_p0_smoke.py` | BF16 NPU 评分 CLI，逐对/逐 token 证据、显存/耗时/预算护栏 |
| `experiments/scripts/verify_p0_run.py` | 结果复核：从逐 token 文件重算 sum、校验边界与 EOS（结果提交后新增，只读） |
| `experiments/tests/test_scoring.py`、`test_selection_audit_coverage.py` | 26 项单测 |
| `experiments/configs/idea01_p0_scoring_audit_20260924.json` | 任务配置（路径、精度、种子、上限） |
| 本目录 `SCORE_SPEC.md` | 分数数学定义与掩码口径 |

## 3. 实现方法与关键假设（与官方代码对应）

- 官方 `Text-to-CadQuery@4f7f5017` 训练串为 `prefix + output + EOS`、≤1024 token、因果 LM 整串训练；本项目只把 **response+EOS** 作为目标集合求和，prefix 仅作条件上下文。shift 约定：logits 第 j 行预测第 j+1 token；目标行取 `target_start-1 .. L-2`。
- tokenizer：`AutoTokenizer.from_pretrained(model_dir, use_fast=False)` → `Qwen2Tokenizer`，`add_special_tokens=False`，`eos_id=151645`（`<|im_end|>`）。
- 前缀边界：逐样本断言 `tok(prefix)` 是 `tok(prefix+output)` 的严格前缀；不满足即记 `boundary_error` 并跳过（本次实测 0 例）。
- log-softmax 在 FP32：torch logits `.float().cpu().numpy()` 后 numpy 计算；sum 用 float64 累加。
- 隔离：评分路径只接收 query 文本与候选 `output`；UID 仅用于输出簿记与审计层。

## 4. 测试证据（本地，Windows，Python 3.14 + numpy 2.5.2）

- `python -m pytest experiments/tests/ -q` → **26 passed**。
  - 手构造 logits：one-hot 目标下 sum≈0；off-by-one 扰动后 sum 显著恶化；uniform logits sum = -3·ln V；EOS 在目标集合内、prompt 不在；非法 `target_start`/形状抛 `ScoringError`；NaN 检出。
  - 选择规则：合格行按 uid 升序取前 N、排除超长/未映射/边界错；同种子可复现；错配 ≠ 自身；每候选 3 条件。
  - 审计：合成 CSV/ZIP 的 unique/ambiguous/missing、different 记录、跨 split UID 交集。
- 本地数据审计彩排（本地 `.cache` 副本，CPU）：110,263/110,263 唯一映射，train 2 条程序例外（`0000/00000102`、`0000/00000144`），与 Codex 初测一致；跨 split UID 交集全 0。
- 真实样例人工复核 2 例（c00_correct、c01_correct 的逐 token 表）：首目标 = 程序首 token `import`，位置 = prefix 长度；末目标 = `<|im_end|>`；`pred_row = position-1` 全成立。

## 5. 远端执行命令（yunnan-share2，全部在 checkout 外写输出）

```bash
# 检出（工作树预先确认为空）
cd /home/ma-user/work/CADRetrieval && GIT_TERMINAL_PROMPT=0 git fetch origin && \
  git checkout --detach 4e19259fd7f05cdddbb95ab11ea36a723a46d2aa
# 资产：OBS 与本机副本各 sha256sum -c SHA256SUMS → 11/11、6/6、11/11、6/6 全 OK
# 环境变量：PYTHONDONTWRITEBYTECODE=1 HF_HOME/XDG_CACHE_HOME/TMPDIR 指向 CADRetrieval_runtime 下
RUN=/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p0_20260924_r01
python experiments/scripts/run_data_audit.py --data-dir <数据运行副本> --out-dir $RUN/audit
python experiments/scripts/run_coverage.py  --data-dir <数据运行副本> --model-dir <模型运行副本> \
  --audit-dir $RUN/audit --out-dir $RUN/coverage --splits val,test,train --max-len 1024 --n-candidates 20 --seed 20260924
python experiments/scripts/run_p0_smoke.py --config experiments/configs/idea01_p0_scoring_audit_20260924.json \
  --selection $RUN/coverage/selection.json --out-dir $RUN/smoke --pair-ids "0"    --device npu:0
python experiments/scripts/run_p0_smoke.py ... --pair-ids "1-59" --device npu:0
```

运行后 `git status --porcelain` 为空；原始输出另存 OBS `/data/data_dev/CADRetrieval/runs/idea01_p0_20260924_r01/`（74 文件，`MANIFEST_SHA256.txt`），本地镜像 `experiments/runs/idea01_p0_20260924_r01/` 已逐文件对远端清单复核（73/73 一致，清单自身除外）。

## 6. 资源实测（远低于 1 卡小时上限）

- NPU 占用：两次调用各含模型加载约 19 s；60 次前向合计 **17.0 s**（单对 0.18–0.87 s）；NPU 总占用 ≈ 96 s ≈ **0.027 卡小时**。
- 峰值显存：单对最大 **6.99 GB**（BF16，最长 745 token 输入）。
- CPU 侧：CSV 索引 + ZIP 逐条核对约数分钟；三分词覆盖（110,263 行，slow tokenizer）约 7 分钟。

## 7. 已知缺口（详见 BLOCKERS.md / RESULTS.md）

- UID ≠ 设计家族；近重复/几何等价未量化；数据仓库 license 元数据为空（仅内部研究用途，不公开再分发）。
- `<|im_end|>` 在三种条件下 logprob 均约 -21（模型很少主动结束程序），占 sum 比重不小；已在 RESULTS.md 分析其对边际的影响。
- `environment.json` 为第二次调用（pair 1–59）所覆盖写入；两次调用仅 `--pair-ids` 不同，SHA/环境一致。
