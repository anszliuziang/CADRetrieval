# RESULTS — idea01_p1_conditional_sensitivity_20260925 / 阶段 1（第 0 号配对 1 次真实评分）

状态：**等待 Codex 阶段 1 复审**。仅执行了已获授权的阶段 1：`--pair-ids "0"` 一次 NPU 前向。**未运行 `--resume`、`0-9`、`all`，未计算任何排名/R@K，未触碰 P2/P3。** 下一阶段（累计 10 对）未获授权，本文件不含任何自行扩跑内容。

## 1. 授权与版本（事实）

- 授权文档：origin/master `a3de6f0c163d7aa72c88df5049c8c042fc9bd212` 的 `experiments/ACTIVE_TASK.md`（`pilot_authorized`，仅阶段 1；runner 运行时再 fetch 校验，environment 记录该 blob sha256 `903c96aeb1f94f48888bcd2d4810fbc520c11f62d464a7d86a8e90877e17bd84`）。
- **执行代码 SHA**：`52ad27f2da5070e3f4a3ab6d66a11f79bd514b3c`（服务器 `checkout --detach` 后 `git rev-parse HEAD` 一致；运行前后 `git status --porcelain` 均为空）。
- **选择清单**：`/home/ma-user/work/CADRetrieval_runtime/plans/idea01_p1_20260925/p1_selection.json`，sha256 `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964`（与批准值一致；runner preflight 同时验证 `approved_selection_sha256`）。
- **资产**：评分前先在服务器比对两份 `SHA256SUMS` 文件哈希——模型 `98c1c44b9052218a79f48e2c5f7879b579031981048612c23304bfdc9fe2c785`、数据 `f75a7a48c9c97fe6933f8b10dbe13e779da95fbeeab88bb1a612104a5f57fd34`，均与 `ACTIVE_TASK` 批准值一致（P1 计划 2026-09-25 替代路径；**未访问卡住的 `/data/data_dev`**）。runner 载入权重前重新逐项核验运行副本：**模型 11/11、数据 6/6 通过**。
- 环境：`/home/ma-user/work/envs/cadretrieval-idea01-torch271`（torch 2.7.1+cpu、torch_npu 2.7.1.post1、transformers 4.53.1）；设备 `npu:0`，BF16；env spec 双哈希 canonical `10c18c17…`、git-blob `b6cdd884…`（与 P0 相同）。运行前 `npu-smi` 确认卡上无进程。
- 执行窗口：2026-09-25 00:16 UTC，外层 `timeout 600` 未触发；进程墙钟 36.81 s。

## 2. 执行命令（事实）

```bash
cd /home/ma-user/work/CADRetrieval
git status --porcelain                       # 空（干净）
GIT_TERMINAL_PROMPT=0 git fetch origin       # origin/master = a3de6f0
git checkout --detach 52ad27f2da5070e3f4a3ab6d66a11f79bd514b3c
git rev-parse HEAD                           # 与批准 SHA 一致
sha256sum <模型>/SHA256SUMS <数据>/SHA256SUMS   # 与批准的两个清单文件哈希一致
npu-smi info                                 # 无运行进程
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=.../hf_cache XDG_CACHE_HOME=.../xdg_cache TMPDIR=.../tmp
RUN=/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01
timeout 600 /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_p1_matrix.py \
  --config experiments/configs/idea01_p1_conditional_sensitivity_20260925.json \
  --selection .../plans/idea01_p1_20260925/p1_selection.json \
  --out-dir $RUN --pair-ids "0" --device npu:0 | tee $RUN/stage1_console.log
# MATRIX-EXIT=0
```

## 3. 原始结果（事实）

产物（服务器 checkout 外 `$RUN`；本地镜像 `experiments/runs/idea01_p1_20260925_r01/`，5/5 文件 sha256 与服务器一致）：

| 文件 | sha256 |
| --- | --- |
| `pairs.jsonl` | `ed0da343556e41b3c28784cf3f910f20ab36c1c1bb4267fa409a02f6e921fee8` |
| `tokens/q00_0_self.jsonl` | `d3abdb1804aa9838c4c1a4ad7575ba7fe5352165456b2931f26b506270e00cb3` |
| `budget_ledger.json` | `a384bbbdb6a74a7a12e20fd49040726dc68a6a6ed715511f1857151a92c32e3a` |
| `environment_20260925T001609Z-e541c97f.json` | `1d76a7ebcb63332f5b9d8e7819c8f754c6b361779875df570007a771f7c3fe14` |
| `stage1_console.log` | `87a19a653b5024e6ece95c07c9bd1e9ac1cc22cc0586c85166d582d36793afd9` |

逐对记录（`q00_0_self`，conditional self；query_uid=candidate_uid=`0054/00549200`，row 5136，仅簿记）：

- `status=ok`、`all_finite=true`；`sum_logprob=-40.3757508507636`、`mean_logprob=-0.18269570520707512`、`eos_logprob=-21.363788604736328`
- `n_prefix_tokens=140`、`n_input_tokens=361`、`n_target_tokens=221`；`input_ids` 全长 361、末位 `151645`（`<|im_end|>`）——**与 CPU 门 `p1_pair_tokenization.jsonl` 对该配对的预估值逐项一致**（140/361/221）
- `elapsed_s=0.868`；`peak_npu_alloc_bytes=6,640,821,248`（6.64 GB）
- 记录绑定：`git_sha=52ad27f…`、`selection_sha256=c86fb569…`、模型/数据 revision 与完整路径、双环境哈希、`device=npu:0`、`dtype=bfloat16`、`invocation_id=20260925T001609Z-e541c97f`

逐 token 证据（221 条）：首目标 `position=140, pred_row=139, token="import", logprob≈-1.36e-05`；末目标 `position=360, pred_row=359, token=<|im_end|>, logprob=-21.3638`；全部 `pred_row=position-1`。本地用 `p1_verify.verify_pair_record` 复核：**0 错误**（token 和↔sum、mean、EOS、位置连续、shift、`token_id==input_ids[position]` 全部通过）。注：全量 `verify_p1_run.py` 的 414 覆盖检查按设计在阶段 1 会报 413 missing——它只适用于完整 run；单对一致性已通过。

预算台账（`budget_ledger.json`）：本次调用 `forwards=1`、`wall_s=36.81`、exit 0、已收尾；**累计 1 前向 / 36.81 s（0.0102 卡小时）**——阶段 1 预算（0.25 卡小时、10 分钟）与全程硬上限（416/45min/1卡时）均远未触及。

失败与偏差：**无**。无哈希失败、无边界错误、无非有限值、无 OOM、无超时、无预算触发；外层 timeout 未触发；`/data/data_dev` 未访问；服务器源码树未编辑。

## 4. 解释（我的分析，非结论）

1. **修复后评分链路端到端真实可运行**：授权绑定（origin/master fetch + approved_sha/清单 SHA）、资产复核、权重加载、shift/EOS 口径、逐 token 落盘、台账累计在真实 1 对上全部按设计工作；CPU 门 tokenize 与 NPU runner 的 tokenize 在该对上逐字段一致（140/361/221），说明两阶段 tokenizer 行为无漂移。
2. **分数形态与 P0 一致**：EOS 一项 logprob -21.36，占 sum（-40.38）的 52.9%；非 EOS 部分 220 token 合计约 -19.01（均值 ≈ -0.086）。次差目标均为数字字符（"4" -2.57、"0" -2.33、"2" -2.32、"9" -2.25，集中在程序末尾的 exporters 样板数字）。EOS 支配效应意味着后续 414 对的边际/排名解读必须把 EOS logprob 分层报告（分析模块已含该列）。
3. **成本外推（供阶段 2 参考，非授权）**：单对 0.868 s（最长 697 token 的对预计 <1.5 s）；414 对前向约 6–10 分钟；每次调用另有授权 fetch + 资产复核约 30 s 与加载约 3.4 s。三阶段（1→+9→+404）总墙钟预计 <15 分钟、<0.3 卡小时，在全程预算内。
4. 阶段 1 单对**不构成任何排序/敏感性证据**；其价值仅为仪器验证。

## 5. 建议（供 Codex 决定，非授权）

- 若阶段 1 复审通过，建议按已批准框架授权阶段 2：`--pair-ids "0-9" --resume`（新增 9 对，验证 `--resume` 跳过逻辑在真实文件上的行为），随后再授权阶段 3 的 `all --resume`。
- `--resume` 在真实已有记录上的逐字段绑定核验（R1）在阶段 2 会首次真实触发；建议 Codex 在阶段 2 前后比对 `pairs.jsonl` 仍恰为 1/10 条唯一记录。
- 910C `/data/data_dev` 挂载恢复前，继续按 P1 计划替代路径归档；本阶段产物待 Codex 指示后转存共享 OBS。

---

# 阶段 2 追加记录（2026-09-25，`--pair-ids "0-9" --resume` 新增 9 对、累计 10 对）

状态：**等待 Codex 阶段 2 复审**。仅执行了 db0f944 授权的内容：同一 run 目录追加 9 对。**未运行 `all`、其他任何前向或排名分析；不进入全量。**

## 1. 授权与前置核验（事实）

- 授权：origin/master `db0f9446a6f29c39f05729a9a07fa7b2fa1c3c30` 的 `ACTIVE_TASK.md`（阶段 2，`pilot_authorized`；阶段 1+2 累计 ≤0.25 卡小时/10 分钟）。runner 运行时再次 fetch 校验，阶段 2 environment 记录授权 blob sha256 `e88b8cd2060b685092608e75eae6434463ca2240729933c96d111445b03ffd65`、`origin_ref_sha=db0f9446…`、`fetch_returncode=0`、problems 空。
- 运行前逐项核验通过：Git 工作树干净；`checkout --detach 52ad27f2…` 后 `git rev-parse HEAD` 一致；模型/数据 `SHA256SUMS` 文件哈希 `98c1c44b…`/`f75a7a48…` 与批准值一致；选择清单 sha256 `c86fb569…` 一致；`npu-smi` 无进程；**未访问 `/data/data_dev`**。
- 阶段 1 原始字节（运行前记录）：`pairs.jsonl` 1 行/3240 字节/sha `ed0da343…`；`tokens/q00_0_self.jsonl` sha `d3abdb18…`。

## 2. 执行命令（事实）

```bash
RUN=/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01
timeout 600 /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_p1_matrix.py \
  --config experiments/configs/idea01_p1_conditional_sensitivity_20260925.json \
  --selection .../plans/idea01_p1_20260925/p1_selection.json \
  --out-dir $RUN --pair-ids "0-9" --resume --device npu:0 | tee $RUN/stage2_console.log
# MATRIX-EXIT=0；preflight 资产复核模型 11/11、数据 6/6；载入权重前通过授权与已有记录绑定核验
```

## 3. 结果与运行后核验（事实）

- **resume 跳过证据**：控制台日志含 `[p1-matrix] resume: skipping verified-completed pair q00_0_self`；preflight 报 `9 new forwards planned (1 completed skipped)`。
- **阶段 1 原始字节不变**：运行后 `head -c 3240 pairs.jsonl` 的 sha256 仍为 `ed0da343…`（服务器端）；本地复核首行+换行与阶段 1 整文件哈希一致；`tokens/q00_0_self.jsonl` sha 仍为 `d3abdb18…`。
- **pairs.jsonl 恰 10 个不同 pair_id**：`q00_0_self, q00_1_off, q00_2_off, q00_3_off, q00_4_method, q00_5_method, q00_6_length, q00_7_length, q01_0_self, q01_1_off`；全部 `status=ok`、`all_finite=true`；两条 invocation 记录（`…e541c97f`、`…80042d56`）。
- **台账**：累计 **10 前向 / 75.42 s（0.0209 卡小时）**；两次调用均 `finalized: true, exit 0`（36.81 s + 38.61 s）——阶段 1+2 累计预算（0.25 卡小时/600 s）与全程硬上限均未触及。
- **逐 token 核验**：本地 `p1_verify.verify_pair_record` 对全部 10 对复核 **10/10 通过**（token 和↔sum、mean、EOS、位置连续、shift、`token_id==input_ids[position]`）。
- **镜像一致性**：本地 `experiments/runs/idea01_p1_20260925_r01/` 与服务器逐文件 sha256 **14/14 一致**（pairs.jsonl `b625f987…`、10 个 token 文件、ledger `8cbda253…`、stage-2 environment `970d2374…`、stage2_console.log `9f0dd864…`）。新增 environment 文件名独立（`…T002610Z-80042d56`），未覆盖阶段 1 文件。
- 运行后服务器 `git status --porcelain` 为空；失败/偏差：**无**（无哈希失败、非有限、OOM、超时、预算触发）。

原始分数（仅逐对事实，**未计算名次/胜率/R@K**）：

| pair_id | 来源 | n_target | sum | mean | eos_logprob | 峰值显存 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q00_0_self | self（阶段 1） | 221 | -40.376 | -0.1827 | -21.364 | 6.64 GB | 0.868 s |
| q00_1_off | selected-offset | 261 | -105.401 | -0.4038 | -20.057 | 6.68 GB | 0.973 s |
| q00_2_off | selected-offset | 232 | -85.966 | -0.3705 | -20.243 | 6.65 GB | 0.220 s |
| q00_3_off | selected-offset | 409 | -174.239 | -0.4260 | -21.434 | 6.81 GB | 0.362 s |
| q00_4_method | method-neighbor | 221 | -60.983 | -0.2759 | -21.195 | 6.64 GB | 0.213 s |
| q00_5_method | method-neighbor | 221 | -68.615 | -0.3105 | -20.118 | 6.64 GB | 0.210 s |
| q00_6_length | length-neighbor | 221 | -120.884 | -0.5470 | -20.140 | 6.64 GB | 0.209 s |
| q00_7_length | length-neighbor | 221 | -129.609 | -0.5865 | -21.046 | 6.64 GB | 0.210 s |
| q01_0_self | self | 243 | -44.337 | -0.1825 | -18.964 | 6.64 GB | 0.230 s |
| q01_1_off | selected-offset | 399 | -184.529 | -0.4625 | -21.988 | 6.78 GB | 0.347 s |

## 4. 解释（我的分析，非结论）

1. **R1 的 resume/累计机制在真实文件上首次按设计工作**：已有记录绑定核验通过 → 显式跳过 → 只追加 → 累计 10 对恰好 10 条唯一记录；阶段 1 字节零改动；台账两次调用独立收尾。这支持阶段 3（`all --resume`，新增 404 对）的同一机制，但是否执行取决于 Codex。
2. **耗时结构**：9 对前向合计约 2.99 s（0.21–0.97 s/对，随 token 数增长），模型加载 3.4 s，每次调用固定开销（授权 fetch + 资产复核）约 30 s。按此外推全量 414 对：前向约 2–4 分钟 + 固定开销，总墙钟远低于 45 分钟/1 卡小时上限。
3. **分数形态**：与 P0/阶段 1 一致，EOS logprob 全部在 -19~-22 区间，占各对 sum 比重很大（如 q00_0_self 的 52.9%）；这些只是仪器一致性事实。self 对（-40.4、-44.3）与其余 8 对的关系、候选间边际等**留待 Codex 授权的分析阶段**，此处不做任何排序解读。
4. 单对峰值显存 6.64–6.81 GB，随输入长度温和增长；最长计划配对 697 token 预计仍 ≪ 65 GB HBM。

## 5. 建议（供 Codex 决定，非授权）

- 若阶段 2 复审通过，建议授权阶段 3：`--pair-ids "all" --resume`（新增 404 对，累计 414），完成后先 `verify_p1_run.py` 全量验证再 `run_p1_analysis.py`。
- 建议阶段 3 授权前 Codex 抽查本阶段任一新增对的 `input_ids` 与 pinned JSONL 重建一致性（与阶段 1 复核方式相同）。

---

# 阶段 3 追加记录（2026-09-25，`--pair-ids "all" --resume` 新增 404 对、累计 414 对 + 全量验证 + 预定分析）

状态：**等待 Codex 阶段 3 复审**。仅执行 9390e7c 授权内容。分析严格在验证器 414/414 通过之后、按固定脚本与固定参数运行；未根据分数改任何筛选/阈值/候选/脚本。**未启动 P2/P3，未改代码/配置/选择清单/模型/数据。**

## 1. 授权与前置核验（事实）

- 授权：origin/master `9390e7cfac8a09f05ce7319d02cf7966bf71994b` 的 `ACTIVE_TASK.md`（阶段 3；全程 416 前向/45 分钟/1 卡小时，外层 `timeout 1200`）。runner 再 fetch 校验：environment（`environment_20260925T003722Z-d88fff78.json`）记录 `origin_ref_sha=9390e7cf…`、blob sha256 `66c83a3d55a8d18a1fcccfe7afa802911a6eeeaee496f69ffea62264893ac642`、fetch rc 0、problems 空。
- 运行前核验：Git 干净；HEAD=`52ad27f2…`；模型/数据 `SHA256SUMS` 文件哈希 `98c1c44b…`/`f75a7a48…` 一致；选择清单 `c86fb569…` 一致；NPU 无进程；**未访问 `/data/data_dev`**。
- **resume 审计（运行前记录）**：10 对原始字节——`pairs.jsonl` 10 行/34,167 字节/`b625f987…`、10 个 token 文件（`d3abdb18…` 等）、ledger `8cbda253…`。

## 2. 执行与运行后核验（事实）

命令（唯一一次评分调用）：`timeout 1200 ... run_p1_matrix.py --config $CFG --selection $SEL --out-dir $RUN --pair-ids "all" --resume --device npu:0`，`MATRIX-EXIT=0`。

- resume：明确跳过已有 10 对，**新增 404 次前向**；全部 `status=ok`、`all_finite=true`。
- **10 对原始字节不变**：运行后 `head -c 34167 pairs.jsonl` sha 仍 `b625f987…`；抽查 token 文件未变。
- **pairs.jsonl 恰 414 行、414 个不同 pair_id**；台账：累计 **414 前向 / 217.19 s（0.0603 卡小时）**，三次调用均 finalized/exit 0（36.81 + 38.61 + 141.77 s）；三个全程硬上限未触及；timeout 未触发；运行后 Git 树干净。
- **全量验证（分析的必要前提）**：服务器 `verify_p1_run.py --run-dir $RUN --selection $SEL` → **records=414 expected=414 failures=0，exit 0**（逐 token 和↔sum、mean、EOS、位置连续、shift、`token_id==input_ids[position]`、覆盖无重复/缺失/多余）。本地镜像独立复核 `verify_pair_record` **414/414 通过**。
- 镜像与归档：本地 `experiments/runs/idea01_p1_20260925_r01/`（430 文件：414 token + pairs/ledger/3 environment/5 log/3 分析产物/2 审计清单）与服务器 sha256 **422/422 一致**；本地生成不自引用 `MANIFEST_SHA256.txt`（429 条）后经 cvteam14 转存共享 OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_p1_20260925_r01/stage3_full_mirror/`，挂载上 `sha256sum -c` **429/429 通过**（stage1/stage2 快照未触碰；staging 已清理）。

## 3. 预定分析结果（事实；固定脚本、固定 seed 20260925、bootstrap 10,000 次，分析总耗时 4.66 s）

**主诊断（同候选边际，32 个被选中程序各自 own vs 3 个偏移 query，raw sum）**：
- **32/32 正边际**；均值 **60.138**（95% CI **[56.450, 64.190]**，候选 UID 重抽样）；中位数 58.018（CI [52.092, 65.569]）；最小 46.48、最大 91.84。

**次诊断（8 候选 exact-UID 配对 sanity；非语义检索 R@K）**：

| 分数臂 | mean_rank | top-1（严格） | 95% CI | 逐对胜率 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| raw sum | 1.47 | 0.781 | [0.625, 0.906] | 0.933 | [0.871, 0.978] |
| mean（长度诊断） | 1.22 | 0.875 | [0.750, 0.969] | 0.969 | [0.933, 0.996] |
| corrected=S(q,x)−S(∅,x)（候选先验校正诊断） | 1.06 | 0.938 | [0.844, 1.000] | 0.991 | [0.978, 1.000] |

**基线（同一 8 候选集合）**：固定种子随机 top-1 0.156 / 胜率 0.464；response 长度升序 0.000 / 0.384；空 query 分数 0.062 / 0.545；字符 3-gram TF-IDF（IDF 仅本轮 158 个候选程序拟合）0.375 / 0.710。三个分数臂全部显著高于全部基线。

**分层诊断（描述性）**：正例 response 较长者边际更大（短组均值 52.72 vs 长组 67.55）；注释比例较低者边际更大（64.77 vs 55.51）；方法相似度分层退化（32 个 query 的最大方法近邻相似度全为 1.0，无法二分——CadQuery 方法词汇高度共享的既有事实）。self 对 EOS logprob 均值 -21.36（min -23.25, max -18.96），EOS 支配效应持续存在。

**近重复粗筛（分数无关）**：224 个正例-候选对，**103 对 token Jaccard ≥ 0.5**（最高 0.947），**0 对逐字相同**（无数据准入错误）。高相似对集中在 method-neighbor（构造使然：方法名集合全等 → 程序 token 高度重叠）。

**未 top-1 的 query（事实清单）**：q12（0045/00450329：三臂均 rank 2–3，其 0.922 近重复 method 候选胜出）、q13（0001/00018422：三臂 rank 2，其 0.947 近重复候选胜出）、q15/q17/q22（仅 sum 臂 rank 2）、q23（sum rank 3）、q25（sum rank 4）、q28（sum rank 6、mean rank 4）。corrected 臂仅 q12、q13 未 top-1——两例都有 token Jaccard >0.92 的近重复候选排在配对 UID 前。

## 4. 解释（我的分析，非结论）

1. **仪器检查通过且与 P0 一致**：同候选正确文本相对其他 UID 文本的边际在 32/32 程序上稳定为正（均值 60.1，P0 在 20 候选上为 58.2），CI 不含 0；同一程序 token 完全相同，边际不可能由候选长度单独产生。
2. **exact-UID 配对链可检索性良好**：corrected 臂 30/32 top-1、胜率 0.991，远高于所有基线；sum<mean<corrected 的梯度与长度先验效应一致（sum 惩罚长度；空 query 校正去除候选先验）。这支持"配对数据上分数对文本条件敏感且可用于排序"的仪器结论。
3. **两个持续失败案例恰好是最近重复对**：q12/q13 的胜出候选都是 token Jaccard >0.92 的近重复程序——这更像"近似等价设计竞争"而非评分器失效；它直接印证计划警告：其他 UID **可能几何等价**，exact-UID 名次会系统性低估语义可接受度。
4. **EOS 支配未消除**：EOS 一项约占 sum 的 40–53%；边际结论对 EOS 稳健（同候选 EOS 项相同会抵消），但 sum 臂的排名会被 EOS 噪声影响，corrected/mean 更可靠。

## 5. 限制（必须随结论一起引用）

- 全部数字是**官方配对数据上的 exact-UID 配对 sanity**，不是独立语义检索 R@K；无独立 qrels、无设计家族/几何等价核验，P2 问题未闭合。
- 32 个 query、每 query 8 候选的小矩阵；103/224 对高 token 相似说明候选集合含大量近重复，名次口径偏保守。
- UID ≠ 设计家族；近重复粗筛是 token 级提示，不是几何证明。
- mean 臂只是长度诊断；corrected 是已知候选先验校正诊断，均非新方法或完整条件概率。

## 6. 建议（供 Codex 决定，非授权）

- P1 目标（计划 §1 的仪器检查）已达成，建议 Codex 复核原始文件后按 §5 裁决；在独立语义真值闭合前不启动 P2/P3。
- 若进入后续轮次，建议优先做 q12/q13 类近重复对的几何等价核验与 EOS/长度分层的更大样本确认。
- 910C `/data/data_dev` 挂载恢复前，本轮归档以共享 OBS `stage3_full_mirror/` 为权威副本。


