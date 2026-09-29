# M1b 实现交接：CADmium 真实 runner（本地实现，NPU 预算 0）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 总计划： `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- 子计划（本合同）： `experiments/plans/IDEA01_M1B_CADMIUM_NPU_BENCH_2026-09-30.md`
- 当前 `experiments/ACTIVE_TASK.md` 状态（2026-09-30 读取）： `state: implementation`，`approved_sha: 无`，`gpu_budget: 0 卡小时`，`max_scored_pairs: 0`
- 本文件作者： Claude Code / Kimi K3（本地实现者）。**本轮没有提交、没有推送、没有 SSH、没有任何模型前向；本文档是唯一新增/修改文件。**

## 1. 作者归属：Kimi 草稿/修复 vs Codex 改动

为避免审查时混淆，明确区分：

**Kimi（本地实现者）编写、未经提交的工作区文件：**

- `experiments/src/idea01/cadmium_runner.py` —— 三轮演进：
  1. 初稿（被 Codex 预审查出 6 项缺口，见 `experiments/reviews/IDEA01_M1B_CODEX_PRE_REVIEW_2026-09-30.md`）；
  2. 第一轮修复：封闭资产 SHA 清单、授权门、输入隔离、单次前向 spot check、原子输出，46 项 CPU 测试；
  3. 第二轮修复（针对 2026-09-30 追审“授权来源”阻塞项）：授权文本改为**只从 `git fetch origin` 后的 `origin/master:experiments/ACTIVE_TASK.md` 读取**（复用已审 `p1_runner.fetch_authorized_active_task`，先校验 origin URL），要求 HEAD detached 且等于批准 SHA、工作树干净；测试改为真实临时 Git 仓库的双版本场景（批准提交 SHA ≠ 被批准代码 SHA，由构造保证）。当前 50 项测试。
- `experiments/tests/test_cadmium_runner.py` —— 对应上述两轮修复的 50 项 CPU 测试。

**Codex 编写/暂存、Kimi 未触碰：**

- `experiments/plans/IDEA01_M1B_CADMIUM_NPU_BENCH_2026-09-30.md`、`experiments/ACTIVE_TASK.md`（M1b 激活、零预算）；
- `experiments/reviews/IDEA01_M1B_CODEX_PRE_REVIEW_2026-09-30.md`（预审查 6 项 + 追审）；
- `experiments/MANIFEST.md` 的 3 行追加（工作区未提交修改，记录上述审查与两份配置的登记；非本任务代码，我未编辑）；
- `experiments/configs/cadmium_asset_sha256_20260930.json` 与 `experiments/configs/cadmium_m1b_pilot_selection_20260930.json` 的**登记与核验口径**由 Codex 在 MANIFEST 中记录。runner 生产 CLI 硬性要求 `--asset-manifest` 必须就是被审 checkout 内的 `experiments/configs/cadmium_asset_sha256_20260930.json`（路径不等即退出 2），因此该清单是审查对象的一部分，不是运行时可替换参数。

以上全部文件**尚未提交**，因此当前没有可批准的完整 SHA；`approved_sha` 待 Codex 审码后填写。

## 2. 实现内容与方法（对应合同条目）

复用已审组件，不重新实现数学：`cadmium_scorer.py`（M1a 已审：tokenization、边界、终止符、4096 上下文拒绝、tokenizer 指纹、`score_batch_consistency`）、`scoring.py`（已审 numpy 核心）、`p1_runner.fetch_authorized_active_task`（已审双版本授权获取）。

1. **资产预检（合同 2，审查项 1，fail closed）**：`preflight_assets` 在任何权重加载前执行。base 目录与 adapter 目录必须存在且路径含 pinned revision（`488639f1…` / `6c6a79d1…`，生产 CLI 无关闭开关）；`--asset-manifest` 为**必填**，须 SHA256 覆盖加载实际读取的每一个文件——base `config.json`、safetensors 索引与全部 shard（或单文件）、adapter `adapter_config.json` + 恰一个权重文件 + tokenizer 文件；磁盘上存在的 `generation_config.json`、`special_tokens_map.json`、`added_tokens.json`、`vocab.json`、`merges.txt`、`chat_template.jinja` 等可静默影响加载的小文件也必须在清单内。缺文件、未覆盖、哈希不符均在任何前向前抛出 `RunnerError`（CLI 退出 2）。
2. **运行授权门（审查项 2 + 追审，fail closed）**：`authorize_run` 依次校验——origin URL 为项目远端 → `git fetch origin` → 从 `origin/master:experiments/ACTIVE_TASK.md` blob 解析 `state: pilot_authorized`、完整 40-hex `approved_sha`、`max_scored_pairs > 0`、`gpu_budget > 0` → HEAD 必须 **detached** 且 `rev-parse HEAD == approved_sha` → `git status --porcelain` 为空。任何一项不符即拒绝。工作树里的 `ACTIVE_TASK.md` 永远不作为授权来源（一个提交无法在自身文件里预写自己的 SHA；批准提交与被批准代码提交是两个 SHA，均记录进审计字段）。当前真实 origin/master 状态为 `implementation`/0/0，**该门按设计拒绝**（有测试锁定此行为）。
3. **预算台账（审查项 2/5）**：`BudgetLedger` 持久化累计已前向对数、`started_at_epoch`、绑定的 `approved_sha`/对数上限/墙钟上限（=卡时×3600/占用卡数）；换批准、改上限、改墙钟的旧台账拒绝加载；墙钟跨重启累计不重置；超限扣账在任何前向前拒绝且不持久化。 scorer 在整个 pending 集合**首次前向前一次性扣账**，每个 batch 前复查墙钟。
4. **输入隔离（审查项 3）**：`load_pair_requests` 机器检查每行 JSON 恰含 `query_text`、`candidate_json_desc` 两个字符串字段——UID/qrels/annotation/多余键/重复键/非字符串均退出 2，且此门在授权门与任何模型加载**之前**。UID↔pair_index 映射属外层清单/日志，runner 永不见 UID。
5. **评分数学（合同 3）**：tokenization/边界/终止符/4096 拒绝完全复用已审 `build_cadmium_sequence`（`len(full_ids) > 4096` 记 `out_of_training_context`，绝不截断）。设备端对 causal shift 后的预测行做 **FP32 `log_softmax` + gather**，按 `logits_chunk_rows`（默认 512）分行 chunk 以界定 FP32 工作区；只有逐对标量（sum/mean/finite）、长度、target ids 及（`--keep-token-scores` 时）逐目标 token log-prob 回 CPU，整批 `[B, L, V]` logits 不过 CPU。右 padding 仅用于 batching。
6. **一致性门（合同 3，审查项 4）**：`spot_check_with_cpu_core` 对**同一次真实前向**的 logits 分别做设备 gather 与已审 CPU 核心（`score_pair_from_logits`）对照，只消耗 1 对预算（修掉了初稿的重复前向）；`verify_batch_consistency` 对同一固定 pair 清单在 batch=1 与 batch=B 下评分，超出事前容差（rtol=1e-5, atol=1e-6）即抛错停止（设计上消耗两倍预算，因为只有真实 logits 上的比较才有意义）。
7. **输出原子性（审查项 5）**：`_atomic_write_new` 临时文件 + fsync + 硬链接发布，目标已存在即拒绝覆盖；CLI 还拒绝把 output/summary/budget 写进 Git checkout 内，拒绝复用已存在的输出名（新 run 用新 run_id）。
8. **可测试性**：模块在无 torch/transformers/peft 时仍可 import（惰性导入），全部门径可在 CPU 上测试；`require_revision_in_path=False`、直接构造 `RunAuthorization` 等仅为测试入口，生产 CLI 无对应开关（有测试证明 `--max-pairs`/`--no-revision-path-check`/`--active-task` 这类放宽参数不存在）。

## 3. 官方来源与 revision

- base：`Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5`
- adapter：`chandar-lab/CADmium-3B` @ `6c6a79d160595207804171be56a037883b369abd`（tokenizer 从 adapter 目录加载，须通过 M1a 指纹）
- system message：`experiments/configs/cadmium_system_message.json`（源自 chandar-lab/CADmium @ `2b2a5e7…`，文件内记录 SHA256）
- 资产 SHA 清单：`experiments/configs/cadmium_asset_sha256_20260930.json`（base 5 项 + adapter 9 项，共 14 个文件）
- 固定单对 pilot 选择：`experiments/configs/cadmium_m1b_pilot_selection_20260930.json`（query `Q00000`、cad `0026/00267624`；pair JSONL SHA256 `501756a7…`；CPU 审计全长 1018 token，远在 4096 内；不含文本本体、不读 qrels）
- 云脑运行副本（checkout 外，已经官方 SHA256 核对；仅为资产，非运行授权）：
  - `/home/ma-user/work/CADRetrieval_runtime/models/Qwen2.5-Coder-3B-Instruct/488639f1ff808d1d3d0ba301aef8c11461451ec5/`
  - `/home/ma-user/work/CADRetrieval_runtime/models/CADmium-3B/6c6a79d160595207804171be56a037883b369abd/`
  - 数据：`/home/ma-user/work/CADRetrieval_runtime/datasets/CADmium-ds/ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4/`

## 4. 本地测试命令与结果（本轮实际执行）

```
python -m unittest discover -s experiments/tests -p "test_cadmium_runner.py" -v
# Ran 50 tests in 7.555s — OK（本机 CPU，无 torch import、无网络；Git 测试只用本地路径远端）

python -m unittest discover -s experiments/tests -v
# Ran 208 tests in 19.897s — OK（含已审 M1a scorer/scoring/M0/P1 全部回归）
```

50 项 runner 测试覆盖：清单必填/缺文件/哈希不符/未覆盖 shard/歧义 adapter 权重/单文件 base 布局/路径遍历与坏 SHA/未清单化小文件/pinned 路径检查无生产旁路；授权字段纯文本拒绝（非 pilot_authorized、占位或短 SHA、0 对数、0 卡时）；**真实临时 Git 仓库**双版本端到端（批准提交≠代码 SHA、工作树陈旧文本不能自授权、attached HEAD 拒绝、HEAD≠批准 SHA 拒绝、origin URL 不符拒绝、非 Git 目录拒绝、cards_used 折算墙钟）；台账持久/重启/越顶/跨批准/改 cap/改墙钟拒绝；输入隔离全部拒绝路径；CLI 三门在模型加载前退出 2；放宽开关不存在；原子输出不覆盖。

## 5. 已知缺口与剩余工作（诚实清单）

**本合同内尚未完成：**

1. **批量 benchmark 模式未实现（合同 5）**。当前 CLI 只有评分入口，没有合同要求的可重入测速器：固定 pair 集、batch `1,2,4,8(,16)`、≥2 次 warmup、纯前向与端到端计时、pairs/s 与 target tokens/s、HBM 峰值、失败率、长度分桶分位、OOM 记录并停止放大。建议下一短任务交付独立 bench 脚本（复用本 runner 的 scorer 与台账，不动评分数学）。
2. **单对 pilot 的 spot-check 驱动未暴露为 CLI**。`spot_check_with_cpu_core` 是类方法；合同 pilot 第 1 步“1 对真实分数与完整 token 对照”需要一个极薄的、同样经审的驱动脚本构造 scorer 并调用它（或给 CLI 加 `--spot-check` 模式）。当前只能用 `--keep-token-scores` 得到设备端逐 token 分数，CPU 核心对照尚未接线成命令。
3. **pair 清单构建器未实现（合同 4）**：从已审 M0 `query.jsonl` 与 pinned CADmium-ds test parquet 只取 `uid,json_desc` 生成 UID-free pairs JSONL + 外层 UID 映射（qrels 永不进评分进程）的外层脚本尚未写。pilot 单对清单已由 Codex 配置固定，但 10 对与 ≤512 对/档的固定样本清单仍需该构建器。

**明确未做（预算 0，按合同停止）：** 未 SSH、未加载真实权重、无任何 NPU/CPU 模型前向；设备端 FP32 log-softmax 与 CPU 核心在真实 logits 上的偏差、batch=1/B 数值等价、真实 HBM 峰值均未测量——这些是 pilot 审核后的验收内容，不是已实现的事实。`ACTIVE_TASK.md` 特别核验项（Text-to-CadQuery 99,236 训练 UID 与官方 test 交叉 4,581 的独立复算）属另一臂，本轮未涉及。

**风险预估（供审查参考，非实测）：** bf16 3B 权重约 6–7 GB；logits 本体 `[B, L, 151936]` bf16 在 B=8、L 接近 4096 时约 10 GB，加上 LoRA 与激活，batch 8–16 长序列存在 OOM 风险；chunk 只界定 FP32 gather 工作区（512 行约 0.3 GB），不缩小 logits 本体。测速合同已要求 OOM 即记录并停止该档放大。

## 6. 单对 pilot 命令提案（pilot 第 1 步，待批准后执行）

前提：Codex 审码通过 → 本地提交推送 → `ACTIVE_TASK.md` 改为 `pilot_authorized` 并填入该提交的完整 `approved_sha`、正的 `max_scored_pairs` 与 `gpu_budget` → 远端 `git fetch origin` 后 `git checkout --detach <approved_sha>`（工作树须干净）。以下命令**当前不会也不能运行**（授权门会拒绝，已被测试锁定）：

```bash
ssh yunnan-share2
cd /home/ma-user/work/CADRetrieval          # detached @ <approved_sha>, git status 干净
RUN=/home/ma-user/work/CADRetrieval_runtime/runs/m1b_pilot_g1_onepair_<run_id>
mkdir -p "$RUN"
# pairs_one.jsonl 按 configs/cadmium_m1b_pilot_selection_20260930.json 固定内容生成于 $RUN，
# 生成后核对 sha256 == 501756a7b5ce0e2dc15c08ffabf37d0c8517b3fb4eef2740aff75e4dee812db7
source /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/activate
env PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/ma-user/work/CADRetrieval_runtime/hf \
    XDG_CACHE_HOME=/home/ma-user/work/CADRetrieval_runtime/xdg \
    TMPDIR=/home/ma-user/work/CADRetrieval_runtime/tmp \
    PYTHONPATH=experiments/src \
python -m idea01.cadmium_runner \
  --base-model-path /home/ma-user/work/CADRetrieval_runtime/models/Qwen2.5-Coder-3B-Instruct/488639f1ff808d1d3d0ba301aef8c11461451ec5 \
  --adapter-path    /home/ma-user/work/CADRetrieval_runtime/models/CADmium-3B/6c6a79d160595207804171be56a037883b369abd \
  --system-message-config experiments/configs/cadmium_system_message.json \
  --pairs-jsonl     "$RUN/pairs_one.jsonl" \
  --out-jsonl       "$RUN/scores.jsonl" \
  --summary-json    "$RUN/summary.json" \
  --batch-size 1 --device npu:0 --torch-dtype bfloat16 \
  --asset-manifest  experiments/configs/cadmium_asset_sha256_20260930.json \
  --repo-root       /home/ma-user/work/CADRetrieval \
  --keep-token-scores
```

缺口提示：该命令产出设备端逐 token 分数与 sum/mean；合同第 1 步要求的“设备 gather vs 已审 CPU 核心同 logits 对照”还需上面缺口 2 的薄驱动（预算消耗恰为 1 对）。预计单对前向秒级、HBM 占用 <10 GB；具体以实测为准并记录进 `environment.json`。

## 7. 停止线与请求

- 当前 `gpu_budget: 0`、`max_scored_pairs: 0`，本实现阶段到此停止；授权门设计为在该状态下拒绝任何前向。
- 请 Codex 审查：`cadmium_runner.py`、`test_cadmium_runner.py`、两份 config 清单及上述剩余工作划分；审码通过后填写完整 `approved_sha` 并决定是否将 spot-check 驱动与 benchmark 模式拆为下一短任务。

`review_required`
