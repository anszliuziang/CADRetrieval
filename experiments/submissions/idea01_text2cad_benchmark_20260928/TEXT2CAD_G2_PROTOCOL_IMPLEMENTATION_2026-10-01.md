# Text2CAD G2 测速协议修复：实现报告（2026-10-01）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`，真实 checkpoint/BERT 模型前向/NPU 预算全部 0）
- 合同：`experiments/plans/IDEA01_M1C_G2_PROTOCOL_REPAIR_2026-10-01.md`（上位 M1c 2026-09-30；G1 batch Codex 审查 2026-10-01 诊断的五个缺口）
- 本地起点：`32b83f76068b53b1bed12026acd36f58fa96f9cf`
- 本阶段未连接服务器、未部署、未移动远端 HEAD、未安装依赖、未读 qrels/CSV/UID 正例映射；全部实现与测试均为本地 CPU 假数据。

## 1. 改动文件与实现方法

| 文件 | 改动 |
|---|---|
| `experiments/src/idea01/text2cad_bench_protocol.py`（新增，约 1,000 行） | 纯 CPU 协议模块：真实生产批清单、精确分片权重、tier 计划与预算同源枚举、双进程文件 barrier、逐 rep 窗口重叠核验、较慢卡 ETA、token 长度审计纯函数 |
| `experiments/src/idea01/text2cad_speed_bench.py` | 新 CLI：`audit-text-lengths`、`select-g2`（纯 CPU）与 `profile-consistency`、`eta-g2`、重接线的 `throughput`（G2 协议）；修 `parse_batch_tiers` 自动追加；`_gated_setup(g2=)` 双 selection 固定；v2 结果 schema |
| `experiments/src/idea01/text2cad_frozen_runner.py` | 仅新增 `BertTextCache` 命中/未命中计数与 `stats()`（+9 行纯附加，无数学或授权范围变化；全套 runner 测试原样通过） |
| `experiments/scripts/run_text2cad_speed_bench.py` | 仅 docstring：列出新模式并声明 G2 配置未提交前全部 fail-closed、本轮无任何真实前向授权 |
| `experiments/tests/test_text2cad_bench_protocol.py`（新增 45 测试） | Task 1/2/3 症状回归与协议核验 |
| `experiments/tests/test_text2cad_speed_bench.py` | Task 2/3/4 驱动/预算/门回归；`parse_batch_tiers` 旧断言按合同改写 |

未修改：CLAUDE.md、ACTIVE_TASK.md、plans/、reviews/、configs/（含旧 G1 selection 配置与全部资产指纹）、旧 G1 三个门（spot/batch/trim/cache）代码路径、原版模型数学与固定容差、旧估算器（`estimate_full_matrix_seconds` 与旧 `eta` 模式保留供旧测试/历史解释，新 G2 入口不调用池化公式）。`generate_selection` 未动；G2 selection 的 spot/ten_pair 用同一算法重算并有逐字段相等测试。

### Task 1：真实生产批与精确分片权重
- `build_g2_selection(text_records, cad_length_records, text_token_lengths, *, batch_tiers=(1,2,4,8,16,32), texts_per_chunk=1, n_shards=2)`。文本分界逐字节复用 `PairSpace.shard_text_range(2,k)`（测试钉死 3,644 → [0,1822)+[1822,3644)）；`texts_per_chunk=1` 下每文本的 CAD 按 cad_index 顺序切相邻窗口：`production_cad_windows` 生成 floor 满批 + 1 个诚实尾批（不遗漏、不补重复 CAD）。
- 文本按**实际 BERT token 长度**（含 special tokens、不截断）排名分三层；满批按（批最大真实 CAD 长度， 窗口序）分十层，尾批单独一层。逐 tier/shard/cell 记录实际 batch 数、pair 数、实际 batch 尺寸、批最大长度端点；完整矩阵权重隐式可算，每 tier 全 cell pair 和精确等于 29,319,624（生产尺寸测试逐 tier 断言）。
- 每非空 cell 取该 shard 内 token 长度两端点 2 文本 × 该层端点 2 真实相邻 CAD 批，去重不足记录 shortfall（生产尺寸下满层无 shortfall，尾层窗口唯一属预期）。两卡样本天然来自不同文本分片；重复 pair 如实计预算。
- `audit-text-lengths`：只加载固定本地 `BertTokenizer`（`local_files_only`，绝不加载 BertModel；审计里记 `model_loaded: false`），attention_mask 求和得真实 token 数（与 dry-run 审计同一测法），>512 拒绝；记录全部 text_index/token 长度、texts.jsonl SHA、BERT 五文件 SHA；输出强制 checkout 外、拒绝覆盖，且拒绝发生在任何 transformers 导入之前（有测试）。
- `select-g2`：读取派生审计，校验其与活动 texts/BERT 资产绑定后生成新配置；LF 输出、拒绝覆盖、provenance 绑定 texts/cad_length/pair_manifest/token_audit/BERT 五文件 SHA 与规则/schema。**本轮未生成真实 token 数据与新配置**（合同要求）；`G2_SELECTION_CONFIG_REL` 指向的未来提交文件现在不存在，所有 G2 前向/ETA 入口因此 fail-closed，这是预期状态。
- 新扩展 G1 固定 64 对 = 8 个 token 长度分位 × 8 个 CAD 长度分位（0/10/25/50/75/90/99/100），q=100 即真实最长文本与最长 CAD；不读 qrels。新 schema `text2cad-g2-selection-v1` 与旧 schema 互不互认：新 verifier 拒绝旧 schema，旧 verifier 拒绝新 schema（有测试），`_gated_setup` 路径固定层面也不互换（有测试）。

### Task 2：真实批大小、预算与实际并发
- `run_production_batch_tiers` 消费 `build_tier_plans` 产出的**已分好的批**（每 cell 的样本批是文本×端点窗口的显式相邻切片），绝不重新 flatten 组批；满层批小于 tier 直接拒绝（症状：旧 nominal32 实为 20 —— 测试中先用保留的旧 `run_throughput_tiers` 复现该症状，再证明新驱动拒绝同一输入）。尾批尺寸不符同样拒绝。
- 每 cell ≥2 次 warmup（首个 cell 的前两个样本批、计账不计时）、≥3 次计时 rep（下限 3，少于即拒绝）；逐 rep 记录：真实批尺寸、逐 pair 顺序、批最大 CAD 长度、实际 text token 数与 padding 宽度、epoch start/end 测量窗口、preprocess/forward-score/write/E2E、每批 E2E、峰值 HBM、记录文件（每 tier/cell/rep 一份 jsonl，原子发布不覆盖）。OOM 记 `status="oom"` 停止升档并通知 peer；其它异常通知 peer 后中止。
- 预算与驱动同源：`plan_production_forwards` 就 tier_plans 枚举计 warmup+reps 全部 pair（测试断言 charge 流与计划精确相等：5×37 fixture = 410；生产结构下 6 档全测 = 23,580 对/卡、两卡 47,160，见 §4）。所有实际评分 pair 在 forward 前扣共享账本；模型加载与 barrier 等待都在 invocation 墙钟内计卡时。
- 双进程同步：`--sync-dir`（强制 checkout 外）+ `--sync-group`，`SyncBarrier` 按 `t{tier}_tt{t}_{layer}_r{rep}` 同步点对两卡对齐；marker 绑定 git/checkpoint/pair_manifest/selection/plan 五重 SHA，拒绝旧文件/异绑定/同设备（同设备双进程要么撞 marker 拒绝要么超时拒绝）；timeout=180 秒；等待时间单独记 `sync_wait_s`（计卡时、不混入测量窗口）；peer 错误/OOM 写 `.FAILED.json`，对端下一次到达即停，marker 证据不删除、不自动重跑。
- 窗口核验 `verify_concurrent_rep_windows`：每个 (cell, rep) 要求两卡窗口实际重叠 ≥ 较短窗口 80%；反向、缺失、非有限、错开一律拒绝。症状测试：旧 `combine_two_card_results` 接受全局窗口 100% 重叠而逐 rep 完全错开的数据，新核验拒绝。ETA 不再使用全局窗口作并发凭据。
- `parse_batch_tiers`：`--allow-large-batch` 只**允许**明确请求的 16/32，不再自动追加未请求档（旧测试断言改写，新增 16/32 精确请求用例）。

### Task 3：按固定生产分片取较慢卡 ETA
- `estimate_sharded_eta(selection, results_by_device, tier, *, overhead_seconds=None, safety_factor=1.30)`：每卡 `Σ(cell 真实 batch 数 × 该 cell 保守每批 E2E) + 该卡冷启动`；每批成本 = 完整 rep E2E/真实批数，保守取 ≥3 次 rep 最大值（函数内部从逐 rep 原始记录重算，不采信聚合字段，并交叉核对 rep 的 n_batches/n_pairs 与 selection 实际样本批一致）。全量 ETA = 两卡完成时间 **max**，再加额外开销，×≥1.30，对照 604,800 秒门。
- `overhead_seconds=None` 保留全部实测冷启动；显式值必须有限且 ≥ 两卡冷启动之和，超出部分作为两卡之后的附加墙钟；0 或小于冷启动之和直接拒绝（不能绕过冷启动）。NaN/Inf、负值、低安全系数、缺 cell、git/checkpoint/selection/score 绑定不符、v1 旧 schema、重复 invocation/device、档未完成、rep 窗口重叠不足一律拒绝。pair 权重之和严格 29,319,624，否则拒绝（不退子集/rerank）。
- 报告明列每卡 cell 数/批数/pair 数/计算秒/冷启动/完成时间、较慢卡、附加开销、保守 ETA 与等效吞吐；写盘/ledger 已在 E2E 内不重复外加（notes 字段如实声明其它 I/O 来源须经 overhead 进入）。
- 红绿证据：卡 0 每批 0.8s（40 对/秒）、卡 1 每批 1.6s（20 对/秒）、负载各半时 ETA 取慢卡 734,630.4+10 秒 ×1.30 = 955,032.52 秒，且 > 池化理想 2 倍估值的 1.4 倍（证明不是 40+20 池化）；尾层单独高价时按 shard×cell 精确权重求和，大于平均成本估值 2 倍（证明文本/批次分布不等不用理想平均）。
- 旧 `eta` 模式与 `estimate_full_matrix_seconds` 原样保留（旧 CPU 测试全绿）；新 v2 结果被旧 eta 拒绝、旧 v1 结果被 `estimate_sharded_eta` 拒绝（schema 互不互认）。

### Task 4：真正大批及优化组合的 G1 入口
- 新模式 `profile-consistency`（前向消耗、v2 schema、g2 固定）：固定 64 对按网格顺序，先 `reference_b1`（batch1/全长/无缓存）再十个固定变体：batch16、batch32、trim32、cache32、trim+cache × batch1/2/4/8/16/32。合计 **704 评分对** = 64×11；CLI 无任何改候选或改容差的开关（变体表是模块常量，容差用预注册 G1 常量）。
- 每变体前扣 64 对（测试断言 charge 序列恰为 [64]×11）；逐变体报告：真实批尺寸列表、trim/cache 状态、缓存命中统计（新 `BertTextCache.stats()`，cache32 实测 5 entries、hits+misses=64）、逐对两头/总分与有效计数（ref/alt）、最大绝对差与按固定容差归一后的最大差（score 与 logit 双口径）、有限性强检（有效行 logits 与全部 score 字段）。
- 每个变体（含参考）的有效行 logits 以 `weights_only` 可读格式落盘 checkout 外（`profile_logits_<variant>.pt`，原子发布、拒绝覆盖；测试用 `torch.load(weights_only=True)` 逐对核对形状 `(n_real-1, 2, 267)`），Codex 可独立复算全部数字，不只交布尔值。
- 任一比较失败：保留已完成变体原始证据（参考+失败变体 logits 已落盘）、结果 JSON 照常写出、非零退出并停在首个失败（测试：mock 失败时恰只执行 reference+1 变体、后续变体无文件）。此入口**不批准**生产优化：throughput/matrix 仍只有 reference 路径（throughput CLI 无 trim/cache 开关，matrix score 模式原有拒绝逻辑未动）。

## 2. 红绿证据（本地 CPU，`D:\miniconda\python.exe`）

每症状均先写失败测试再修：

1. Task 1 红：`ModuleNotFoundError: idea01.text2cad_bench_protocol`（1 failed）→ 实现后 JSON 往返 tuple/list 漂移 3 failed → 修 `_diff_path` 序列归一 → 绿。CLI 入口红：5 failed（`run_audit_text_lengths`/`run_select_g2` 不存在）→ 绿。
2. Task 2 红：19 failed（tier 计划/barrier/窗口/新驱动不存在；`parse_batch_tiers` 旧自动追加与新断言矛盾）→ 实现后 3 残留（超时消息措辞、旧 combine 已有全局 80% 检查需调整症状夹具、同设备双进程 `os.link` 竞态抛裸 FileExistsError）→ 修后绿。
3. Task 3 红：6 failed（`estimate_sharded_eta` 不存在）→ 实现后 1 处测试校准（尾层成本 100→1000 使"精确权重 > 2×平均"断言有区分度）→ 绿。
4. Task 4 红：3 failed（`run_profile_consistency` 不存在）→ 实现后 1 残留（mock 的 gate 报告 per_pair 为空触发索引错误，报告构造改为防御式）→ 绿。

最终（本轮交付前）：

```
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_bench_protocol.py experiments/tests/test_text2cad_speed_bench.py -q
79 passed, 6 subtests passed in 6.89s        # 中期快照
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_speed_bench.py experiments/tests/test_text2cad_bench_protocol.py experiments/tests/test_text2cad_frozen_runner.py -q
130 passed, 9 subtests passed in 55.27s
D:/miniconda/python.exe -m pytest experiments/tests/ -q                       # 全量（最终）
4 failed, 471 passed, 9 errors, 9 subtests passed in 96.11s
```

既有环境缺口（与本改动无关，基线提交同样存在）：`test_m0_manifest.py` 9 errors（本机无 `pyarrow`）、`test_cadmium_scorer.py::TestRealTokenizerCPU` 4 failed（本机无 `transformers`）。本轮未安装任何依赖（合同禁止）。

## 3. 实现差异（相对既有代码）

- 旧 `throughput` CLI 块整体替换为 G2 生产协议；旧 `run_throughput_tiers`/`plan_throughput_forwards` 保留仅为症状复现与旧单元测试（不再被 CLI 调用，注释标明 legacy）。旧 G1（spot/ten_pair/trim/cache 四门）代码路径逐字节未动。
- `_gated_setup(args, g2=False)`：默认行为与旧完全一致；`g2=True` 固定 `G2_SELECTION_CONFIG_REL` 并以 `verify_g2_selection_against_manifest` 全量重导出核验（schema、provenance 活动 SHA、BERT 五文件、token-audit 绑定、逐 cell 权重重算）。
- `_write_result(..., schema_version=v1)`：旧调用不变；G2 模式写 `text2cad-bench-result-v2`。
- `--reps` 默认值 5→3（仅 throughput 使用；新协议下限 3）。`--sync-dir/--sync-group/--token-audit` 为新参数。
- `score_pairs_detailed(..., cache_stats_out=None)`：仅附加统计输出；缓存包装/恢复语义不变（旧门测试原样通过）。
- `BertTextCache`：仅 +hits/misses 计数与 `stats()`。

## 4. 准确的后续命令（**全部未授权**，须 Codex 审码、推送、逐步 CPU 资产核验并在 ACTIVE_TASK 逐门授权完整 SHA 后才可执行）

### 4.1 CPU 预处理（真实 token 长度资产；服务器或持资产机器，纯 CPU、零模型前向）

```bash
# 1) 真实 BERT token 长度审计（只加载 tokenizer，不加载 BERT 模型；输出 checkout 外）
PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RUNTIME/hf XDG_CACHE_HOME=$RUNTIME/xdg TMPDIR=$RUNTIME/tmp \
/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_text2cad_speed_bench.py \
  --mode audit-text-lengths \
  --pair-manifest-dir $RUNTIME/pair_manifest_v4 \
  --bert-dir $RUNTIME/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b/ \
  --out $RUNTIME/derived/text2cad_text_token_lengths_20261001.json \
  --repo-root /home/ma-user/work/CADRetrieval

# 2) 生成 G2 selection 配置（输出即未来提交路径；先落 checkout 外供 Codex 核验再入库）
PYTHONDONTWRITEBYTECODE=1 /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_text2cad_speed_bench.py \
  --mode select-g2 \
  --pair-manifest-dir $RUNTIME/pair_manifest_v4 \
  --token-audit $RUNTIME/derived/text2cad_text_token_lengths_20261001.json \
  --out $RUNTIME/derived/text2cad_g2_pair_selection_20261001.json \
  --repo-root /home/ma-user/work/CADRetrieval
# Codex 核验 provenance/权重后提交为 experiments/configs/text2cad_g2_pair_selection_20261001.json
```

### 4.2 704 对扩展 G1（profile-consistency；**未授权**，需 pilot_authorized + 每卡 ≥704 对额度；示例单卡）

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RUNTIME/hf XDG_CACHE_HOME=$RUNTIME/xdg TMPDIR=$RUNTIME/tmp \
/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_text2cad_speed_bench.py \
  --mode profile-consistency \
  --checkpoint $RT/models/Text2CAD-v1.0/422d4c93.../Text2CAD_1.0.pth \
  --bert-dir $RT/models/bert-large-uncased/6da4b6a2.../ \
  --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93.../cad_seq.zip \
  --pair-manifest-dir $RT/pair_manifest_v4 \
  --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json \
  --selection experiments/configs/text2cad_g2_pair_selection_20261001.json \
  --out-dir $RT/runs/<profile_run_id> --device npu:0 --cards-used 1 \
  --repo-root /home/ma-user/work/CADRetrieval
# 计划与扣账固定 704 对；11 个 profile_logits_*.pt 与结果 JSON 落在 out-dir
```

### 4.3 G2 生产批测速（throughput；**未授权**，需双卡 pilot_authorized）

精确预算（`plan_production_forwards` 同源计算，6 档全测）：tier 1/2/4/8/16/32 分别 364 / 728 / 1,492 / 3,020 / 6,076 / 11,900 对/卡，合计 **23,580 对/卡、两卡 47,160 对**（两卡 shard 对称故相同）。两进程示例：

```bash
# 卡0（shard0）与卡1（shard1）同一 sync-group，同步点 t{tier}_tt{t}_{layer}_r{rep}
PYTHONDONTWRITEBYTECODE=1 ... python experiments/scripts/run_text2cad_speed_bench.py \
  --mode throughput \
  --checkpoint ... --bert-dir ... --cad-seq-zip ... --pair-manifest-dir $RT/pair_manifest_v4 \
  --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json \
  --selection experiments/configs/text2cad_g2_pair_selection_20261001.json \
  --out-dir $RT/runs/<g2_run_id>/card0 --device npu:0 --cards-used 1 \
  --batch-tiers 1,2,4,8,16,32 --allow-large-batch --reps 3 --warmup-passes 2 \
  --sync-dir $RT/runs/<g2_run_id>/sync --sync-group g2_<run_id> \
  --repo-root /home/ma-user/work/CADRetrieval
# 卡1：同上，--device npu:1 --out-dir .../card1
```

### 4.4 较慢卡全量 ETA（eta-g2；纯 CPU，**未授权**，输入须为 4.3 的真实 v2 结果）

```bash
PYTHONDONTWRITEBYTECODE=1 ... python experiments/scripts/run_text2cad_speed_bench.py \
  --mode eta-g2 \
  --pair-manifest-dir $RT/pair_manifest_v4 \
  --production-batch-size 32 \
  --card-result $RT/runs/<g2_run_id>/card0/bench_throughput_<inv0>.json \
  --card-result $RT/runs/<g2_run_id>/card1/bench_throughput_<inv1>.json \
  --out $RT/runs/<g2_run_id>/eta_g2.json \
  --repo-root /home/ma-user/work/CADRetrieval
# 退出码 0=过七天门，3=未过门； overhead 不传则保留全部实测冷启动
```

## 5. 已知限制

- 真实 BERT token 长度数据与新 G2 配置本轮**未生成**（合同限定）；`text2cad_g2_pair_selection_20261001.json` 不存在，故 throughput/profile-consistency/eta-g2 当前必然 fail-closed —— 这是合同要求的实现态，不是缺陷。
- barrier 的两进程行为在本机用线程+假 sync 夹具测试；真实双 NPU 进程同步、峰值 HBM 与 OOM 升档只能在未来获授权的双卡运行实测，本报告不外推任何吞吐数字。
- ETA 的代表性取决于 cell 端点样本（2 文本 × 2 窗口）；该协议即合同批准的口径，真实数字只能来自 §4.3 运行。
- `parse_batch_tiers` 不再自动追加 16/32 —— 旧行为从未在生产使用（此前测速未运行），旧测试断言已按合同改写。
- 本机全量测试中 `test_m0_manifest.py`（9 errors，缺 pyarrow）与 `test_cadmium_scorer.py::TestRealTokenizerCPU`（4 failed，缺 transformers）为基线既有环境缺口，与本改动无关；未安装依赖。
- 完整 29,319,624 对精确评分与七天门不变；本轮的一切入口都不允许退成子集或 rerank（ETA 对 pair 权重和做严格校验）。

## 6. Git 交付

- 暂存仅本任务文件：`experiments/src/idea01/text2cad_bench_protocol.py`（新）、`experiments/src/idea01/text2cad_speed_bench.py`、`experiments/src/idea01/text2cad_frozen_runner.py`、`experiments/scripts/run_text2cad_speed_bench.py`、`experiments/tests/test_text2cad_bench_protocol.py`（新）、`experiments/tests/test_text2cad_speed_bench.py`、本文件。无数据/权重/缓存/资产；未改 Codex 文件。
- 实现提交完整 SHA：`fca6e977834f2f022ae1740354fbfe14a7bbba8d`（本 SHA 回填紧随其后一个仅本文档的小提交；两个提交均未 push，等 Codex 审核后推送）。

review_required
