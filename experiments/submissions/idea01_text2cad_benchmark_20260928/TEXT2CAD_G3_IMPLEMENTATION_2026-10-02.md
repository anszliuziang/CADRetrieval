# Text2CAD G3 全图库 probe：本地 CPU 实现报告（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`（ACTIVE_TASK state=`implementation`；真实模型/BERT 前向与 NPU 预算均为 0，本轮无 GPU 授权）
- 依据：`experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md`（active_subplan）；上位计划 `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`；最新审核 `experiments/reviews/IDEA01_M1C_G2_B16_CODEX_RESULTS_REVIEW_2026-10-02.md`
- 起点 HEAD：`7d117b3`（Codex 审计提交）。本阶段未 SSH、未 push、未下载/安装依赖、未做任何真实模型/BERT 前向或 NPU 操作、未读 qrels/UID 映射/原 CSV/旧模型分数；全部工作为本地 CPU（`D:\miniconda\python.exe`，Python 3.13.5，torch CPU，pytest 9.1.1）
- **实现提交完整 SHA：`759691b9c6781b5a21401d8e63b427521d259bc2`**（实现 + 测试 + 固定配置；本文件随后单独提交回填）
- 预定 run_id：`idea01_text2cad_g3_probe_20261002`（仅预定，未运行；总 804600 对、12 卡小时/6 小时墙钟上限均为后续授权，不是当前额度）

## 1. 改动（事实）

| 文件 | 改动 |
| --- | --- |
| `experiments/src/idea01/text2cad_g3_probe.py` | 新增（1232 行）。纯 CPU、import 不拉 torch 的 G3 模块：盲选 selector、严格重算验证器、token 审计/冻结 G2 配置 SHA 绑定、计时 sidecar 读写与 resume 侧最小校验、`eta-g3` 双卡估算器、`select-g3`/`eta-g3` 薄 CLI |
| `experiments/configs/text2cad_g3_probe_selection_20261002.json` | 新增。由真实冻结数据生成的 LF 固定配置（4435 行）；SHA256 `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985` |
| `experiments/src/idea01/text2cad_frozen_runner.py` | 修改。可选 `--probe-selection`/`--token-audit`（probe 专用；无 probe 参数时全部输出与改动前逐字节一致）；probe 范围门（仅 Git 内固定路径 + 严格重算验证，batch16/texts_per_chunk=1 钉死）；probe chunk 过滤保持原 PairSpace/全局 text/pair 索引与完整 8046 图库；chunk header 加盖 `probe_selection_sha256` 并双向核验；每 query 计时 sidecar 在 chunk 原子落盘**后**追加；probe summary 增加冷启动/指定卡峰值 HBM/计时文件/scope 指纹 |
| `experiments/src/idea01/text2cad_pair_index.py` | 修改（+11 行）。`verify_chunk_records` 增加可选 `extra_binding_fields`（默认空 = 原行为）；scope 字段即使期望值为 None 也参与比较，全量 chunk 与 probe chunk 在对方的 resume 下都拒绝 |
| `experiments/tests/test_text2cad_g3_probe.py` | 新增（1192 行，36 测试 + 5 子测试） |
| `experiments/tests/test_text2cad_frozen_runner.py` | 修改（3 行）。AST 源码卫生检查扩展到 `text2cad_g3_probe.py` |

未改：模型数学、shift/mask、FP32、trim/cache（仍禁用）、CHUNK_BINDING_FIELDS、正式无 probe 路径、任何 Codex 文档/旧报告。未新增 warmup；未改 `estimate_sharded_eta`/G2 任何实现。

## 2. 盲选 selector 与固定清单（事实）

规则（合同 §2 逐字落实）：每个正式文本分片 `[0,1822)`、`[1822,3644)` 内按 `(n_bert_tokens, text_index)` 排序，取位置 `round(1821*j/49)`，j=0..49；排序去重后必须恰 50（实现中去重不足即失败）；全局层 `min(2, 3*rank//3644)` 与 G2 相同。selector 只读：worker 四文件、token 审计 JSON、冻结 G2 配置；不加载 tokenizer/BERT/模型，不读 qrels/UID 映射/CSV/旧分数（AST 卫生测试钉死 `qrels`/`evaluation_map`/`.csv`/`transformers` 不出现在活代码）。生成拒绝覆盖。

真实数据生成命令与输出：

```
PYTHONPATH=experiments/src D:/miniconda/python.exe -m idea01.text2cad_g3_probe --mode select-g3 \
  --pair-manifest-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker \
  --token-audit D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/derived/text2cad_text_token_lengths_20261001.json \
  --repo-root D:/projects/IdeaSpark/CADRetrievalIdea1 \
  --out experiments/configs/text2cad_g3_probe_selection_20261002.json
=> {"status": "ok", "result": "experiments\\configs\\text2cad_g3_probe_selection_20261002.json"}
```

独立复算（验证器重推导 + 活指纹 + token 审计 + G2 绑定，真实 CPU 输出）：

```
verify_g3_probe_selection_against_manifest / verify_g3_token_audit / verify_g3_g2_config_binding
=> re-derivation + fingerprint + token-audit + g2 binding: PASS
g2 binding: fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce
shard0: n_probe=50 range=[0,1822) layer_text_counts={'0':605,'1':613,'2':604} layer_probe_counts={'0':17,'1':16,'2':17}
shard1: n_probe=50 range=[1822,3644) layer_text_counts={'0':610,'1':602,'2':610} layer_probe_counts={'0':17,'1':16,'2':17}
probe_pairs_total: 804600
```

再生成确定性：临时目录重跑 `run_select_g3` 产物与提交文件**逐字节一致**（测试断言）；文件无 CR（LF）。

清单事实：100 条（50+50），无重复；每片含最短/最长文本（shard0 token 4..98，shard1 9..96）；每片每层 ≥1 probe（17/16/17）；probe 对数 2×50×8046=804600，每卡 402300。provenance 七键精确：`texts_jsonl_sha256 c59b4aef…`、`cad_store_jsonl_sha256 2f441110…`、`cad_length_index_sha256 daac66a1…`、`pair_manifest_sha256 62086beb…`（四者与 reviewed `text2cad_pair_manifest_expected_20260930.json` 逐项一致，已用 sha256sum 复核）、`token_audit_sha256 9abfd5b4…`、`g2_selection_sha256 fe42096c…`、`generated_by`。

100 个全局 text_index（供 Codex 独立复算对照）：

- shard0: 69, 111, 124, 146, 160, 222, 229, 234, 249, 263, 267, 354, 386, 387, 426, 429, 511, 524, 561, 580, 681, 689, 697, 717, 734, 749, 763, 765, 788, 933, 940, 990, 1073, 1094, 1109, 1119, 1180, 1195, 1212, 1222, 1270, 1353, 1378, 1446, 1460, 1499, 1501, 1577, 1593, 1673
- shard1: 1836, 1858, 1877, 1904, 1926, 1971, 2043, 2211, 2286, 2291, 2307, 2357, 2361, 2482, 2534, 2568, 2691, 2745, 2746, 2747, 2775, 2786, 2880, 2883, 2893, 2914, 2944, 2958, 3091, 3106, 3119, 3121, 3129, 3176, 3184, 3225, 3305, 3316, 3370, 3448, 3450, 3489, 3514, 3518, 3534, 3582, 3601, 3619, 3631, 3636

## 3. 生产 runner probe 路径（事实）

- `--probe-selection` 只接受 `repo_root/experiments/configs/text2cad_g3_probe_selection_20261002.json`（解析后路径比较），随后对**活**工件严格重算验证（schema/精确 provenance 七键/四指纹/token 审计文件 SHA 与内嵌表一致/冻结 G2 配置 SHA），任何一项不符在任何加载与授权前 exit 2；`--token-audit` 仅在 probe 下有意义，单独给也 exit 2；probe 与 dry-run 互斥。
- probe 固定 `--batch-size 16`（尾批 14 由原批界自然产生：8046 = 502×16 + 14）与 `--texts-per-chunk 1`，否则 exit 2；FP32/无 trim/无 cache/两卡 shard↔device 沿用既有 score 门。第一 query 无额外 warmup；预算只对选中 chunk 计（每卡 planned=402300，账本上限由 ACTIVE_TASK 授权决定，本实现不改变账本数学）。
- PairSpace/全局 text_index/pair_index/图库顺序不变：probe 只是把这个 shard 的单文本 chunk 计划过滤到 50 条；每条 query 仍遍历全部 8046 CAD；不建 100 条小 manifest、不用 max-pairs 截断。
- resume 双向 scope 隔离：chunk header 在 probe 下携带 `probe_selection_sha256`；`verify_chunk_records(extra_binding_fields=(…,))` 对两边都比较——全量 chunk（无该键）在 probe resume 下拒绝，probe chunk（有该键）在全量 resume 下拒绝；非本片/非 plan 文件沿用原拒绝；不覆盖任何已有文件（原子 hard-link 语义不变）。probe 计划数 != 50 即拒绝。
- 每 query 计时：`timing_s{SS}.jsonl`（out_dir 内，逐行 append+fsync）在 chunk 原子发布**完成后**写入一行：全局 text_index/分片/层/token 数/pick、真实 start/end epoch、E2E（=score+序列化+原子落盘）、实际 pair 数 8046/批数 503/scored/rejected/valid_total、chunk 文件名、scope 指纹（probe SHA、git SHA、pair manifest、checkpoint、score name/spec、device）。第一行只可能在对应 chunk 已落盘后出现；崩溃撕裂末行 → 严格读取拒绝（fail closed）；resume 时已验证完成的 chunk 若无对应计时行 → 拒绝复用并停下（不静默重评、不伪造时间）。
- probe summary 增加：`git_sha`、`pair_manifest_sha256`、`probe_selection_sha256`、`timing_file`、`cold_start_s`（模型+候选加载，perf_counter）、`peak_device_memory_bytes`+`hbm_device`（加载后 reset、评分末读取，NPU 读数失败即异常、CPU 为 None）、`probe_queries_total`。账本/授权/退出码语义不变（每进程 cards_used=1，finalize 记真实 exit_code）。

## 4. CPU ETA 入口（事实）

`eta-g3`（`idea01.text2cad_g3_probe`，纯 CPU）：输入恰好两个 `--card-summary` + 各自 sidecar（summary 内 `timing_file` 名解析于同目录）。核验链：committed 选择配置（钉路径 + 活工件重算 + token 审计 + G2 绑定 + 配置 SHA）→ 每卡 summary（schema/成功状态/device↔shard/batch16/texts_per_chunk=1/FP32 无 trim 无 cache/pinned checkpoint/score 名与 spec/40hex git SHA/authorization 与 budget 一致/冷启动有限非负/本卡峰值 HBM 正整数/chunk 计数与计时条数一致）→ 账本（恰好一条本 invocation 记录、finalized、exit_code 为整数 0、cards_used=1、shard 对应、两卡同 host 同账本文件）→ sidecar（恰好覆盖本片 50 条 probe 各一次、每条 n_pairs=8046/n_batches=503/n_scored=8046/n_rejected=0/E2E==end-start>0/层与 token 数与已验证清单一致/chunk 文件存在/scope 全等）→ 双卡同代码/资产/score/probe 指纹、不同 invocation、真实评分窗口（首 query start → 末 query 落盘 end）重叠 ≥ 较短窗口 80%。估算：每卡每层（全局层）最慢完整 query E2E × 该卡该层完整文本数，加该卡实际冷启动；两卡取 max（永不理想倍乘单卡）；`overhead_seconds` 可给且必须 ≥ 两卡冷启动之和（None=仅冷启动）；×1.30（实现拒绝 <1.30）后与 604800 秒比较（`<=` 通过），exit 0/3，超限不改系数。输出明列每层完整数/probe 数/最慢 query text_index 与 E2E/各卡成本/冷启动/窗口与重叠率/开销/原始输入路径与全部 SHA。

## 5. CPU 测试证据（真实输出）

新测试 `experiments/tests/test_text2cad_g3_probe.py`（36 测试 + 5 子测试）：pick 位置/层边界数学；真实数据契约（50+50/端点/去重/层覆盖/计数/逐字节再生成/LF）；篡改拒绝（越界/重复/错分片/错层/错长度/错 schema/错指纹/注入键/缺节/内嵌表漂移）；非官方尺寸拒绝；mini manifest 上 probe 过滤保持全局索引与全图库、双向 scope 拒绝、非 plan chunk 拒绝、resume 跳过已验证 chunk、缺计时行拒绝；scorer header 仅在 scoped 时加盖（其余字段逐字节一致）；sidecar 往返/撕裂行拒绝/resume 侧 scope·分片·重复拒绝；ETA 手算（每层最慢 query 定价而非均值、慢卡为墙钟、门两侧 604799.0 通过/604800.3 拒绝、开销规则、窗口重叠门、约 40 个 fail-closed 拒绝）；eta-g3 CLI 对真实提交配置端到端（含缺 chunk 文件拒绝）；runner main() probe 门在授权前 exit 2（mock 授权未被调用）；常量交叉核对（复制常量与 loader/vector_scorer/speed_bench/runner 源一致、指纹加载器与 runner 版输出一致、g3 导入不拉 torch）。

```
pytest experiments/tests/test_text2cad_g3_probe.py -q
=> 36 passed, 5 subtests passed in 5.99s

pytest experiments/tests/test_text2cad_g3_probe.py experiments/tests/test_text2cad_frozen_runner.py \
       experiments/tests/test_text2cad_pair_index.py experiments/tests/test_text2cad_bench_protocol.py \
       experiments/tests/test_text2cad_speed_bench.py -q
=> 225 passed, 33 subtests passed in 71.57s
```

受影响既有测试全部通过（frozen_runner 79+3 含 pair_index；bench_protocol+speed_bench 110+25），无 skip/xfail 新增（真实资产 skipUnless 项本机全部实际执行）。未运行与本次无关的全套测试（cadmium/p1 等未受影响）。

## 6. 准确后续命令（**当前未授权**；须经 Codex 审码、ACTIVE_TASK 批准完整 SHA 与预算后按批准 SHA 使用）

服务器 `yunnan-share2`，`$REPO=/home/ma-user/work/CADRetrieval`，`$PY=/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python`，`$RT=/home/ma-user/work/CADRetrieval_runtime`，`$RUN=$RT/runs/idea01_text2cad_g3_probe_20261002`。前置：远端 `git status --porcelain` 为空、fetch 后 `git checkout --detach <批准完整SHA>` 并核对 `git rev-parse HEAD`；两卡 `npu-smi info` 健康且无其它进程；`$RUN` 不存在任何旧产物；ACTIVE_TASK 为 `pilot_authorized` 且 `approved_sha` 等于运行 SHA、`max_scored_pairs=804600`、卡时预算 ≥12 卡小时（`gpu_budget` 对应 43200 卡秒）。两进程可错开 3 秒启动避 Git fetch 锁，必须保留真实并发窗口。

```bash
# card 0（probe 文本分片 [0,1822) 的 50 条，device npu:0）
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" TMPDIR="$RT/tmp" timeout 21600 \
  "$PY" "$REPO/experiments/scripts/run_text2cad_score_matrix.py" --mode score \
  --checkpoint "$RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth" \
  --bert-dir "$RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b" \
  --cad-seq-zip "$RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip" \
  --pair-manifest-dir "$RT/pair_manifest_v4" \
  --expected-keys-manifest "$REPO/experiments/configs/text2cad_checkpoint_expected_keys_20260930.json" \
  --probe-selection "$REPO/experiments/configs/text2cad_g3_probe_selection_20261002.json" \
  --token-audit "$RT/derived/text2cad_text_token_lengths_20261001.json" \
  --batch-size 16 --texts-per-chunk 1 --cards-used 1 --device npu:0 \
  --shard-id 0 --n-shards 2 \
  --out-dir "$RUN/card0" --repo-root "$REPO"

# card 1：同上，仅 --device npu:1 --shard-id 1 --out-dir "$RUN/card1"
```

中断续跑：同命令加 `--resume`（已验证 chunk 与其计时行齐全才跳过；任何 scope/缺计时/外来文件都拒绝）。产物：每卡 50 个 `chunk_s*_.jsonl`、`timing_s0*.jsonl`、`summary_*.json`、共享账本文件；运行后在 `$RUN` 生成 `SHA256SUMS` 取回本地并归档 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g3_probe_20261002/`（cvteam14 仅存储）。

CPU ETA（两卡成功完成后，纯 CPU、可在云脑或本地执行；本地需指向同一 runtime 资产）：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/experiments/src" \
  "$PY" -m idea01.text2cad_g3_probe --mode eta-g3 \
  --pair-manifest-dir "$RT/pair_manifest_v4" \
  --token-audit "$RT/derived/text2cad_text_token_lengths_20261001.json" \
  --repo-root "$REPO" \
  --card-summary "$RUN/card0/summary_<invocation_card0>.json" \
  --card-summary "$RUN/card1/summary_<invocation_card1>.json" \
  --out "$RUN/eta_g3_20261002.json"
# exit 0 = 通过七天门；exit 3 = 超限；exit 2 = 任何核验拒绝
```

G2 门（38.28394 小时）与本门均 ≤604800 秒且代码/结果审核通过，才自动准备正式全量 G4；否则先与用户讨论。100 条 probe 不替代完整 29,319,624 对矩阵。

## 7. 已知限制（事实）

1. runner 主循环的 probe 计时集成（score→发布→sidecar 顺序）在本地无法端到端执行（需授权 + 生产 checkout + 真实权重/NPU）；组成单元（score_chunk header 加盖、atomic_publish、append/build/verify timing record、plan_chunks 过滤与隔离）均有 CPU 测试，main() 仅覆盖到授权前的门。Codex 审码时需重点复核主循环 diff。
2. `eta-g3` 尚未有真实 probe 输出可算；当前只用合成/构造证据验证数学与拒绝。G2 的 38.28394 小时估计仍是唯一真实测量基线。
3. `_peak_device_memory_bytes`/`_reset_peak_device_memory` 与 bench 版语义一致（NPU 读数失败即异常；CPU 返回 None），为回避循环 import 在 runner 内复制 6 行；ETA 拒绝 None/零/异机读数，真实 NPU 运行必须给出正整数。
4. probe resume 的计时完整性检查假定"chunk 发布成功但计时行缺失"只由崩溃造成；该情形按合同 fail-closed 停下等人工处理，不自动修复。
5. G2/G1 预算与账本不可复用；本轮未申请也未消耗任何前向预算。

## 8. 状态

**review_required**。实现提交 `759691b9c6781b5a21401d8e63b427521d259bc2`（git status 干净）；本文件回填该 SHA 后单独提交并停。请 Codex 审源代码、独立复算清单/分层/预算/SHA 与受影响 CPU 测试，通过前不 push、不 NPU。
