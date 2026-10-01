# 原版 Text2CAD 检索矩阵评分器与 910C 测速器（M1c Stage B2）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 子计划： `experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`（G0.5 pair 构建 + runner + 测速器实现部分）
- 前置审查： Stage A `experiments/reviews/IDEA01_M1C_STAGE_A_CODEX_REVIEW_2026-09-30.md`、Stage B1 `experiments/reviews/IDEA01_M1C_STAGE_B1_CODEX_REVIEW_2026-09-30.md`
- 本文件作者： Claude Code / Kimi K3（本地实现者），2026-09-30
- **本轮边界**：只做本地实现、CPU 合成测试、零真实前向的清单/资产审计与 fail-closed 拒绝证据。**没有 SSH、没有 NPU、没有真实模型前向（CPU/NPU 预算均为 0）、没有下载资产、没有改服务器、没有 git commit/push、没有修改 Codex-owned 文件（`ACTIVE_TASK.md`/`plans/`/`reviews/`/`CLAUDE.md`）。** 唯一对 Codex 已审文件的改动是 `text2cad_port/model.py` 的加性可选参数与 `NOTICE.md` 的对应登记（§5，请求重审）；已审 Stage A scorer 逐字未动。
- 本地 HEAD（工作树，未提交）：基于已审 `76dd70aad5fe1531d3d39df0feb75afb669c659d` 的本轮新增/修改文件见 §1；请 Codex 审后提交并回填批准 SHA。

## 1. 新增/修改文件

新增：

| 文件 | 用途 |
| --- | --- |
| `experiments/src/idea01/text2cad_pair_index.py` | 外层 pair/index builder：固定排序唯一文本索引（`(text_sha256, text)` 字典序）、CAD zip member 索引（校验升序 cad_id）、隐式精确笛卡尔积、分片/分块/断点核验；stdlib-only（扫描时惰性 torch） |
| `experiments/src/idea01/text2cad_frozen_runner.py` | 冻结模型 runner：严格加载（manifest 强制）、zip `weights_only` 候选读取+验证、teacher forcing + 已审 scorer、两卡共享持久预算账（锁+心跳+保守陈旧收尾）、fail-closed 授权门、chunk 原子发布/断点恰一次、dry-run 零前向 |
| `experiments/src/idea01/text2cad_speed_bench.py` | G1/G2 CLI：`select`/`eta`（纯 CPU）与 `spot`/`batch-consistency`/`trim-check`/`cache-check`/`throughput`（授权+预算门）；等价门比较函数；长度分布加权 ETA 与七天门 |
| `experiments/scripts/build_text2cad_pair_manifest.py` | builder 薄 CLI |
| `experiments/scripts/run_text2cad_score_matrix.py` | 矩阵评分 CLI 薄封装（含生产命令模板） |
| `experiments/scripts/run_text2cad_speed_bench.py` | 测速 CLI 薄封装（含生产命令模板） |
| `experiments/configs/text2cad_pair_manifest_expected_20260930.json` | 已审清单指纹：四个 artifact 的 SHA256 + pair_manifest.json 自身 SHA + 输入（M0 query/gallery、cad_seq.zip）SHA |
| `experiments/configs/text2cad_g1g2_pair_selection_20260930.json` | 固定选择：1 spot 对、10 批一致对（跨短/中/长 CAD × 短/中/长/最长文本）、30 长度层 × 20 = 600 代表性测速对；含 provenance 与逐对核验事实 |
| `experiments/tests/test_text2cad_pair_index.py` | 29 项测试 |
| `experiments/tests/test_text2cad_frozen_runner.py` | 40 项测试 |
| `experiments/tests/test_text2cad_speed_bench.py` | 24 项测试 |

修改（**请 Codex 重审，见 §5**）：

| 文件 | 改动 |
| --- | --- |
| `experiments/src/idea01/text2cad_port/model.py` | `Text2CAD.forward` 与 `teacher_forced_logits` 新增**可选** `cad_seq_len` 参数（默认 `None` → 271，已审路径逐位不变）；越界以 `bad_cad_seq_len` 拒绝。仅供默认关闭的尾部裁剪门使用 |
| `experiments/src/idea01/text2cad_port/NOTICE.md` | 登记上述两处改动（ShareAlike 改动清单） |
| `experiments/tests/test_text2cad_model_adapter.py` | 新增 `TestCadSeqLenOverride` 3 项等价/拒绝测试（B1 其余 56 项未动且全过） |

## 2. Pair 空间与隔离设计（对应任务第 1 条）

- **隐式精确笛卡尔积**：`pair_index = text_index * 8046 + cad_index`（text-major），`pair_index ∈ [0, 29,319,624)`；永不生成 2,932 万行清单。`PairSpace.coords_of/pair_index_of` 互逆且有范围拒绝。
- **唯一文本索引**：3,644 种逐字文本，按 `(text_sha256, text)` 内容寻址排序——与 M0 的伪随机 query 顺序无关，重建逐字节一致（两次本地构建 + 测试内重建均 byte-identical）。文本只做协议已有的 `strip()` 核验（不等于自身 strip 的文本拒绝），不再归一化。
- **CAD 索引**：8,046 项，校验 gallery 为 cad_id 升序、`cad_seq_member` 恰为规范路径、zip 内恰好存在一次（多出的 train/val member 只计数不报错）。zip SHA 恒对 pinned `9f3a…7eaa` 核验。
- **分片**：`round(k*n_texts/n_shards)` 连续文本段 → 连续 pair 段；任意 `1≤S≤3644` 互斥且完备（测试对 S∈{1,2,3,7,11,3644} 验证无缝无重）。两卡即 `n_shards=2, shard_id∈{0,1}`。
- **分块与断点**：chunk = `texts_per_chunk`（默认 1）个连续文本的全部 8,046 对；chunk 文件 = `chunk_header`（绑定字段：git SHA、pair manifest SHA、checkpoint SHA、device、dtype、trim/cache 标志、score spec）+ 稠密升序 pair 记录；原子发布（hard-link，存在即拒）。`--resume` 重验每个已存在 chunk（计数、pair 序列、绑定、终态），全部通过才跳过；篡改/外来文件/非终态一律大声拒绝。任意 pair 恰一次：完整 chunk 不重算，不完整 chunk 整体重算且预算按实际前向重复扣（保守方向）。
- **隔离**：worker 侧只有 `texts.jsonl`（`text_index/text/text_sha256/n_chars`）与 `cad_store.jsonl`（`cad_index/member`）；`evaluation_map.json`（query_id→text_index、cad_index→cad_id，无相关性信息）单独成文件且 runner/bench 无任何参数能读到它（AST 级卫生测试锁定 runner/bench 不含 qrels/evaluation_map/CSV 引用）；qrels 全管线无人读。zip member 路径含 UID 字符串，仅为存储定位键，不携带任何 query↔CAD 关系（§7 风险 R6 有如实登记）。
- **builder 核验**：精确字段集、重复 JSON 键、`query_id` 格式/唯一、`n_chars`/`text_sha256` 逐字节一致、cad_id 模式/唯一/升序、member 规范路径、期望计数（7934/3644/8046/29,319,624）硬失败。

## 3. Runner（对应任务第 2、3 条）

- **加载**：`load_frozen_text2cad`（已审 B1 loader）强制传 `expected_keys_manifest`；生产 CLI 把该路径钉到 checkout 内已提交清单（resolve 相等才放行），并把 `--expected-pair-hashes` 默认钉到已提交指纹配置。checkpoint/BERT 五文件/zip 的 SHA、172 键三方相等、零张量白名单、逐张量数值相等全部沿用 B1 门。
- **候选读取**：`ZipCandidateStore.preload()` 用 `weights_only=True` 逐 member 读入并做全结构验证（精确键集、dtype、形状、`attn_mask` 与规范 `generate_attention_mask(271)` 逐元素相等、已审 `validate_cad_vec`）；不合格候选记入 `rejections`，其所有 pair 记 `rejected`（不前向、不扣预算）。候选常驻 CPU 内存（约 55 MB；规范 271×271 attn_mask 只存一份），消除逐 pair zip IO。
- **前向与评分**：`torch.inference_mode()` 下 `teacher_forced_logits`（已审 B1 胶水）→ 原始 logits 在设备上转 FP32 后落 CPU（每对约 0.58 MB，批量 8 约 4.6 MB）→ **已审 Stage A CPU scorer 逐字复用**（无独立设备侧 gather 需要审）。逐对记录 `score_x/score_y/score_total/n_valid_x/n_valid_y/n_valid_total/valid_seq_len/n_real_tokens/mean_*（仅诊断）/spec_version`；拒绝记录带稳定 reason。不做静默截断：文本 >512 BERT token 由已审 `TextEmbedder` 硬拒绝（dry-run 另有全文本 token 审计）。
- **授权门（fail-closed，顺序即门序）**：① 清单指纹（已审配置）；② 文本/CAD store 模式；③ out_dir 在 checkout 外；④ `authorize_text2cad_run`：经已审 `cadmium_runner.authorize_run` 读**新 fetch 的** `origin/master:experiments/ACTIVE_TASK.md`（origin URL 校验；批准提交 SHA 与被批准代码 SHA 必然不同，测试用真实临时双版本仓库证明）、要求 `pilot_authorized` + 完整 40 位 `approved_sha` == 本地 **detached** HEAD + 工作树干净 + `task_id` 匹配 + `max_scored_pairs>0` + `gpu_budget>0`；⑤ 生产 checkout 钉死 `/home/ma-user/work/CADRetrieval`；⑥ pinned revision 路径检查；⑦ 共享预算账；⑧ 严格加载。**当前 origin/master 为 `implementation`、预算 0：两个 CLI 的前向模式本地实跑均 exit 2 拒绝（证据见 §6）。**
- **预算账（两卡共享）**：`SharedBudgetLedger` 单文件绑定 `approved_sha`+对数上限+卡秒上限；锁文件（`O_CREAT|O_EXCL`，超时拒绝且**不自动破锁**）内读改写+原子落盘；**每批前向前先扣账并持久化**（崩溃只会多扣不会少扣）；卡秒按 `wall×cards_used` 累计；心跳 30 s，陈旧（>180 s）未完成 invocation 按 `(last_heartbeat-start+2×heartbeat)×cards` 保守收尾；同 shard 活跃双跑拒绝，不同 shard（另一卡）放行。基准与矩阵共用同一本账（`text2cad_m1c_{approved_sha}.json`），benchmark invocation 以哨兵 shard `-1` 登记。
- **dry-run（零前向，预算 0 下允许）**：清单指纹+schema、分片数学、zip SHA/member、checkpoint 结构+172 键严格装载（StubEmbedder，无前向）、BERT 文件 SHA、可选 `--scan-candidates` 全量 8,046 结构复扫、有 transformers 时全 3,644 文本 token 长度审计（>512 即败）。写 `dry_run.json`（存在即拒）。报告 `forward_executed: false`。

## 4. G1/G2 测速器（对应任务第 4 条）

- **固定对集**（已提交配置，provenance 绑定清单 SHA，运行前逐对重核验）：spot 1 对（中位文本×中位 CAD）；批一致 10 对（分位表 [(10,10),(10,50),(10,90),(50,10),(50,50),(50,90),(90,10),(90,50),(90,90),(100,90)]，实测 n_chars∈{71,153,303,469}×n_real∈{23,34,87}，含最长文本×P90 CAD）；吞吐 30 层（文本三分位×CAD 十分位）× 4 文本 × 5 CAD = 600 对，全部具体化且跨档不变。
- **门语义**：`batch-consistency` = 同 10 对 batch=1 对 batch∈{2,5,10}，逐对比较**有效位置 logits**（`t ≤ n_real-2` 行，两头全词表）与最终分数；`trim-check`/`cache-check` = 参考路径（batch=1 全长）对裁剪批/文本缓存批同法比较。预登记容差（见 §5）超限即 exit 3 且对应优化保持关闭；容差永不事后放宽。
- **吞吐**：档 `1,2,4,8`，`--allow-large-batch` 再加 `16,32`；每档同批 pairs、≥2 次 warmup（强制下限）、默认 5 次重复；逐 cell 逐 rep 记录 `preprocess_s`（装配/取数）、`forward_score_s`（前向+传输+评分）、`write_s`（记录落盘）与端到端 rep 墙钟、pairs/s、有效项/s、HBM 峰值（`torch.npu.max_memory_allocated`，CPU 为 null）；OOM 记 `status=oom` 并**停止升档**，不崩账。每个被前向的 pair（含 warmup 与重复）都先扣账；`plan_throughput_forwards` 给出精确计划数供预算预检（测试断言与实际扣账逐数相等）。
- **ETA（纯计算）**：必须给**两份时间窗重叠的**单卡结果（缺一/不交叠即拒；绝不用单卡×2）；按全矩阵 30 格真实对数加权端到端速率，加冷启动/IO/写盘 overhead，乘 ≥1.30；七天门为 `eta_seconds ≤ 604,800`。**注意：精确门限 = 29,319,624×1.30/604,800 = 63.0200 对/秒；计划文本的约数 “63.0” 略低于精确门限（在 63.0 处 ETA 超 168h 约 8 秒，门判不通过）——已在测试中钉死该边界，ETA 报告同时给出实测聚合速率与有效速率两种口径。**

## 5. 优化等价门与已审 B1 代码改动（对应任务第 5 条，**请 Codex 重审**）

**改动理由**：尾部 padding 裁剪需要把交叉注意力 mask 的 tile 维度从固定 271 变为批内最大真实长度−1；已审 `Text2CAD.forward` 写死 `self.cad_seq_len`。改动为加性可选参数（默认 `None` → 原值 271），默认路径逐位不变（新测试 `test_default_unchanged_and_explicit_271_identical` 证明显式 271 == 默认），越界拒绝（`bad_cad_seq_len`）。`NOTICE.md` 已登记。

**合成等价证据（CPU 随机权重，非真实模型）**：
- 裁剪门：全尺寸移植模型（1024/256，8 层）上裁剪批 vs 全长参考，有效位置 logits 9,612 元素在登记容差（rtol 1e-4/atol 1e-5）下 0 违例，实测 max abs 1.4–2.4e-6、max rel ≈1.3e-4（GEMM 形状/kernel 选择的末位 ulp 经 8 层放大），score_total 差 5.5e-6（登记分数容差 atol 1e-4）；小模型批 1 vs 5、裁剪批、缓存批等价门均通过。
- 文本缓存门：假 BERT 的 pad 嵌入刻意为非零（5.0），缓存路径以零回填 padding——等号成立只能依赖掩码链（adaptive 自注意力 kpm + 交叉注意力加性 mask），测试通过即掩码链端到端成立。
- **未证明**：真实 checkpoint + NPU 上的等价（预算 0）。登记容差为**预登记暂定值**（logits rtol 1e-4/atol 1e-5；score rtol 1e-6/atol 1e-4），G1 用同一真实 pair 对照实测确认或拒绝；确认前生产默认 `trim_mode=none`、`text_cache=False`。
- **候选侧前两层缓存**：本轮**未实现**（仅确立门协议）；若要启用须另行设计同 pair 等价门并先审。

## 6. 测试与运行记录（本机实测）

环境：`D:\miniconda\python.exe` Python 3.13.5、torch 2.12.0+cpu、pytest 9.1.1（无 transformers/safetensors/pyarrow）。

```
PYTHONDONTWRITEBYTECODE=1 D:\miniconda\python.exe -m pytest \
  experiments/tests/test_text2cad_model_adapter.py \
  experiments/tests/test_text2cad_vector_scorer.py \
  experiments/tests/test_text2cad_pair_index.py \
  experiments/tests/test_text2cad_frozen_runner.py \
  experiments/tests/test_text2cad_speed_bench.py -q
→ 197 passed (44.6s)
```

分项：adapter 59（B1 原 56 + 新 override 3）、scorer 45（未动）、pair_index 29、runner 40、bench 24。回归全套：`pytest experiments/tests -q` → **403 passed**；4 failed（`test_cadmium_scorer.py::TestRealTokenizerCPU`，缺 transformers）+ 9 errors（`test_m0_manifest.py`，缺 pyarrow）为 Stage A/B1 已登记的既有环境性失败，与本轮无关（本轮未触碰相关文件）。

真实资产零前向实跑（本地，checkout 外输出）：
- builder：`…/scripts/build_text2cad_pair_manifest.py --scan-cad-lengths` 对冻结 M0 v3（query SHA `a0d8504f…`，与云脑审计一致）+ pinned zip：8,046/8,046 候选通过已审 validator（扫描任一失败即中止）；计数 7,934/3,644/8,046/29,319,624 全匹配；artifact SHA 与已提交指纹配置一致；重建逐字节一致。
- dry-run：`run_text2cad_score_matrix.py --mode dry-run` exit 0，`forward_executed: false`，172/172 严格装载+数值相等、BERT 五文件 SHA、zip member 全覆盖。
- 拒绝证据：`--mode score` 与测速 `--mode spot` 在当前状态均 exit 2：`authorization: ACTIVE_TASK.md state is 'implementation', not 'pilot_authorized'; no model forward is permitted`（未建输出目录、未加载权重）。
- `select` 重生成与已提交选择配置逐字节相等；`eta` 用伪造双卡结果（70+70 对/s、窗重叠）→ ETA 75.7h 通过，窗不交叠 exit 2 拒绝。

## 7. 已验证/未验证边界与风险

已验证（本机）：pair 空间数学、分片互斥完备、断点恰一次、账本两卡并发语义（注入时钟）、清单/资产篡改拒绝、真实双版本 Git 授权门、真实资产零前向 dry-run、合成模型上的 batch/trim/cache 等价、ETA 边界算术。

未验证（须等授权）：真实权重+真实文本/CAD 的任何前向；NPU 算子数值；FP32 vs BF16 前向 dtype 决策（默认 FP32；BF16 属 G1 决策项，切换会进 chunk 绑定字段）；batch=1 vs B 与两个优化门在真实设备上的容差确认；BERT 缓存在 NPU 的内存占用（最坏 3,644×98×1024×4B≈1.4 GB，64 GB HBM 无虞）；候选 preload 在云脑的实测耗时（本地机械盘 zip 扫描分钟级）。

风险登记：
- R1 两卡同 ledger 文件依赖锁文件；锁持有者崩溃→后续 run 拒绝并要求人工确认（fail-closed，有意为之）。
- R2 崩溃 chunk 的已扣 pair 不会退还（保守多扣），极端反复崩溃可能提前耗尽预算→停等审查，符合安全方向。
- R3 G2 全六档（含 16/32）计划前向 18,140 对（默认四档 12,060；G1 四门合计 81 对）；预算须由 Codex 在 `ACTIVE_TASK.md` 显式批准。
- R4 心跳/陈旧参数（30 s/180 s）为常量；修改需重审。
- R5 文本缓存把 embedder 换成适配器模块（`_CachedEmbedder`），记录在 chunk 绑定 `text_cache`；断点续跑换标志会因绑定不一致拒绝（有意）。
- R6 zip member 路径含 UID 字符串（存储定位必需）；worker 输入除此无任何 UID/query_id/qrels 语义，AST 卫生测试锁定。
- R7 `.pyc`/缓存在本机测试运行于仓库内生成（`__pycache__`）；服务器生产命令模板已含 `PYTHONDONTWRITEBYTECODE=1` 且所有输出在 checkout 外。

## 8. 云脑路径与生产命令模板（须 Codex 批准 SHA 后）

资产（checkout 外 `/home/ma-user/work/CADRetrieval_runtime/`）：
`models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth`、`models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b/`、`datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip`。

先在云脑以获批 SHA 重建并核验 pair 清单（指纹须与 `experiments/configs/text2cad_pair_manifest_expected_20260930.json` 逐字节一致）：

```
RT=/home/ma-user/work/CADRetrieval_runtime
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RT/hf XDG_CACHE_HOME=$RT/xdg TMPDIR=$RT/tmp
PY=/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python
cd /home/ma-user/work/CADRetrieval   # detached 获批 SHA，工作树干净
$PY experiments/scripts/build_text2cad_pair_manifest.py \
  --query-jsonl <M0 query.jsonl 运行副本> --gallery-jsonl <M0 gallery.jsonl 运行副本> \
  --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93.../cad_seq.zip \
  --out-dir $RT/pair_manifest_v1 --repo-root . --scan-cad-lengths
```

零前向 dry-run（不耗预算）：`run_text2cad_score_matrix.py --mode dry-run --checkpoint … --bert-dir … --cad-seq-zip … --pair-manifest-dir $RT/pair_manifest_v1 --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json --out-dir $RT/runs/<run_id> --repo-root . --shard-id 0 --n-shards 2`（BERT 真实加载/token 审计在云脑环境执行；本机无 transformers 已记录跳过）。

G1（批准后，单卡）：`run_text2cad_speed_bench.py --mode spot|batch-consistency|trim-check|cache-check --selection experiments/configs/text2cad_g1g2_pair_selection_20260930.json --out-dir $RT/runs/<bench_id> --device npu:0 --cards-used 1 --repo-root .`（其余资产参数同上）。

G2（批准后，两卡同时在互斥 shard 上各跑一次 `throughput`，再 `eta` 合并）：
```
# 两卡各自（device npu:0 / npu:1，--cards-used 2）：
--mode throughput --batch-tiers 1,2,4,8 [--allow-large-batch] --reps 5 --warmup-passes 2 …
# 合并（纯 CPU）：
--mode eta --card-result <card0.json> --card-result <card1.json> --production-batch-size <获批档> [--out eta.json]
# exit 0 = 过七天门；exit 3 = 不过（停下讨论）；exit 2 = 输入拒绝
```

全量矩阵（G4，另行批准）：`run_text2cad_score_matrix.py --mode score --shard-id {0,1} --n-shards 2 --cards-used 2 --batch-size <获批档> --out-dir $RT/runs/<run_id> --repo-root .`；断点续跑加 `--resume`。

## 9. 已知缺口

1. B1 已审 `model.py`/`NOTICE.md`/adapter 测试本轮有改动（§5），**B1 审查对该文件失效，须随本轮一并重审**。
2. G1/G2 的任何真实测量都未做（预算 0）；登记容差为暂定值。
3. 候选侧前缀缓存未实现（仅协议）。
4. 云脑 pair 清单重建、dry-run（含真实 BERT 加载与文本 token 审计）尚未在服务器执行。
5. `eta` 的 overhead 默认取两份结果的实测冷启动之和；失败重试余量目前由 1.30 安全系数统一覆盖，若 G2 实测失败率显著，应在 RESULTS 中如实记录并讨论是否上调。

未发现需要改变任务口径的不一致（ETA 精确门限 63.0200 vs 计划约数 63.0 已在 §4 如实登记，按更严的精确数学执行，不算口径变更），故未创建 BLOCKERS.md。

`review_required`
