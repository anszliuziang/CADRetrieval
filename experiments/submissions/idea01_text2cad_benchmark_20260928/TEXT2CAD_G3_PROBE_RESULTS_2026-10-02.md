# Text2CAD M1c G3 双卡全图库 probe 结果收集报告（2026-10-02）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 收集合同：`experiments/plans/IDEA01_M1C_G3_RESULTS_COLLECTION_2026-10-02.md`；上位 G3/R1/R2 合同与 R3 代码审查（`experiments/reviews/IDEA01_M1C_G3_CODEX_CODE_REVIEW_R3_APPROVAL_2026-10-02.md`）继续适用于结果
- 运行授权依据（历史，已结束）：origin/master `936b49259c2dc1c3ca2eba8a0cd8c09af512c764` 的 `ACTIVE_TASK.md`（`pilot_authorized`，804,600 对 / 12 卡小时 / 仅 npu:0+npu:1；ACTIVE_TASK blob SHA256 `73af672329c29bb0e62fb80010bc354be7c25ef13760511b4defd770cfad8de0`，两卡 summary 内记录一致）
- 运行代码 SHA（approved == actual）：`487d02b9ac92f5e2338718e45b1f023c308dd332`（远端干净 detached；启动前、结束后、本收集收尾三次 `git rev-parse HEAD` 均逐字符一致、`git status --porcelain` 0 行）
- 固定 run_id：`idea01_text2cad_g3_probe_20261002`；两个 invocation：`20261002T075359Z-4037ac89`（npu:0 / shard0，python PID 3294246）、`20261002T075359Z-09217532`（npu:1 / shard1，python PID 3294325）
- 收集时 ACTIVE 状态：`review_required` / `approved_sha=none` / GPU 0 / `max_scored_pairs=0`。本轮仅 CPU 收集与分析：**零新模型/BERT/NPU 前向、零重复评分、未改服务器 checkout**；历史 summary/账本保持绑定原 `487d02b9…` 与已结束的 12 卡小时授权，未按当前 0 预算改写

本文件是本轮唯一新增交付文档（原 G3 实现/R1/R2/LAUNCH 报告均不覆盖）；未改动任何源码、配置、计划、审查、`ACTIVE_TASK.md` 或既有报告；本轮结束时未做 git push。本报告**不含任何 probe R@K 或检索质量结论**——804,600 条原始分数留待 Codex 独立遍历审核。

## 1. 事实（Facts）

### 1.1 完成状态独立只读复核（2026-10-02T09:15–09:22Z，全部通过）

- SSH `yunnan-share2` BatchMode 通过（主机 `notebook-ceff1d49-f99f-4086-a3b2-3d6f13ed4225`）。远端 checkout `/home/ma-user/work/CADRetrieval` HEAD == `487d02b9ac92f5e2338718e45b1f023c308dd332`，`status --porcelain` 0 行；未 fetch、未换 checkout、未改任何源文件。
- `npu-smi info`：两张 Ascend910 均 `OK`、AICore 0%、无运行进程（HBM 3102/2872 MB 为该机无进程基线显示，同 G1/G2 记录）。
- run 根 `$RT/runs/idea01_text2cad_g3_probe_20261002`：`card0/`、`card1/` 各 52 文件 = **50 个完整 chunk + 1 timing sidecar + 1 summary**；根目录含启动命令×2、3 行包装 shell×2、launcher pid/launch_time×2、stdout/stderr/exit_code×2、`pre_launch_git_npu.txt`。
- **退出码 card0=0、card1=0**（包装 shell 在命令返回后直接落盘，无管道掩盖）。两卡 stdout 各一行状态 JSON：`{"status":"ok",...,"chunks":50,"stats":{"batches":25150,"pairs_forwarded":402300,"pairs_rejected":0}}`；stderr 各 1,298 字节，逐行仅为已知 torch_npu `collect_env` owner UserWarning（同 G1/G2），无其他内容。
- 两个 summary：`status=ok`、`failure=null`、`chunks_completed_now=50`/`chunks_previously_completed=0`、`pairs_planned=402300`、`stats.batches=25150`、`pairs_rejected=0`；scope 指纹（`probe_selection_sha256`/`pair_manifest_sha256`/`checkpoint_sha256`/`git_sha`/score/spec/device/dtype/trim/cache）两卡与已审固定值逐项一致。
- 启动 PID 3294241/3294321、3294246/3294325 均已不存在（`ps` 无残留）；timing sidecar 各 50 行。

### 1.2 启动与执行记录（与 LAUNCH 交接一致，仅复核未重述全部）

- 两条启动命令逐字存档 `launch_command_card0.txt`/`launch_command_card1.txt`；本轮在本地镜像对两文件逐词 diff：**仅 `--device npu:0|1`、`--shard-id 0|1`、`--out-dir card0|card1` 三处不同**，其余（env 前缀、`timeout 21600`、`--mode score --batch-size 16 --texts-per-chunk 1 --cards-used 1 --n-shards 2`、`--probe-selection` 固定配置路径、`--token-audit` 等）逐字相同。
- 启动时序：card0 2026-10-02T07:53:51Z、card1 +3 秒；scorer started_epoch 1790927639.165 / 1790927639.436。最后 finalize ended_epoch 1790929302.398（08:21:42Z）；自首次启动至最后 finalize 全程 **1,671.4 秒（27.86 分钟）**，远低于 6 小时墙钟与每进程 21,600 秒 timeout。

### 1.3 真实覆盖、对数、批数、冷启动、卡时、HBM 与两卡窗口

| 项 | card0（npu:0/shard0） | card1（npu:1/shard1） | 合计 |
| --- | --- | --- | --- |
| probe 文本（盲选固定） | 50（[0,1822) 分片） | 50（[1822,3644) 分片） | 100 唯一文本 |
| 完整 chunk | 50 | 50 | 100 |
| 每 chunk 结构 | 1 文本 × 全图库 8,046 CAD（8,046 scored 行 + header，eta-g3 逐行通读核验） | 同左 | — |
| 评分对数 | 402,300 | 402,300 | **804,600** |
| 批数（batch16，每 query 503=16×502+14） | 25,150 | 25,150 | 50,300 |
| 拒绝对数 | 0 | 0 | 0 |
| 冷启动 cold_start_s | 33.16292490391061 | 32.88705727108754 | 66.04998217499815 |
| 真实卡秒（账本 charged） | 1,652.94 | 1,662.96 | **3,315.90**（占 43,200 卡秒上限 7.68%） |
| 指定卡峰值 HBM（bytes） | 1,893,844,992 | 1,894,202,880 | —（各卡 65,536 MB 中约 1.76 GiB） |
| 评分窗口（首 query start → 末 query 落盘 end，epoch） | [1790927672.3362107, 1790929292.1016169] = 1,619.77 s | [1790927672.3269393, 1790929302.393881] = 1,630.07 s | **窗口重叠率 1.0**（≥0.8 门） |

- 逐 query E2E（本地对 timing sidecar 只读统计，未触碰分数值）：npu:0 min 24.380 / mean 32.391 / max 47.647 s，50 条合计 1,619.53 s（串行首尾相接）；npu:1 min 24.546 / mean 32.596 / max 44.433 s，合计 1,629.82 s。每条 timing 记录 `n_pairs=8046 / n_batches=503 / n_scored=8046 / n_rejected=0` 无一例外。
- 吞吐（评分窗口口径）：npu:0 248.4 对/s、npu:1 246.8 对/s；两卡速率差 0.6%。
- probe 占完整矩阵比例：804,600 / 29,319,624 = **2.744%**。准确完整 A = **3,644 唯一 strip 文本 × 8,046 CAD = 29,319,624 对**；评价用 7,934 原查询的恢复留到 G4（本轮未读 qrels/UID 映射/原 CSV）。
- score 身份：`two_head_factorized_logprob_sum` / `text2cad-vector-score-v1`，FP32、完整 272 行、batch16/尾 14、trim `none`、text_cache `false`、`texts_per_chunk=1`、每进程 `cards_used=1`、`n_shards=2`——与 G1/G2 已审实现一致。

### 1.4 最终账本（final ledger）

- 生产路径 `$RT/budgets/text2cad_m1c_487d02b9ac92f5e2338718e45b1f023c308dd332.json`（复数 budgets，R2 更正路径）；两卡 finalized 后 `cp` 取为 `$RUN/final_ledger.json`，源文件与副本 SHA256 同为 `43d8e03d16b60f199cd96bddbeb66d1bf99e935da1c38ef58d834b07dd3da61e`（取回前确认 `$RUN` 内无同名旧文件）。
- 内容核验：`approved_sha=487d02b9…`、**恰好两条 invocation**（shard0/shard1，均 `finalized=true`、`exit_code=0`、`cards_used=1`、同主机）、`pairs_charged=402300×2=804600`（== `max_pairs`，remaining 0）、`card_seconds_spent=3315.90` ≤ `max_card_seconds=43200.0`。
- card0 summary 内账本快照为第一完成卡的**合法 pending-peer 形态**（快照时刻 card1 已扣 398,782 对、未 finalize）；card1 summary 快照与最终账本一致（804,600 / 3,315.90）。最终账本为完整性仲裁，两卡计费区间真实包含各自评分窗口。
- budgets 目录内其余 4 份 G1/G2 旧账本未触碰。

### 1.5 资产与环境身份（结束后实算复核，全部一致）

- 结束后（09:21–09:22Z）对关键资产重算 SHA256（exit 0，证据 `post_run_asset_sha256.txt` SHA `bb33fcb33ca1c823662594f22965a3cf9649fb41c7ba2bc7a62656163ce7b671`），与启动前核验及两卡 summary 逐项一致：
  - checkpoint `Text2CAD_1.0.pth`：`235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`
  - BERT 五文件：config `f9a38c9ffa369f288c3206d0bf352dfed11d7bdee93aa5b2de6d7fdc7bff7871`、model.safetensors `28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4`、tokenizer.json `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`、tokenizer_config `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10`、vocab `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
  - `cad_seq.zip`（606,626,050 B）：`9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa`
  - worker 四文件：pair_manifest `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`、texts `c59b4aef8137b1e71c1581291a4c1d2bcbafe36f22d81d01431a5cc02dad7a21`、cad_store `2f441110f013bb614b0bd2fdb7a3b49569a231bbf3f8eaa2c03ac1d725cb1380`、cad_length_index `daac66a102e34fe45c9cb474922076b0d62345ab0c9796f917a0f8085c6ea0c3`
  - token 审计：`9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8`
  - detached 树上固定盲选配置：`f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`（G2 绑定配置 `fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce`）
- 环境（`post_run_environment.txt`，仅元数据读取、零 import 烟雾、零安装）：python 3.11.10；torch 2.7.1（+cpu）；torch_npu 2.7.1.post1；transformers 4.53.1；peft 0.14.0；accelerate 1.0.1；safetensors 0.4.5；numpy 1.26.4；CANN toolkit 8.3.RC1（V100R001C23SPC001B235）；Linux 5.10.0-182 aarch64；既有克隆环境 `/home/ma-user/work/envs/cadretrieval-idea01-torch271` 未安装/升级任何包。
- 结束后 Git/NPU（`post_run_git_npu.txt`）：HEAD 仍 `487d02b9…`、工作树干净、两卡 OK 无进程、无残留 scorer/eta 进程。本收集收尾（09:35:54Z）第三次复核同样干净。

### 1.6 CPU eta-g3 执行（真实 CLI/stdout/stderr/exit）

- 准备：`cp` 最终账本 → `$RUN/final_ledger.json`（§1.4）；确认 `$RUN` 内无任何 `eta_g3_*` 旧产物。准确 CLI 逐字存档 `eta_g3_command.txt`（SHA `711a0b2a4e4c66a811f698438f045473005ea294f0c0cea63b3c141dbf9abbf8`）：

```
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/ma-user/work/CADRetrieval/experiments/src /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python -m idea01.text2cad_g3_probe --mode eta-g3 --pair-manifest-dir /home/ma-user/work/CADRetrieval_runtime/pair_manifest_v4 --token-audit /home/ma-user/work/CADRetrieval_runtime/derived/text2cad_text_token_lengths_20261001.json --repo-root /home/ma-user/work/CADRetrieval --card-summary /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g3_probe_20261002/card0/summary_20261002T075359Z-4037ac89.json --card-summary /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g3_probe_20261002/card1/summary_20261002T075359Z-09217532.json --final-ledger /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g3_probe_20261002/final_ledger.json --out /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g3_probe_20261002/eta_g3_20261002.json
```

- 与 R1 报告 §5 模板逐字一致（恰两个 `--card-summary`、必填 `--final-ledger`、四文件 worker/token 审计/repo-root；未传 `--overhead-seconds`，即 None=仅实测冷启动，不改系数/上限）。
- 执行：纯 CPU 后台进程（bash PID 3397902），<25 秒完成。**exit=0**（`eta_g3_exit_code.txt`）；stdout 一行（`eta_g3_stdout.txt`）：`{"status":"ok","result":".../eta_g3_20261002.json","eta_seconds":103866.44651402211,"passes_seven_day_gate":true}`；stderr 0 字节。
- exit 0 的语义（已审实现）：committed 选择配置重算绑定 + token 审计 + G2 绑定 + 两卡 summary/账本快照/最终账本全链核验 + **全部 100 个真实 chunk 逐行通读并对审记 worker 长度索引逐行等值核验** + timing↔chunk 一一对应 + 窗口重叠 ≥0.8 + ETA ≤ 604,800 s，全部通过；任一拒绝为 exit 2、超门为 exit 3，均未发生。
- 输出 `eta_g3_20261002.json`（SHA `71ea8c81c9d4cc273826e8828ecd0ed2c76a1e04d10adafa27170fb65840bea2`，LF、3,476 B；`--out` 拒绝覆盖，本次为首次创建）。

### 1.7 全量 ETA 数值（eta_g3_20261002.json 原文关键字段）

- `full_matrix_pairs=29,319,624`；`probe_pairs_total=804,600`；`schema=text2cad-g3-eta-v1`；`safety_factor=1.3`；`seven_day_seconds=604,800`；`passes_seven_day_gate=true`。
- `overhead_seconds_input=null` → `extra_overhead_seconds=0.0`（仅两卡实测冷启动 66.05 s 计入，无额外 I/O/失败余量输入）。
- **各层最慢 query × 全层数**（每卡 = Σ_层 最慢完整 probe query E2E × 该卡分片该层完整文本数 + 本卡实测冷启动；永不理想倍乘单卡）：

| 卡 | 层 | 层完整文本数 | probe 数（=计划） | 最慢 query text_index | 最慢 E2E (s) | 层成本 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| npu:0 | 0 | 605 | 17 (=17) | 933 | 47.64747166633606 | 28,826.720 |
| npu:0 | 1 | 613 | 16 (=16) | 387 | 37.25734782218933 | 22,838.754 |
| npu:0 | 2 | 604 | 17 (=17) | 1212 | 46.686471939086914 | 28,198.629 |
| npu:1 | 0 | 610 | 17 (=17) | 2786 | 41.73798894882202 | 25,460.173 |
| npu:1 | 1 | 602 | 16 (=16) | 2958 | 39.50636267662048 | 23,782.830 |
| npu:1 | 2 | 610 | 17 (=17) | 3448 | 44.43339157104492 | 27,104.369 |

- 每卡：npu:0 compute 79,864.104 + cold 33.163 = **completion 79,897.267 s**；npu:1 compute 76,347.372 + cold 32.887 = completion 76,380.260 s。
- **慢卡 npu:0**；`wall_seconds_presafety=79,897.26654924778`；`window_overlap_fraction=1.0`。
- **全量 ETA = 79,897.2665 × 1.30 = 103,866.4465 s = 28.8518 小时 ≤ 604,800 s（七天门通过，exit 0）**。
- 输入绑定（eta JSON `inputs`）：两 card summary 路径、final ledger 路径、G3 选择 `f8521598…4985`、G2 选择 `fe42096c…1dce`、pair manifest `62086beb…2557`。

### 1.8 归档与三处逐项 SHA 核验

- run 目录共 131 文件（130 产物 + `SHA256SUMS`），407.9 MB。`SHA256SUMS` 130 条、全 LF（0 CR 行）、无自引用、相对 `./` 路径；其自身 SHA256 = **`59991e8a7e21a4ad664a7ed5d605fd30d89a8b89c628f5c2e337495e43e3702f`**。
- 云脑本地核验：`cd $RUN && sha256sum -c SHA256SUMS` exit **0**、0 FAILED（临时文件写 checkout 外 /tmp，退出码直接落盘读取，未用管道尾命令掩盖）。
- 本地镜像：`scp -r`（exit 0）取回 `D:\projects\IdeaSpark\CADRetrievalIdea1\experiments\runs\idea01_text2cad_g3_probe_20261002\`（取回前确认本地无同名目录；Git 忽略）；本地 `sha256sum -c` exit **0**、130 OK、0 FAILED。
- 共享 OBS 归档：`scp -r`（exit 0）至 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g3_probe_20261002/`（上传前确认 OBS 无同名目录、未触碰任何既有 run；cvteam14 仅存储，未用其 GPU、未碰 `/data/data_dev`）；OBS 上 `sha256sum -c` exit **0**、130 成功、0 FAILED；OBS 端 `SHA256SUMS` 文件 SHA 与本地逐字符一致（`59991e8a…702f`）。
- 关键单文件 SHA（本地镜像实测，与清单一致）：final_ledger `43d8e03d…a61e`、eta JSON `71ea8c81…bea2`、eta CLI `711a0b2a…bbf8`、card0/1 summary `fb4aceb6…4302`/`bfaa6025…05eb`、timing `4944a4b4…1c67`/`6a14998f…0b3b`、pre/post Git-NPU `1f608d58…6a24`/`d6750039…45ca`、环境 `30ab3f30…93a3`、资产复核 `bb33fcb3…b671`（全量 130 条见归档 `SHA256SUMS`）。
- 服务器原 run 目录保持原样（131 文件，09:35:54Z 复核）；远端 checkout 全程零改动。

## 2. 解释（Interpretation，与事实分开）

- **两卡真实并发且性能对称**：窗口重叠率 1.0、 charged 卡秒差 0.6%（1,652.94 vs 1,662.96）、逐 query 均值差 0.6%（32.391 vs 32.596 s），无互扰迹象；probe 的 248 对/s/卡可作为同机型同 score 配置的实测下限参考。
- **ETA 的保守性来源清晰**：慢卡墙钟 79,897 s 中，每层按该层**最慢** probe query 定价（npu:0 折合 43.85 s/文本），相对实测均值 32.39 s/文本已有约 1.35 倍余量，再乘 1.30；均值速率直推的同口径下界约 16.4 小时/卡。28.85 小时不是均值外推，而是「逐层最坏 + 慢卡 + 1.30」的工程上界。
- **与 G2 保守 ETA 的关系**：G2 的 38.28394 小时为测速外推保守值；本轮 G3 的 28.8518 小时是**生产 score 路径上、含原子落盘在内的真实全图库 probe 实测**估计。两者独立地均 ≤ 604,800 s。按 ACTIVE 口径，仍需 Codex 对原始结果独立复算后才进入 G4 规划。
- **预算量级（我的推导，非代码输出）**：若两卡并发各跑满各自 completion 估计，全量隐含卡秒 ≈ (79,897.27+76,380.26)×1.30 ≈ 203,161 卡秒 ≈ **56.4 卡小时**，约为 G3 授权 12 卡小时的 4.7 倍——G4 预算需按此量级另批，不能复用 G3 已结束的授权。
- **证据链自洽**：eta-g3 以 exit 0 通过，意味着其 fail-closed 链（100 chunk 逐行对审记长度索引等值、timing↔chunk 一一对应、最终账本恰两 invocation 且 804,600 对/3,315.90 卡秒在 cap 内、scope/资产/代码/选择五指纹绑定 committed 值）全部成立；ETA 输入即真实运行的已核验输出，不是声明值。
- probe 未覆盖的 3,544 条文本中，长度分层各层的真实最慢 query 可能比 probe 抽样更慢；ETA 未对该风险另加系数（抽样按 token 长度等距盲选，端点含最短/最长）。

## 3. 局限（Limitations）

1. probe 仅覆盖完整矩阵 2.744%（100/3,644 唯一文本 × 全 8,046 CAD）；**100-query probe 不替代完整 29,319,624 对矩阵**，本报告不含也不支持任何 R@K/语义有效性结论；804,600 条分数的独立遍历审核属 Codex。
2. ETA 为工程估计：假设两卡在约 28.9 小时墙钟内独占并发可用；`overhead_seconds` 未给（仅实测冷启动），额外 I/O、长时热漂移、失败重试余量全部只在 1.30 系数内；全量 3,644 文本的 chunk 写盘分布与稳态 gallery 读取未实测。
3. 原版训练 label_smoothing=0.1，双头分数不称 CAD 几何精确 NLL；分数的语义有效性须在全量矩阵 + qrels 评价阶段判断。
4. card0 summary 的账本快照为合法 pending-peer 形态（对卡当时未 finalize）；完整性以最终账本为准（两条 invocation 均 finalized/exit0/精确扣账）。
5. 窗口与重叠率按主机 epoch 计算；NPU 异步 kernel 的 device 侧时间未单独仪表化。
6. 评价用 7,934 原查询恢复与本阶段分离，留到 G4；本轮全程未读 qrels/UID 映射/原 CSV/旧模型分数，也未按分数调整任何清单/方法/阈值/系数。
7. 历史 summary/账本保持绑定原 `487d02b9…` 与已结束的 12 卡小时授权；当前 ACTIVE（review_required/GPU0/approved none）不回写它们，本收集消耗 0 GPU、0 新前向。

## 4. 建议（Suggestions，非授权）

- 建议 Codex 按 ACTIVE 独立遍历 804,600 条实际 score/索引/有效长度、最终账本、timing 与 ETA，并抽验 OBS 端 SHA（清单 `59991e8a…702f`）后，再裁决 G4 全量；本轮收集不替代该复核。
- 若进入 G4：预算量级建议按 §2 推导的约 56.4 卡小时 + 余量另批（G3 的 12 卡小时授权已结束）；保持原 score/full 272/FP32/batch16/无 trim/无 cache 与固定盲选清单不变，保留账本 + timing sidecar + 包装 shell 直接落 exit 的证据模式。
- G4 阶段再恢复 7,934 原查询并只在评价模块读 qrels；gallery metadata 不得夹带 caption。
- probe 各层最慢 query（§1.7 表）在全量稳态下是否复现可观察，以收紧未来 ETA 上界，但**不应**据此在见分后回调任何阈值/系数。

---

回填：本报告首次提交 SHA `de02bb4aa248dc37c3f4077b70193d5681d2e807`（仅含本文件；本回填为第二次 doc 提交，两者均未 push）。
