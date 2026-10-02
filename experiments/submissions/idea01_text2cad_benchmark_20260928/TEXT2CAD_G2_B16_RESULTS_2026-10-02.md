# Text2CAD M1c G2 batch16 双卡真实测速结果报告（2026-10-02）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 子计划：`experiments/plans/IDEA01_M1C_G2_B16_2026-10-02.md`；上位合同 `IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`；启动前审查 `experiments/reviews/IDEA01_M1C_G2_B16_CODEX_CODE_REVIEW_2026-10-02.md`
- 授权依据：origin/master `fe1b2d4fb6de5e0ba01dffe916bc400b6dae6f8d` 的 `ACTIVE_TASK.md`（`pilot_authorized`，仅 batch16 双卡 throughput 各一次，16,200 对 / 2 卡小时 / 仅 npu:0+npu:1；ACTIVE_TASK blob SHA256 `15bd4c8dd99dbe0c360694348d290c6c391b467960a04ce5c79432152026604b`，两卡 bench JSON 内记录一致）
- 运行代码 SHA（approved == actual）：`08a0183c86afc7198eed1c798d289d4b6f0c4a42`（远端干净 detached 此完整 SHA；本报告提交 SHA 见文末回填）
- 固定 run_id：`idea01_text2cad_g2_b16_20261002`；两个 invocation：`20261002T032844Z-1b87df86`（npu:0 / shard0，python PID 2970888）、`20261002T032844Z-49d89578`（npu:1 / shard1，python PID 2970969）

本文件是本轮唯一新增交付文档，未改动任何源码、配置、计划、审查、`ACTIVE_TASK.md` 或既有报告；本轮结束时未做 git push。无任何 R@K / 检索质量结论——本轮仅为吞吐测速与 ETA。

## 1. 事实（Facts）

### 1.1 前置核验（全部通过后才启动；时间 2026-10-02T03:23–03:26Z）

- 本地工作树干净，本地 HEAD == origin/master == `fe1b2d4…`（授权文档提交）；`08a0183…` 与 `fe1b2d4…` 均为 origin/master 祖先链成员。
- SSH `yunnan-share2` BatchMode 免密通过。远端 checkout `/home/ma-user/work/CADRetrieval` 启动前工作树干净（`git status --porcelain` 0 行），原 HEAD 为上一轮 G1 的 `a2219e2…`；`GIT_TERMINAL_PROMPT=0 git fetch origin`（909d655..fe1b2d4）后 `git checkout --detach 08a0183c86afc7198eed1c798d289d4b6f0c4a42`，`git rev-parse HEAD` 与批准 SHA 逐字符一致，detach 后工作树干净。全程未在远端编辑/reset/clean/SCP 任何源码。
- `npu-smi info`（03:23:51Z）：两张 Ascend910 均 `OK`、AICore 0%、无运行进程；无进程状态下的 HBM 基线显示为 3103 / 2872 MB（运行前后相同，属该机基线显示，非本轮占用）。
- 运行根 `runtime/runs/idea01_text2cad_g2_b16_20261002` 不存在；`runtime/runs/` 仅有既往 run（P0/P1/G1 各目录），无同名 sync marker。
- 资产逐项复核：token 审计 `derived/text2cad_text_token_lengths_20261001.json` == `9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8`；detach 树上选择配置 == `fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce`；worker 目录 `pair_manifest_v4` 四文件 == 固定清单 `text2cad_pair_manifest_expected_20260930.json`（manifest `62086beb…2557`、texts `c59b4aef…4a21`、cad_store `2f441110…1380`、cad_length_index `daac66a1…e0c3`）；checkpoint `Text2CAD_1.0.pth` == `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`；BERT 五文件与 `cad_seq.zip` 在位。worker 目录不含 qrels/UID 评价映射，本轮未读取任何 qrels/CSV/正例映射。
- 脚本 `experiments/scripts/run_text2cad_speed_bench.py`、expected-keys 配置、环境 python（3.11.10，既有克隆环境，未安装/升级任何依赖）均在位。

### 1.2 执行与退出状态

- 服务器 `yunnan-share2`；运行目录 checkout 外 `runtime/runs/idea01_text2cad_g2_b16_20261002/{card0,card1,sync}`（启动前逐一确认不存在后新建）；`PYTHONDONTWRITEBYTECODE/HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_HOME/XDG_CACHE_HOME/TMPDIR/OMP/MKL/OPENBLAS` 均按授权命令固定，缓存/临时目录全部指向 checkout 外。
- 两条 throughput 命令与授权模板逐字一致（`--mode throughput --batch-size 16 --batch-tiers 16 --allow-large-batch --warmup-passes 2 --reps 3 --cards-used 1 --device npu:0|1 --sync-dir $RUN/sync --sync-group g2_b16_20261002 --timeout 3600` 包装），仅 `--device` 与 `--out-dir` 按卡不同；完整命令逐字存档于 `launch_command_card0.txt`（SHA `e10c4381…1cb93`）、`launch_command_card1.txt`（`c18b5f39…1cb93` 前缀不同，全文见归档）。card0 先启动、3 秒后 card1；两进程实际启动 epoch 1790911724.674 / 1790911724.680（03:28:44Z 时间戳的 invocation 名源于命令生成时刻）。
- 退出码：**card0 = 0、card1 = 0**（`exit_code.txt` 由包装 shell 在 `timeout 3600` 返回后直接写出，未经管道掩盖）。各自 stdout 仅一行状态 JSON（`status: ok`，`escalation_stop: null`）；stderr 各 1298 字节，逐行核对仅 torch_npu `collect_env` owner UserWarning，非警告行数为 0。
- 运行后（03:38Z，证据文件 `post_run_git_npu.txt`）：远端 HEAD 仍 `08a0183…`、工作树干净、两卡 OK 无进程、无残留 bench 进程。源码树全程零改动。

### 1.3 计划绑定与真实分片

- 两卡 `sync.common_plan_sha256` 均 == `e88343f660ab955dccb1844befd7029362cedd7d89b0dfc49b431d8fc831a603`；`plan_sha256`/sync.local：npu:0 == shard0 `fed82ddfc9f898cdcddbc81cbac9246ee93a9c0808fade1f69301258b40fa661`，npu:1 == shard1 `75302a04e0854733a20a594379387a70eccf422e1ddb646d5dccac813e5c3dc6`，与审查记录逐项一致。
- `sync/` 目录 198 个 marker（99 点 × 2 卡），schema `text2cad-g2-sync-v1`，binding 含 checkpoint/公共计划/git/清单/选择配置五 SHA；payload 记录本地计划 SHA、批尺寸列表与对数。
- **注意（按审查限定如实记录）**：两份 bench JSON 的 generic `environment.shard_id=0, n_shards=1` 是未使用的 matrix RunnerConfig 默认值，**不**代表真实分片。真实分片证据：sync.shard_id 0/1、上述两份不同本地计划 SHA、device→shard 映射、cell 名 `shard0_*`/`shard1_*`、账本 shard_id、以及不相交的 pair 清单（card0 文本索引实测范围 [160,1784] ⊂ [0,1822)，card1 [1879,3590] ⊂ [1822,3644)；两卡 198 个 score 文件两两比对**无一**跨卡相同）。

### 1.4 批次、计数与分数有效性

每卡（两卡逐项相同结构）：

- 33 cell = 3 文本 token 层 ×（10 个 CAD 长度层 + 1 尾批层）；每 cell 恰好 3 个计时 rep（rep 0/1/2）+ 对前两真实样本批的 2 次 warmup；计时 rep 合计 99。
- 批尺寸实测集合 = {16, 14}；三个尾批 cell（每 text tier 一个）`is_tail=true`、`batch_size=14`（真实尾批，非截断/重组）；其余 30 cell 批尺寸 16。计时批数 378/卡。
- 计时对数 6,012/卡；warmup 对数 2,088/卡（每 cell 2×前两批）；**扣账合计 8,100/卡、16,200 总计**，与静态预算 `dual_shard_plan_binding(selection,[16])` 逐项一致。
- score 文件：198 个 jsonl（99 rep × 2 卡），共 12,024 条记录（6,012/卡），字段 `score_x/score_y/score_total/mean_x/mean_y/mean_total/n_valid_*` 全部有限且 `n_valid_total>0`（非有限/非法计数 = 0）；score 名 `two_head_factorized_logprob_sum`，spec `text2cad-vector-score-v1`，FP32、全长 272、无 trim、无文本缓存。
- 确定性佐证：每卡 33 个 cell 的 3 个 rep score 文件全部逐字节相同（同一 cell 重复评分结果一致）。

### 1.5 计时（不含同步等待）与冷启动

| 量 | card0 (npu:0) | card1 (npu:1) |
| --- | --- | --- |
| 冷启动 cold_start_s | 33.739 | 33.424 |
| 测量窗口 measurement window (s) | 85.071 | 84.855 |
| 99 个计时 rep 的 E2E 合计 (s) | 30.292 | 31.775 |
| 其中 forward_score 合计 (s) | 17.881 | 17.979 |
| rep sync_wait：mean / max (s) | 0.1116 / 0.4409 | 0.1108 / 0.4765 |

- 窗口起点均已对齐（barrier 同步后）；计时 rep 的 `window` 字段为该 rep 实际工作区间，`sync_wait_s` 单独记录、不混入 E2E。窗口与 E2E 合计的差值含 warmup 前向、预处理、原子写盘与 barrier 到达差。

### 1.6 双卡窗口重叠（80% 门）

- 99 个同步点逐一按两卡 `window` 计算 `重叠/较短窗口`：**min = 0.99236（tt1_tail_r2），mean = 0.99745，median = 0.99762；低于 0.80 的点数 = 0/99**。
- 最差 5 点（ratio, point, 重叠s, 较短窗口s）：(0.99236, tt1_tail_r2, 0.0945, 0.0952)、(0.99376, tt0_tail_r2, 0.0888, 0.0893)、(0.99379, tt2_d0_r0, 0.2373, 0.2388)、(0.99386, tt2_d9_r1, 0.2398, 0.2413)、(0.99387, tt2_d8_r2, 0.2436, 0.2451)。
-  SyncBarrier 默认 poll 降至 0.001 s 的修复（本 SHA 唯一源码差异）在真实 NPU 负载下生效；未修改 80% 门、timeout=180 或任何数值口径。

### 1.7 峰值 HBM

- card0 `peak_device_memory_bytes` = 1,893,842,944（1.764 GiB，hbm_device=npu:0）；card1 = 1,893,712,896（1.764 GiB，npu:1）。每卡 HBM 65,536 MB，峰值占比约 2.7%。

### 1.8 预算计账（本 SHA 共享账本，未沿用 G1 余额）

- 账本 `runtime/budgets/text2cad_m1c_08a0183c86afc7198eed1c798d289d4b6f0c4a42.json`：**pairs_charged = 16,200 / 16,200（remaining 0）；card_seconds_spent = 168.71 / 7,200**（84.45 + 84.26，约 0.0469 卡小时，上限 2 卡小时）。
- 两个 invocation 各扣 8,100 对，均 `exit_code=0`、`finalized=true`、`note=ok`，PID 2970969/2970888 与本报告记录一致；`stale_invocations_auto_finalized` 为空；失败前向 0、重试 0。
- 只读快照 `RUN/budget_snapshot.json`（chmod a-w，SHA `37fbbe8e…9bb`）；源账本未改动。

### 1.9 eta-g2（纯 CPU，一次，exit 0）

- 命令存档 `eta_g2_command.txt`：`--mode eta-g2`，两个 `--card-result` 为实际 bench JSON，`--production-batch-size 16`，**未传** `--overhead-seconds`（保持测得 cold 与默认 1.30 系数）。stdout 一行 `status: ok`；stderr 仅同类 owner 警告。
- 输出 `RUN/eta_g2_b16_20261002.json`（SHA `d354d5ad…f76d`）要点：
  - `slower_card = npu:1`；`wall_seconds_presafety = 106,017.052`；`safety_factor = 1.3` → **`eta_seconds = 137,822.168`（38.284 小时 ≈ 1.60 天）**；`seven_day_seconds = 604,800`，**`passes_seven_day_gate = true`**。
  - `full_matrix_pairs = 29,319,624`；每 shard 14,659,812 对；每卡生产批数 916,466；`cold_start_seconds_sum = 67.163`（两次实测冷启动之和，已计入）；`extra_overhead_seconds = 0.0`、`overhead_seconds_input = null`。
  - `effective_two_card_pairs_per_s = 212.735` vs `required_two_card_pairs_per_s = 63.022`。
  - 口径（JSON notes 原文）：每卡计算量 = 该卡 shard 各 cell「精确生产批数 × 三 rep 中最慢每批 E2E」之和；写盘/账本时间已在实测 E2E 内不重复计；墙钟取较慢卡，不做理想 2× 池化。
  - 单 rep 尖峰 cell（max 每批 E2E，典型值约 0.10–0.11 s）：shard0_tt0_d9 0.1485、shard0_tt1_d2 0.1509、shard1_tt0_d8 0.1481、shard1_tt1_d4 0.1510、shard1_tt1_d6 0.1557、shard1_tt1_d9 0.1518、shard1_tt2_d3 0.1616、shard1_tt2_d4 0.1617、shard1_tt2_d6 0.1610；shard1_tt1_tail 0.2482（尾批小样本均值放大）。这些按合同公式保守进入 ETA。

### 1.10 三端归档（逐 SHA 核验）

- 云端源目录保留：`yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g2_b16_20261002`（7.9 MB，未删除；`/data/data_dev` 卡住的挂载全程未用）。
- `SHA256SUMS`（云端生成，相对路径，不含自身，415 项；文件 SHA `dcd3d914…5529`）。
- 本地镜像 `D:\projects\IdeaSpark\CADRetrievalIdea1\experiments\runs\idea01_text2cad_g2_b16_20261002`（Git 忽略目录）：scp 取回后 `sha256sum -c` **415/415 OK**。
- 共享 OBS `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g2_b16_20261002/`：目标预先不存在才创建；scp 后首次校验因该机中文 locale 输出「成功」导致 grep 计数为 0，随即以 `LC_ALL=C sha256sum -c` 严格复核 **exit=0、415/415 OK**。cvteam14 仅存储，未占其 GPU。
- 关键产物 SHA256：bench card0 `45835470…f8018`、bench card1 `a890d5e8…de5b0`、eta `d354d5ad…f76d`、budget_snapshot `37fbbe8e…9bb`、post_run_git_npu `30c2630c…46e5`（完整 415 项见归档 SHA256SUMS）。

### 1.11 失败与异常

无。无 OOM、无 NaN/非有限分数、无截断、无资产错配、无 peer 异常、无超时（timeout 3600 未触发）、无预算超额、无重试；`escalation_stop = null`（两卡）。G1 batch32 旧失败记录保持原样，未重命名、未重用旧账。

## 2. 分析（Analysis，与事实分离）

- **80% 并发门以显著裕度通过**：真实 NPU 双进程 99/99 点最小重叠 0.992，远高于 0.80 门槛，与 Codex 独立 CPU 见证（0.999912）方向一致；说明此前 0% 重叠确为轮询分辨率问题而非协议缺陷。审查限定「不能因 CPU 修复默认真实 NPU 重叠通过」已由本轮真实窗口直接验证。
- **ETA 的保守性来源**：合同公式取每 cell 三 rep 中最慢每批 E2E。多数 cell 稳定在 0.10–0.11 s，但 8–9 个 cell 出现单 rep ~0.15–0.16 s 尖峰（最慢卡 npu:1 的 tt2 高层 CAD 段更密集），使这些 cell 计时放大约 50%；若尖峰为一次性事件（如首触编译/主机 I/O），稳态全量可能低于 38.3 h，但本报告不作此假设。
- **吞吐与资源**：实测计时 rep 平均约 190–200 对/s/卡，保守 ETA 口径折合 138 对/s/卡；峰值 HBM 1.76 GiB 仅为单卡 64 GiB 的 ~2.7%，batch16 FP32 全长在 910 上显存远非瓶颈。G1 batch32 的失败是数值一致性问题，本轮不据此推断其原因或结论。
- **同步开销**：rep 级 sync_wait 均值 0.11 s、最大 0.48 s，99 点合计约 11 s/卡，占 85 s 测量窗口约 13%；测量窗口与计时 E2E 合计之差还包含 warmup 前向（每 cell 2×2 批）与预处理/写盘，均在合同允许口径内。
- **确定性**：同 cell 三 rep score 文件逐字节相同，说明同设备同输入 FP32 前向结果可复现；跨卡零相同文件且文本索引区间不交，再次排除「两卡实为单卡重复跑同分片」的解释。
- **与七天门的关系**：`passes_seven_day_gate=true`（137,822 s ≤ 604,800 s）是本轮测速 + 1.30 系数的结论；按合同 §4/§5，七天判断仍须 G3 稳态全图库复核，本轮短计时不能直接启动全量。

## 3. 局限（Limitations）

1. 两份 bench JSON 的 generic `environment.shard_id=0/n_shards=1` 为 matrix RunnerConfig 默认值，不能当真实分片证据（真实分片见 §1.3）；结果审核时不得把该默认字段解释为单卡运行。
2. ETA 与完整矩阵仍存在口径差：本轮测速含原子写盘与预处理，但全量的 chunk 写盘分布、稳态 gallery 流式读取、失败/额外 I/O 未计入（`extra_overhead_seconds=0.0`、`overhead_seconds_input=null`）。
3. 窗口重叠用主机 epoch 计算；NPU 异步 kernel 的 device 侧时间未单独仪表化。
4. 尖峰 cell 成因未隔离（合同未要求）；ETA 已按其最慢值保守计。
5. 本报告不构成任何 R@K/语义检索结论，也不授权 G3 或全量矩阵。

## 4. 建议（Suggestions，非授权）

- 本轮全部合同验收点通过且保守 ETA 38.3 h ≤ 七天门；按合同 §5 自动去向，建议 Codex 独立从原始 rep/权重重算 ETA（不采信单个 passed 字段），复核通过后再规划 G3 稳态全图库验证与全量门判断。
- G3 设计可重点观察 §1.9 列出的尖峰 cell（尤其 npu:1 的 tt2_d3/d4/d6）在稳态下是否复现，以收紧 ETA 上界。
- 建议在 G3/全量阶段保留本轮的 sync marker + 窗口重叠核验流程（本轮证明其在真实负载下有效）。

---

回填：本报告首次提交 SHA `42a5105a9f6d33f068774d33d090cb63ed78f37e`（仅含本文件；本回填为第二次 doc 提交，两者均未 push）。
