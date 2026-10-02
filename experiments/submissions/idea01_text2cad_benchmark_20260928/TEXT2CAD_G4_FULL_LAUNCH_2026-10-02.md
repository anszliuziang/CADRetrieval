# Text2CAD G4 正式全量启动交接（LAUNCH）

2026-10-02。执行代理：本地 Claude Code / 官方 Kimi K3。本报告只覆盖**启动事实与启动后只读确认**，不含全量结果、不含检索精度结论。

- 任务：`idea01_text2cad_benchmark_20260928`
- 上位计划：`experiments/plans/IDEA01_M1C_G4_EXECUTION_2026-10-02.md`
- 授权审查：`experiments/reviews/IDEA01_M1C_G4_CODEX_FULL_APPROVAL_2026-10-02.md`
- 授权来源：`origin/master:experiments/ACTIVE_TASK.md` @ `67e89a20eb6fc3b35d27c3ba2363da4311d9113c`（state=`pilot_authorized`）
- 批准运行 SHA：`56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`
- run_id：`idea01_text2cad_g4_full_20261002`
- 本报告提交 SHA：`（提交后回填）`

## 一、事实（启动前核验）

本地（Windows 仓库根）：

- `git rev-parse HEAD` = `67e89a20eb6fc3b35d27c3ba2363da4311d9113c`，`git status --porcelain` 为空。
- `git fetch origin` 后 `origin/master` = 同一 `67e89a2`。
- `origin/master:experiments/ACTIVE_TASK.md` 与本地工作树文件 sha256 相同：`53b305c2b4407b16b09b85f769e218b50fdac983a6b5a7d34c204ce170de8ef9`；关键字段 `pilot_authorized` / approved `56ac0f8…` / 336 卡小时（1209600 卡秒、604800 秒墙钟）/ max 29319624 对 / run_id 一致。

远端 yunnan-share2（只读 preflight，2026-10-02T13:10:41Z epoch 1790946641）：

- checkout `/home/ma-user/work/CADRetrieval`：`git rev-parse HEAD` = `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`（批准值，detached），porcelain 为空。未执行 reset/clean，未编辑任何文件。
- `GIT_TERMINAL_PROMPT=0 git fetch origin` 成功（`92db5b1..67e89a2`），远端 `origin/master` = `67e89a2`；从其读出的 ACTIVE 关键字段与授权一致。运行 HEAD 保持 detached `56ac0f8`，未切到 master。
- `npu-smi info`：两张 Ascend910 均 `OK`，`No running processes found`（启动前 HBM 显示 3103/2872 MB 驻留、无进程，与历次门状态一致）。
- 无 text2cad/g4_launch 残留进程。
- 磁盘：`/home/ma-user/work`（runtime 与本地同一文件系统）可用 4.0T ≥ 100GB；`/proc/mounts` 仅见 `/data/data_dev`（按约定不触碰）。共享权威 OBS 经 cvteam14 只读核验：`/data/obs_data_liuza` 可用 256T ≥ 100GB，旧 witness run 目录原样未动。
- fresh 检查全部通过：run 根、`….controller.nohup.log`、`….controller.nohup.pid`、`budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json` 及 `.lock` 均不存在；budgets 目录仅有 G1/G2/G3 时代其他 SHA 的旧账本（不动）；runs 下仅有旧 CPU 见证目录 `m1c_g4_posix_cpu_20261002`（不复用不改写）。
- Python：`/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python` = 3.11.10（spec10c18c17 环境 warm reuse，未装包未改环境）。
- SSH 运输按既定规约 `tr -d '\r' | bash -s`，本轮无运输失败；未在服务器写任何 shell/源码文件。

## 二、事实（一次启动）

启动命令（与执行合同第 4 节逐字一致，外层加 fresh 守卫与 PID 落盘；仅此一次）：

```bash
PYTHONDONTWRITEBYTECODE=1 \
HF_HOME=/home/ma-user/work/CADRetrieval_runtime/hf \
XDG_CACHE_HOME=/home/ma-user/work/CADRetrieval_runtime/xdg \
TMPDIR=/home/ma-user/work/CADRetrieval_runtime/tmp \
nohup /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  /home/ma-user/work/CADRetrieval/experiments/scripts/run_text2cad_g4_launch.py \
  --approved-sha 56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719 \
  > /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g4_full_20261002.controller.nohup.log 2>&1 < /dev/null &
```

- 启动前守卫再次核验 HEAD=批准值、树干净、四个 fresh 路径不存在、cache 目录存在；全部通过后才 nohup。
- pre-launch UTC `2026-10-02T13:11:50Z`（epoch 1790946710）；外层 controller PID **3682036** 已写入 `….controller.nohup.pid`；随后 ps 确认存活；SSH 断开后其 PPID 已为 1（nohup 生效）。
- 外层 nohup 日志截至 13:15:19Z 为 0 字节（controller 经 `controller_events.jsonl`/`controller_launch.json` 落盘，stdout 被缓冲；如实记录）。

controller 落盘事实（`controller_launch.json` / `controller_events.jsonl`）：

- `common_start_epoch` = 1790946711.70277（`2026-10-02T13:11:51.70Z`）；`common_deadline_epoch` = 1791551511.70277（`2026-10-09T13:11:51.70Z`）；差值恰 604800.0 秒；`term_at` = deadline−90s、`kill_at` = deadline−30s（grace 在预算内）。
- `git` 块：`head_sha`/`approved_sha` = `56ac0f8…`，`authorization_ref_sha` = `67e89a2`，`active_task_blob_sha256` = `53b305c2…`（与本地一致），`detached`=true，`worktree_clean`=true，`task_state`=`pilot_authorized`，预算字段一致。
- 两条 spawn 事件：`removed_device_mask_vars` = []（未遮蔽设备）；`spawn_error`/`record_error` 均 null；stagger 3.0s。
- plans：shard0 `npu:0` 文本区间 [0,1822)，shard1 `npu:1` 区间 [1822,3644)；各 `n_chunks` 1822、`n_pairs` 14659812、`n_batches` 916466、batch 16、尾批 14、`texts_per_chunk` 1。
- `npu_preflight`：两芯片 OK、0 进程（原始输出已存）；`runtime_free_bytes` ≈ 4.30 TB。

两 fresh scorer（真实 PID 与 NPU 一一对应，npu-smi 实测 PID 3682124→Chip0、3682194→Chip1）：

| shard | PID | device | ledger invocation_id | started_epoch | out-dir |
| --- | --- | --- | --- | --- | --- |
| 0 | 3682124 | npu:0 | `20261002T131158Z-c3812433` | 1790946718.99 | `…/card0` |
| 1 | 3682194 | npu:1 | `20261002T131159Z-6ce644de` | 1790946719.34 | `…/card1` |

两 scorer 完整 argv 已由 controller 存入 `controller_launch.json`（`cards.0.argv` / `cards.1.argv`），并与 `ps` 实际命令行逐字一致：`--mode score`、固定 checkpoint/BERT/cad_seq.zip 三个 2026-09-30 钉住资产路径、`--pair-manifest-dir …/pair_manifest_v4`、`--expected-keys-manifest` 指向 checkout 内配置、`--batch-size 16 --texts-per-chunk 1 --cards-used 1 --device npu:{0,1} --shard-id {0,1} --n-shards 2 --repo-root /home/ma-user/work/CADRetrieval`。argv 中**无** probe-selection / token-audit / resume / cache 类参数。

共享账本 `budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json`：

- 恰 **两条** invocation，`cards_used` 均 1，`approved_sha` 正确；`max_pairs` = **29319624**，`max_card_seconds` = **1209600.0**；`finalized` 均 false，30 秒心跳正常（13:13:48Z 与 13:15:19Z 两次采样均在推进）。
- `pairs_charged` 进度采样：13:13:48Z 合计 36632；13:15:19Z 合计 79630（shard0 38968 / shard1 40662）。`card_seconds_spent` 仍为 0.0（按既有账本语义在后续结算，启动阶段如实记录）。

## 三、事实（首 chunk / 计时 / 健康）

首个完整 chunk 两卡均已落盘并通过只读检查（13:13–13:15Z 采样）：

- `card0/chunk_s00_t00000_00001.jsonl` 与 `card1/chunk_s01_t01822_01823.jsonl` 各 **8047 行**（1 行 chunk_header + 8046 条 pair 记录）。
- chunk_header：`n_pairs` 8046、`n_scored` 8046、`n_rejected` **0**；`git_sha` = `56ac0f8…`；`checkpoint_sha256` = `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`（两卡一致）；`pair_manifest_sha256` = `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`（两卡一致）；`torch_dtype` float32、`trim_mode` none、`text_cache` false；`score_name` = `two_head_factorized_logprob_sum`，`spec_version` = `text2cad-vector-score-v1`；shard1 pair_index 起点 14659812 = 1822×8046，分片边界正确。
- 非有限扫描：两首 chunk `NaN|Infinity|-Infinity` 计数均为 **0**；`reject/error` 词条各仅命中 header 的字段名 `"n_rejected": 0` 一次，无实际拒绝/错误记录。抽样 pair 记录 `status` = scored、分数为有限负值、含 `n_valid_*` 诊断字段。
- 首 chunk 完成时间（文件 mtime，UTC）：card0 `13:12:57.79Z`、card1 `13:13:01.57Z`；即各自 scorer 启动后约 59–62 秒（含模型/BERT 加载）。
- 稳态节奏（前 4–5 个 chunk mtime 间隔）：card0 约 38s/31s/43s，card1 约 42s/34s/28s/32s；至 13:15:19Z card0 完成 4 chunk、card1 完成 5 chunk。主机名 `notebook-ceff1d49-f99f-4086-a3b2-3d6f13ed4225`。
- stderr：两卡各仅 4 行已知 `torch_npu collect_env` owner `UserWarning`（与历次门相同），无其他 stderr 输出；无真实错误、无 NaN、无拒绝。
- 启动后复核：远端 `git rev-parse HEAD` 仍 = `56ac0f8…`，porcelain 为空。

## 四、解释（与事实分开）

- 授权链三路一致：本地 `67e89a2`、远端 fetch 后 `origin/master`、controller 落盘的 `authorization_ref_sha`/`active_task_blob_sha256` 均指向同一授权；运行代码身份 `56ac0f8` 在 HEAD、controller 记录与两个 chunk_header 三处一致。
- 进程拓扑符合合同：一个外层 controller（nohup、PPID 1）+ 每卡恰一个 fresh scorer，设备不遮蔽、错峰 3 秒、共同 deadline；账本两条 invocation 与 ps/npu-smi 三方 PID 互相对应。
- 早期观察节奏（约 28–43 s/chunk）若持续，单卡 1822 chunk 的量级约十几到二十余小时，低于正式保守 ETA 38.28394 小时，更远低于 604800 秒硬墙钟；但这只是**前 4–5 个 chunk 的局部观察**，不构成完成时长承诺，长运行中的波动、checkpoint/IO 阶段差异均未计入。
- 计费/结算字段（`card_seconds_spent` 等）由已审 controller 按既有语义维护，启动阶段为 0 属预期行为，不代表漏记。

## 五、限制与未完成事项

- **启动成功 ≠ 全量完成 ≠ 检索精度**。本轮只验证了两卡各首个完整 chunk 与进程/账本/deadline 正确性；剩余每卡约 1817–1818 个 chunk 尚未评分。
- 正式 ETA 仍取 max(G2,G3)=38.28394 小时的保守值；上述早期节奏只是观察，不能用于提前宣布完成时间。
- 外层 controller nohup 日志当前 0 字节（输出缓冲/事件走 jsonl）；若 controller 退出时有 stderr 将落入该文件，后续监控时复核。
- stdout 日志当前为空（Python 缓冲），chunk 级证据以 jsonl 与 mtime 为准。
- 未做、也不允许做：预热、重试、空查询、probe-selection/token-audit/resume、改参数、重启或重复评分、运行中 fetch/detach/编辑远端 checkout、触碰旧 G1/G2/G3/CPU 目录与旧账本。
- 后续（非本轮授权）：按合同每 30 分钟只读监控；两片各 1822 chunk、两 exit0/finalized、全 29319624 对零拒绝后，收集完整矩阵/最终账本/环境/退出/计时/失败日志，本地+OBS 逐 SHA 核验，再按上位合同另立独立 CPU 评价。
- 本报告提交后不 push；Codex 复核后统一推送。

## 六、复现证据指针

- 远端 run 根：`/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g4_full_20261002/`（`controller_launch.json`、`controller_events.jsonl`、`card{0,1}.pid.txt`、`card{0,1}.launch_time.txt`、`launch_command_card{0,1}.txt`、`card{0,1}.{stdout,stderr}.log`、`card{0,1}/chunk_*.jsonl`）。
- 外层记录：`….controller.nohup.log`（当前空）、`….controller.nohup.pid`（3682036）。
- 账本：`/home/ma-user/work/CADRetrieval_runtime/budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json`。
- 授权：`origin/master:experiments/ACTIVE_TASK.md` @ `67e89a20eb6fc3b35d27c3ba2363da4311d9113c`；批准运行 SHA `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`。
