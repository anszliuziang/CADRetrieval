# Text2CAD G3 双卡全图库 probe：启动交接（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`；合同 `experiments/plans/IDEA01_M1C_G3_EXECUTION_2026-10-02.md`（上位 G3 probe 合同与 R1/R2 准入继续有效）；批准 `experiments/reviews/IDEA01_M1C_G3_CODEX_CODE_REVIEW_R3_APPROVAL_2026-10-02.md`
- 授权来源：fetch 后 origin/master `936b49259c2dc1c3ca2eba8a0cd8c09af512c764` 的 `ACTIVE_TASK.md`（`pilot_authorized`；checkout 内旧 ACTIVE 不是授权来源，runner 自 origin/master 读取并比对完整 SHA）
- **approved == 实际运行完整 SHA：`487d02b9ac92f5e2338718e45b1f023c308dd332`**（远端干净 detached；启动前后 `git rev-parse HEAD` 逐字符一致、`git status --porcelain` 0 行）
- run_id：`idea01_text2cad_g3_probe_20261002`；服务器 `yunnan-share2` 两卡 npu:0/1；本文件为唯一新增交付文档，未改任何源码/配置/Codex 文档，未 push

## 1. 启动前核验（全部通过才启动；2026-10-02 07:47–07:53Z）

- 本地：工作树干净；HEAD == origin/master == `936b492…`；approved SHA 之后仅 4 个文档改动（CLAUDE/ACTIVE/执行合同/R3 批准），零代码差异。
- 远端 Git：checkout `/home/ma-user/work/CADRetrieval` 原 HEAD `08a0183…`（G2）工作树干净；`GIT_TERMINAL_PROMPT=0 git fetch origin`（fe1b2d4..936b492）后 `git checkout --detach 487d02b9ac92f5e2338718e45b1f023c308dd332`；未 reset/clean/编辑/SCP 任何源文件。授权字段核对：state=pilot_authorized、approved_sha 一致、gpu_budget=12（43200 卡秒）、max_scored_pairs=804600、run_id 一致。
- NPU：`npu-smi info` 两张 Ascend910 均 OK、AICore 0%、无运行进程（HBM 3103/2873 MB 为该机无进程基线显示，同 G2 记录）。
- 旧产物：run 根 `$RT/runs/idea01_text2cad_g3_probe_20261002` 不存在；账本 `$RT/budgets/text2cad_m1c_487d02b9ac92f5e2338718e45b1f023c308dd332.json` 不存在（budgets 目录仅有 G1/G2 旧账本，未触碰）。
- 资产逐项 SHA256 复核（启动前实算）：checkpoint `235b137e…34598`；worker 四文件 manifest `62086beb…02557` / texts `c59b4aef…a21` / cad_store `2f441110…380` / cad_length_index `daac66a1…0c3`；token 审计 `9abfd5b4…27e8`；detached 树上固定选择配置 `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`；BERT 五文件、cad_seq.zip（606,626,050 B）、expected-keys 配置、runner 脚本均在位。环境 python 3.11.10（既有克隆环境，未安装/升级）。

## 2. 启动事实（Fact）

- 命令：与已审 G3 实现报告第 6 节模板逐字一致（绝对路径展开），逐字存档 `$RUN/launch_command_card0.txt` / `launch_command_card1.txt`；两文件逐词 diff 仅 `--device npu:0|1`、`--shard-id 0|1`、`--out-dir card0|card1` 三处不同。固定参数：`--mode score --batch-size 16 --texts-per-chunk 1 --cards-used 1 --n-shards 2`，`timeout 21600`，env：`PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP/MKL/OPENBLAS=4 HF_HOME/XDG_CACHE_HOME/TMPDIR` 指 checkout 外；未设置任何遮蔽设备编号的可见性变量。
- 后台方式：每卡一个 3 行包装 shell（`$RUN/run_cardX.sh`，runtime 证据非源码）执行命令文件并直接落 `cardX.exit_code.txt`（无管道掩盖）；`nohup … < /dev/null &`，SSH 断开后继续。
- 启动时序：card0 先、3 秒后 card1（避 fetch 锁、保留真实并发窗口）。

| 项 | card0（npu:0 / shard0） | card1（npu:1 / shard1） |
| --- | --- | --- |
| 启动 UTC / epoch | 2026-10-02T07:53:51Z / 1790927631 | 2026-10-02T07:53:54Z / 1790927634 |
| launcher PID | 3294241 | 3294321 |
| python PID | 3294246 | 3294325 |
| invocation_id | 20261002T075359Z-4037ac89 | 20261002T075359Z-09217532 |
| scorer started_epoch | 1790927639.165 | 1790927639.436 |

## 3. 启动确认（短轮询，单次 ≤30 秒；07:54–07:56Z）

- 进程：两 launcher 与两 python 均存活；`npu-smi info` 进程表 chip0=PID 3294246、chip1=PID 3294325，与命令 device 一一对应。
- 扣账：共享账本按 approved SHA 路径创建，`max_pairs=804600`、`max_card_seconds=43200.0`、两 invocation 各 `cards_used=1`、heartbeat 正常；07:55Z 已扣 15,600 对、07:56Z 已扣 40,024 对（card0 19,004 + card1 21,020），计费随真实前向推进。
- 首个完整 chunk + timing：card0 已发布 `chunk_s00_t00069_00070.jsonl`、`chunk_s00_t00111_00112.jsonl` 与 `timing_s00.jsonl`；card1 已发布 `chunk_s01_t01836_01837.jsonl`、`chunk_s01_t01858_01859.jsonl` 与 `timing_s01.jsonl`。起始 text_index 69/111、1836/1858 与固定盲选清单每片前两位一致。timing 行字段含完整 scope 指纹（probe_selection_sha256/git_sha/pair_manifest_sha256/checkpoint_sha256/score_name/spec_version/device）与 window/e2e/n_pairs/n_batches/n_scored/n_rejected。
- stderr：两卡仅已知 torch_npu `collect_env` owner UserWarning（同 G1/G2）；stdout 暂无（runner 结束时输出状态 JSON）；exit_code.txt 尚未出现（运行中）。
- 启动后 Git：HEAD 仍 `487d02b9…`、`status --porcelain` 0 行。

## 4. 预算与范围（不变项复述）

两卡合计卡时上限 12 卡小时 = 43,200 卡秒（账本 cap 已核对一致）、6 小时墙钟、每进程 timeout 21,600 秒；总 804,600 对（每卡 50 固定文本 × 完整 8,046 CAD = 402,300 对 / 25,150 批）；batch16/尾 14、FP32、完整 272 行、无 trim/cache、无额外 warmup/验证前向；仅这两次 fresh scorer 调用，不自动重试、不扩大预算。全程未读 qrels/UID 映射/原 CSV/旧模型分数。

## 5. 路径

- 远端 run 根：`yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g3_probe_20261002/`（launch_command_cardX.txt、run_cardX.sh、cardX.launcher.pid、cardX.launch_time.txt、cardX.stdout.log、cardX.stderr.log、pre_launch_git_npu.txt、card0/、card1/；exit_code.txt 与 summary 待进程结束出现）
- 共享账本：`/home/ma-user/work/CADRetrieval_runtime/budgets/text2cad_m1c_487d02b9ac92f5e2338718e45b1f023c308dd332.json`（完成后取回为 `$RUN/final_ledger.json`，复数 budgets，R2 更正路径）

## 6. 未完成项（交 heartbeat 监控；Kimi 本阶段停止等待）

1. 两卡各 50 个完整 chunk + timing + 两个 summary、真实 exit code（启动交接时各完成 2/50 chunk）。
2. 两卡 exit0/finalized 后：取 final ledger → `$RUN/final_ledger.json`；CPU `python -m idea01.text2cad_g3_probe --mode eta-g3`（四文件 worker/token 审计/repo-root/恰两 card summaries/**必填 `--final-ledger`**；exit 0/3/2 语义不变），不覆盖旧产物。
3. 100 完整 chunks/timings/两 summary/final 账本/环境/启动退出证据的本地 `experiments/runs/<run_id>/` 镜像 + cvteam14 共享 OBS `/data/obs_data_liuza/CADRetrieval/runs/<run_id>/` 逐项 SHA256 核验（cvteam14 仅存储；避 /data/data_dev；SHA256SUMS 用 LF）。
4. 本地 `TEXT2CAD_G3_PROBE_RESULTS_2026-10-02.md`（事实/解释/建议分开），原 G3 报告不覆盖。
5. G3 结果与 ETA 经 Codex 独立复算 ≤7 天才准备 G4；Kimi 不自行启动 G4，100 条 probe 不替代完整矩阵。

异常处置（预设）：未知进程/脏 Git/旧产物/错误/OOM/NaN/超限 → 保存证据停下，不覆盖删除/降批/更换模型/重跑；API 或真实外部阻塞 → 保存已做动作与证据并报告，不换 API/不自行扩大预算。
