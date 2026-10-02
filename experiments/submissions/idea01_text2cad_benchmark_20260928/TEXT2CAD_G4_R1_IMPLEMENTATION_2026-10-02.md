# Text2CAD G4 R1：启动控制器生命周期最小修复实现报告（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`；子合同 `experiments/plans/IDEA01_M1C_G4_R1_LIFECYCLE_FIX_2026-10-02.md`；针对首次审查 `experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R1_2026-10-02.md` 的两个 Important 与一个 Minor
- 基线：`ca709757b39e71b7782ac12acb4594dcb222f7ab`；**本轮实现提交（代码+测试）：`226c4108f57e332ebfe967df24980ccc35c0595b`**；本报告提交 SHA 见文末回填
- 本轮零模型/BERT/NPU 前向、零网络、零安装包、零远端改动（未 fetch/detach/SCP）；不改任何既有 scorer/loader/math/ledger/worker/config/env/Codex 文档/旧报告；评分方法、资产、full 固定 argv、29319624 对、b16 尾 14、FP32、full272、无 trim-cache、336 合计卡时、共同 604800 秒均不变
- 状态：**review_required**。完整 SHA 经 Codex 复审并另写 ACTIVE 前，真实评分额度始终 0；Kimi 不自行启动 G4、不占卡

## 1. 改动文件（事实；仅此三个，git 证据）

| 文件 | 变更 | 用途 |
| --- | --- | --- |
| `experiments/src/idea01/text2cad_g4_launch.py` | +约340/−50 行 | 两 Important + npu-smi Minor 修复（控制器模块） |
| `experiments/scripts/run_text2cad_g4_launch.py` | 文档串 | 生产 nohup 示例补 `PYTHONDONTWRITEBYTECODE=1` 与 checkout 外 cache/tmp 前缀 |
| `experiments/tests/test_text2cad_g4_launch.py` | +约460 行 | 新故障复现/进程组测试；CommonDeadline 注入 group_signal_fn；npu-smi mixed 测试 |

`git diff ca70975..226c410 --stat`：3 files changed, 824 insertions(+), 52 deletions(-)；无其他文件进入提交。原始 CPU 证据镜像（git 忽略）：`experiments/runs/m1c_g4_r1_cpu_witness_20261002/`（witness 驱动+JSON+两套件日志+SHA256SUMS）。

## 2. Important 1 修复映射（事实）：spawn 后写证据异常不再失去监管

审查指出的两条失控路径均在 `launch()`，修复后语义如下（括号为新测试）：

1. **Popen 成功与后续记录失败分离**（spawn 循环 try/except/else）：try 块只覆盖 launch_command 写入、cache 目录、stdout/stderr 打开与 `popen_fn` 本身——这些失败属**真实 spawn 失败**，该卡无活进程，写 NONE 退出是诚实且最终的（原语义保留，`test_spawn_failure_card1_keeps_card0` 仍过）。Popen 一旦成功进入 else：pid/launch_epoch 先落内存，pid.txt/launch_time.txt 任一写失败只记入 `card.record_error`（`PostSpawnFaultTest.test_pid_write_failure_keeps_real_exit_and_peer`——即 Root 注入 `card0.pid.txt` 场景的修复后复现）。**活进程不再被预写 NONE 假退出**，exit 文件每卡全程只写一次（reap 当下真实码），旧"先 NONE 后 reap 撞 fail-if-exists  RunnerError"的双重写入路径不复存在。
2. **controller_launch.json 写失败不再抛出 wait loop**：记入 `launch_record_error`，两卡继续在共同 deadline 下受监管至真实退出（`test_launch_record_write_failure_children_still_supervised`——Root 第二注入场景复现）。
3. **event 日志失败容错**：`events()` 闭包捕获异常记入 `event_errors`（上限 5 条），绝不因日志磁盘故障中断监管（`test_event_log_failure_keeps_supervision`）。
4. **exit 文件写失败不炸 wait loop**：`_record_card_exit` 先落内存再尽力落盘，失败记 `record_error`，同伴监管不受影响的语义保持（普通一卡失败 peer 正常、不自动 retry——既有 `test_one_card_fails_peer_not_misreported` 仍过）。
5. **末次 Git 读取异常**：`recheck_git_state` 包 try，失败记入 `post_launch_git.error` 且 status=failed；此时全部孩子已 reap，无物可遗留（`test_final_git_recheck_failure_still_records_exit`）。
6. **磁盘全故障底限**：`controller_exit.json` 实在写不出时，先保证所有孩子 reap、真实退出已尽力落盘，再以 RunnerError 退出（main → exit 2），**不伪称 ok**（`test_exit_record_write_failure_raises_after_children_reaped`）。
7. status=ok 条件扩展为：两卡真实 exit0 ∧ 无 deadline 强制 ∧ 无 spawn_error ∧ 无 record_error ∧ 无 leftover_descendants ∧ 无 launch_record_error ∧ 无 event_errors ∧ 末次 git 复核通过。exit record 新增 `launch_record_error`/`event_log_errors`/`process_group_supported` 与逐卡 `record_error`/`leftover_descendants`/`group_cleanup` 字段（schema_version 仍为 text2cad-g4-launch-v1，纯增字段）。

## 3. Important 2 修复映射（事实）：自有进程组 TERM/KILL 与父退出后后代收尾

1. **平台分界**：`_PROCESS_GROUP_SUPPORTED = os.name=="posix" and hasattr(os,"killpg")`；`_posix_group_signal(pgid, sig)` 只用 `pgid==card.pid`——控制器对每个卡都以 `start_new_session=True` 创建子进程，孩子是自身 session/进程组组长，组内只含它与其后代，**不可能碰到其他作业**。Windows 无 killpg，`group_signal_fn` 解析为 None，全链路降级为 PID-only 并把降级如实写进证据（见 §5 边界）。
2. **deadline 两路径组信号**：wait loop 的 TERM/KILL 改经 `_signal_card`：POSIX 对整个组发信号（组长与后代同时收到），事件 `term_sent`/`kill_sent` 增记 `signal_scope`（`process_group`/`pid_only`/`already_gone`）；组已空或信号异常回退 PID 句柄并如实记 scope。TERM@deadline−(60+30)s、KILL@deadline−30s 的预算内宽限不变。
3. **父退出后后代 drain**（新 `_drain_card_group`，在**所有**父进程 reap 之后执行，同伴监管绝不被拖延）：对每组先 signal-0 探测；空→`empty`；settle 窗口（默认 10 s）内自然排空→`drained_within_grace`（不判异常）；超时仍存活→`leftover_descendants=True`、事件 `descendants_detected`→组 TERM→`terminated`→必要时组 KILL→`killed`，KILL 后仍不空→`survivors_after_kill` 如实记录。所有等待以 `hard_end=deadline+kill_wait_s` 封顶——**共同七天 deadline 与宽限不被扩大**；drain 过的卡组使 status=failed（异常诚实上报，不自动 retry）。已核实生产 scorer（`text2cad_frozen_runner.py`）除一处只读 `subprocess.run` git 调用外不派生持久子进程（grep 无 num_workers/Popen/multiprocessing），正常全量运行 drain 首探即 `empty`，无误伤路径。
4. `_SIGTERM/_SIGKILL` 经 `getattr` 解析（Windows 无 `signal.SIGKILL`），降级平台不会在 import/运行时触 AttributeError。

## 4. Minor 修复映射（事实）

1. **npu-smi 进程行解析**：旧正则 `\|\s*\d+\s+\d+\s+\d+` 期待空格分隔三数字，真实进程表是竖线分列（`| 2     0  | 3294246 | python | 1849 |`），旧代码检不出进程行、只靠缺 marker 间接拒绝——**mixed 输出（进程行+marker 同在）会被误接受**。现按竖线分列：fields[0] 为 `^\d+\s+\d+$`（NPU+Chip）且 fields[1] 为纯数字 pid 才记进程行；芯片详情行（`| 0  4 | 0000:95:00.0 |`）因 bus-id 非纯数字不误判。新测试 `test_busy_row_with_marker_present_still_refused`（真实格式 mixed 拒绝）与 `test_chip_detail_row_not_misread_as_process`；既有 5 项 preflight 测试与真实 G3 证据输出仍通过。仅修本机解析，未泛化任何资源平台。
2. **控制器自身环境前缀**：wrapper 文档串与本报告 §7 的生产 nohup 示例均在 controller 命令前加 `PYTHONDONTWRITEBYTECODE=1` 与 checkout 外 `HF_HOME/XDG_CACHE_HOME/TMPDIR`（此前只在 child env 有）。旧报告原文未改。

## 5. CPU 验证证据（事实；命令、exit、平台边界）

全部在仓库根目录、`D:/miniconda/python.exe`（Python 3.13.5，Windows win32）执行：

```
git checkout ca70975（基线，修改前）
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_g4_launch.py -q
=> exit 0；52 passed, 6 subtests passed in 24.51s          # 修改前基线
# 修改并提交 226c410 后（工作树干净）：
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_g4_launch.py -q
=> exit 0；62 passed, 6 subtests passed in 32.87s           # +10 新测试
D:/miniconda/python.exe experiments/tests/test_text2cad_g4_launch.py
=> exit 0；Ran 62 tests OK in 31.8-33.5s                    # 不依赖 pytest 的 stdlib 路径
D:/miniconda/python.exe g4_r1_cpu_witness_20261002.py       # checkout 外 witness 驱动
=> exit 0；repo_head=226c4108f57e332ebfe967df24980ccc35c0595b，worktree_clean=true
```

证据文件：`experiments/runs/m1c_g4_r1_cpu_witness_20261002/`（`G4_R1_CPU_WITNESS_20261002.json`、`g4_r1_cpu_witness_20261002.py`、`pytest_run.log`、`unittest_run.log`、`witness_console.log`、`SHA256SUMS`；驱动原件同时在 checkout 外 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_launch_r1_witness_20261002/`）。witness JSON 逐项对比 Root/fresh 修复前证据：

- Root 注入 `card0.pid.txt`：修复前 RunnerError 撞库、card0 假 NONE、peer 存活、无 controller_exit；**修复后 exception=null、两卡真实 exit "0"/"0"、controller_exit_exists=true、still_running=[]、status=failed、record_error 落盘**。
- Root 注入 `controller_launch.json`：修复前两真实子进程遗留无监管；**修复后同上全部收尾**。
- fresh 两树模式（同参数 budget 0.9/term 0.2/kill 0.2/后代 1.5 s）：本机 Windows 无 killpg，`group_cleanup="unsupported_platform"`、`process_group_supported=false` 如实记录，后代照旧存活——**这是声明的 Windows 降级，不声称 POSIX 证据**；witness 自行清理其创建的后代。

**平台边界表（报告义务）**：

| 断言 | Windows 本机（已执行） | 说明 |
| --- | --- | --- |
| post-spawn 写失败收尾（pid/launch/event/exit/末次 git 五类） | **真实**（真实子进程+真实临时 git 仓库+注入写失败） | 平台无关逻辑，本机即真实证据 |
| 正常两卡成功/一卡失败 peer 成功/真实退出/共同 deadline 门 | **真实**（既有 52+6 全部保持通过） | 未因修复放宽 |
| pgid/信号序列与 drain 逻辑 | **mock**（注入 POSIX 语义 fake，已明确标注非 Linux 证据） | `_fake_group_signal`、`test_drain_terminates_surviving_group_mock` |
| 真实进程组 TERM/KILL 与后代收尾 | **未执行**（win32 无 killpg；本机只验证降级如实记录） | 待远端零 NPU CPU 见证，见 §6 待审命令 |

AST 稳定性（套件内 `AstStabilityTest` 通过）：七个已审定义 `assemble_batch/forward_and_score_batch/score_pair_token_diagnostic/load_gated_model_and_store/SharedBudgetLedger/authorize_text2cad_run/ZipCandidateStore` 与 `487d02b9ac92f5e2338718e45b1f023c308dd332` 的 `ast.dump` 完全一致——本轮未触碰 scorer 文件。

## 6. 待审远端 POSIX 见证命令（**本轮未执行**；须 Codex 审核完整 SHA 后另行授权，零 NPU/零模型/CPU-only）

Git 内见证即已提交的 `experiments/tests/test_text2cad_g4_launch.py`：同一文件在 Linux 上 `_PROCESS_GROUP_SUPPORTED=True`，`ProcessGroupTest` 两个真实父+后代测试自动走 POSIX 分支（真实 killpg、组空断言、marker 不落盘），其余 60 项与平台无关。禁止 SCP 发源码；只按第 3 节协议检出已审 SHA：

```bash
# yunnan-share2；$REPO=/home/ma-user/work/CADRetrieval，$RT=/home/ma-user/work/CADRetrieval_runtime
# 操作者前置（人工）：cd $REPO && git status --porcelain（空）&& git fetch origin &&
#   git checkout --detach <Codex批准的完整SHA> && git rev-parse HEAD（逐字符核对）
cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR="$RT/tmp" HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" \
  /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/tests/test_text2cad_g4_launch.py -v \
  > "$RT/runs/g4_r1_posix_witness_<SHA>.log" 2>&1
# 预期 exit 0、"OK"；ProcessGroupTest 两测试输出真实 POSIX 组收尾断言；
# 不产生/不触碰任何 NPU、模型、账本或 G4 run 产物；结束后 git status --porcelain 仍为空
```

## 7. 修复后的未来生产启动命令（**当前未授权**；评分额度仍 0）

操作者前置同旧报告（干净检出已审完整 SHA、npu-smi 两卡 OK 无进程、≥100GB、fresh run 根/账本）。控制器自身命令（本轮 Minor 修正：前缀落在 controller 上，不只 child env）：

```bash
# yunnan-share2
PYTHONDONTWRITEBYTECODE=1 \
HF_HOME=/home/ma-user/work/CADRetrieval_runtime/hf \
XDG_CACHE_HOME=/home/ma-user/work/CADRetrieval_runtime/xdg \
TMPDIR=/home/ma-user/work/CADRetrieval_runtime/tmp \
nohup /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  /home/ma-user/work/CADRetrieval/experiments/scripts/run_text2cad_g4_launch.py \
  --approved-sha <Codex批准的完整40hex> \
  > /home/ma-user/work/CADRetrieval_runtime/runs/g4_controller.nohup.log 2>&1 &
```

两条固定 scorer argv 与旧报告 §4 逐字一致（本轮未改 `build_card_argv`；`FixedArgvTest` 原样通过）。

## 8. 解释（与事实分开）

- **保活监管而非就地格杀**：pid.txt 等证据写失败时选择继续监管活进程（deadline 不变、真实退出照 reap、PID 仍在 events/launch_record/exit_record 三处落盘）而不是立刻终止作业——38 小时级 scorer 不应因证据通道抖动被毁；审查允许"可在异常时结束作业"，取破坏性更小且仍满足"不留无监督孩子"的方案。代价是该 run 最终 failed，需人工判读后处置，符合"不自动 retry"。
- **drain 放在全体 reap 之后**：drain 的 settle/TERM/KILL 等待若插在 wait loop 内会拖延同伴的 deadline 监管；放到全体 reap 后，同伴监管零延迟，drain 等待统一以既有 `deadline+kill_wait_s` 硬界封顶，七天预算与宽限不扩。
- **settle 窗口区分两类后代**：正常 scorer 不派生持久子进程，首探即 `empty`；短命助手在窗口内自然结束记 `drained_within_grace` 不判异常；只有超过窗口仍存活才 TERM/KILL 并判 failed——避免误伤又保证控制器返回时其创建的组无一存活（fresh 见证的"返回后仍写标记"在 POSIX 下不复存在）。
- **ok 条件收紧是单向的**：新增的 record_error/event_errors/launch_record_error/leftover_descendants 只可能把原本会 failed 或原本错误报 ok 的情形判 failed，不会把真实失败判成 ok；真实退出码与共同 deadline 判定未改。
- **Windows 降级是有意边界**：本机只用于开发验证；生产 Linux 上同一文件即真实组语义。测试把 POSIX 断言与降级断言写成显式分支，杜绝把 Windows/mock 误当 Linux 证据。

## 9. 限制与已知缺口（事实）

1. **真实 POSIX 进程组证据仍缺**：本机 win32 无 killpg，组 TERM/KILL 与后代 drain 在本机只有 mock 与降级记录；须 §6 远端见证（Codex 另授权）补齐后方可视为 Linux 生命周期已验证。
2. pgid 复用竞态：父退出到 drain 探测之间若旧 pgid 被新进程组复用，理论上可能探到他人组（signal-0 只读，TERM 仅当组非空且紧随 reap，秒级窗口、Linux pid 回绕下概率极低）；已在代码注释声明。
3. drain 最坏追加时长：两卡各 settle10+grace5+wait5=20 s（生产默认），仅在真有遗留后代时发生，且封顶于 `deadline+kill_wait_s`；正常全量运行首探 `empty` 为毫秒级。
4. `survivors_after_kill`（如 D 态不可杀）如实记 failed，控制器不悬挂；该情形需人工在服务器处置。
5. npu-smi 解析仍按本机 25.2.1 真实格式；格式漂移 fail-closed 拒绝，需人工复核。
6. 本轮未触碰远端 checkout、未创建任何远端 run/账本、未安装包、未跑任何模型/BERT/NPU 前向；witness/测试只创建并清理自己的 CPU 子进程与临时 git 仓库。

## 10. 下一轮建议（非授权）

Codex 复审重点建议：`launch()` 的 spawn try/except/else 分界、`_signal_card`/`_drain_card_group` 的 pgid 使用与 hard_end 封顶、ok 条件扩展、events 容错闭包、npu-smi 新列解析；随后另行授权 §6 远端零 NPU POSIX 见证（干净 detached 已审 SHA 上跑同一 unittest 文件），见证通过后再考虑是否进入 pilot_authorized 流程。任何源代码/配置改动使本轮审核失效，须重新提交。

## 11. 状态与回填

**review_required**。实现提交 `226c4108f57e332ebfe967df24980ccc35c0595b`（git status 干净、未 push）；本报告提交 SHA：`6fd8407e1db1170eaab4a094588bcad7b9a908cc`（本行为 SHA 回填，单独提交）。
