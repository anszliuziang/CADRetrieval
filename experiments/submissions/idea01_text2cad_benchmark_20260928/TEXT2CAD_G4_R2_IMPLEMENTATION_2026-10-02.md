# Text2CAD G4 R2：完成状态与共同deadline收口实现报告（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`；子合同 `experiments/plans/IDEA01_M1C_G4_R2_COMPLETION_FIX_2026-10-02.md`；针对复审 `experiments/reviews/IDEA01_M1C_G4_CODEX_CODE_REVIEW_R2_2026-10-02.md` 的两个已独立复现 Important、见证有效性问题和 group_alive Minor
- 基线：`eea509ec431a4302d809eb50a82f83b509645e43`（R1 代码 226c410/报告回填 0f66086 之上）；**本轮实现提交（代码+测试）：`e8ef0cdd0349f255d8a299ae1b32fbf3fcdd1263`**；本报告提交 SHA 见文末回填
- 本轮零模型/BERT/NPU 前向、零网络、零下载/安装、零远端 checkout 改动（未 fetch/detach/SCP 源码；仅按 R1 既有协议把 CPU 证据镜像到 cvteam14 OBS 并逐 SHA 核验，证据归档不构成远端源码部署）；不改任何旧 scorer/loader/math/ledger/worker/config/env、Codex 文档或旧报告；评分方法、资产、full 固定 argv、29319624 对、每片 1822 chunk/14659812 对/916466 批、b16 尾 14、FP32、full272、无 trim-cache、无 probe-token-audit-resume、仅两次 fresh scorer、336 合计卡时、共同 604800 秒均不变
- 状态：**review_required**。完整 SHA 经 Codex 复审并另写 ACTIVE 前，真实评分额度始终 0；Kimi 不自行启动 G4、不占卡、不 push

## 1. 改动文件（事实；仅此两个，git 证据）

| 文件 | 变更 | 用途 |
| --- | --- | --- |
| `experiments/src/idea01/text2cad_g4_launch.py` | +约144/−74 行（净） | 两 Important + probe Minor 修复（控制器模块） |
| `experiments/tests/test_text2cad_g4_launch.py` | +约549 行（净） | Root 复现回归、DescendantDeadlineTest 新类、ProcessGroupTest 见证顺序修复、PostSpawnFaultTest 末事件测试 |

`git diff eea509e..e8ef0cd --stat`：2 files changed, 693 insertions(+), 74 deletions(-)；无其他文件进入提交。`run_text2cad_g4_launch.py` 无需改动（wrapper 只转调 main，本轮修复全在模块与测试）。原始 CPU 证据镜像（git 忽略）：本地 `experiments/runs/m1c_g4_r2_cpu_witness_20261002/`（witness 驱动+JSON+console/unittest/pytest 三日志+SHA256SUMS 六文件），同六文件已在 cvteam14 OBS `/data/obs_data_liuza/CADRetrieval/runs/m1c_g4_r2_cpu_witness_20261002/` 落地并 `sha256sum -c` 全部通过（exit 0）；驱动原件在 checkout 外 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_r2_cpu_witness_20261002/`。

## 2. 反例确认（事实；修复前复现）

修复前在基线 `eea509e` 上用与 Root 相同的两个场景复核（驱动 `repro_prefix_r2.py`，checkout 外，stdout-only 不覆盖 Root 原件，exit 0），与 Root `CODEX_G4_R1_REMAINING_GATES_20261002.json` 逐项一致：

1. `final_event_write_failed`（两个真实 CPU 孩子 exit 0，仅注入 controller_exit 事件 OSError）：修复前 returned_status="ok"、returned_event_errors 非空、persisted_status="ok"、persisted_event_errors=**空**——返回值与持久 JSON 都不诚实。
2. `late_parent_descendant_deadline`（mock direct drain：deadline 1600、父 1599 reap、调用方 hard_end=deadline+30=1630）：修复前真实函数到 **1609.0** 才请求 TERM——晚于原共同 deadline 9 秒，+30 被当成了后代继续运行宽限。

## 3. Important 1 修复映射（事实）：后代组信号全部收回原共同 deadline 内

`_drain_card_group` 签名由单一 `hard_end` 改为 `deadline`（原共同 deadline，信号调度界）与 `confirm_end`（deadline+kill_wait_s，仅记账界）：

1. **settle 窗口**：`settle_end = min(now + settle_s, deadline − (term_grace_s + kill_wait_s))`——保证 settle 结束后完整 TERM 宽限与 KILL 仍能落在 deadline 内；临近 deadline 时窗口**缩短直至跳过**，绝不后移。
2. **TERM/KILL 截止不增加**：settle 后立即 TERM；TERM 宽限 `min(now + term_grace_s, deadline − kill_wait_s)` 封顶，需要时 KILL 也在 deadline 内发出。
3. **deadline+kill_wait_s 只作记账**：仅封顶 **KILL 已发出之后** 的确认等待（确认死亡/记账），不再授权任何后代在 deadline 后运行或等到 deadline 后才发信号。
4. **极端已超时立即处理且诚实 failed**：drain 自身在截止后才启动（如父在病态记账界才被 reap）时，不再按任何未来时刻安排——TERM/KILL **立即背靠背**发出，活作业绝不为了不记晚信号而被遗漏；凡在 ≥ 原 deadline 时刻发出的信号置 `group_deadline_breach=True` 并写入 exit record，该 run 永远 failed、不能宣称预算守住。
5. **ok 条件扩展**：新增 `not group_deadline_breach` 与 `group_probe_error is None`（见 §5）；exit record 逐卡新增 `group_deadline_breach`、`group_probe_error` 字段，事件 `descendant_term_sent`/`descendant_kill_sent` 增记 `after_deadline`（schema_version 仍 text2cad-g4-launch-v1，纯增字段）。
6. 覆盖边界（均见 §6 测试）：父近 deadline 退出（deadline−1）、父先退出 peer 仍活（drain 不拖延 peer reap）、最后一片退出后残留、drain 启动已超时、KILL 后仍存活（survivors_after_kill 如实记）。

Root 场景修复后复核（驱动 `repro_postfix_r2.py`，exit 0）：同一 mock 下 TERM 于 **1599.0** 发出（≤1600），无任何信号越过 deadline，breach=False，leftover_descendants=True（诚实异常→failed）。

## 4. Important 2 修复映射（事实）：末次事件 I/O 进入最终状态与持久 JSON

`launch()` 末尾顺序改为：先 `events({"event": "controller_exit", ...})` 追加 → 再把 `event_errors` 刷新进 exit_record 并在非空时强制 `status="failed"` → 最后写 `controller_exit.json` 并返回同一 record。由此：

- 仅末事件写失败时，**返回状态与持久 JSON 同为 failed、两处 event_log_errors 同含该错误**（Root 场景修复后复核：returned/persisted 均 "failed"，两卡真实 exit 0/0 保留，无活孩子遗留）；
- fail-if-exists（`_atomic_write_text` 不变）、真实退出码、不无限 retry（未新增任何重试）均保持；
- 磁盘完全不可写仍走既有底限：所有孩子已 reap 后 `controller_exit.json` 写不出 → RunnerError → main 退出 2，非零且不伪称 ok（既有测试 `test_exit_record_write_failure_raises_after_children_reaped` 保持通过）。

## 5. Minor 修复映射（事实）：非 ESRCH probe 错误不再误报 empty

`_drain_card_group` 的 `group_alive` 现在只把 `ProcessLookupError`（ESRCH）判为空；其他 OSError（如 EPERM）传播到 drain 各阶段统一捕获：记 `card.group_probe_error` 与 `group_cleanup="probe_error:<type>: <msg>"`、不发任何信号、run failed（ok 条件含 `group_probe_error is None`）。局部修复，未引入通用资源平台；`_signal_card` 的 deadline 路径与旧七评分核心均未触碰。

## 6. 见证顺序修复（事实）：先采集断言、后清理，区分 live/zombie

- 新增 `_pid_state`（POSIX：kill(pid,0) + `/proc/<pid>/stat` 区分 live/zombie/gone；Windows：只读 OpenProcess 探测；未知如实记 unknown）与 `_group_members_posix`（/proc 枚举组成员及状态；无 /proc 返回 None 并退化为 killpg 探测，限制如实记录）。
- `ProcessGroupTest` 两个真实树测试重构：controller 返回后**立即** `_observe_tree_state` 采集真实组/descPID 状态并**在 try 内先行断言**，`finally` 只清理测试自建 PID——测试清理永远发生在采集与断言之后，不能代替被测 controller 收尾。POSIX 断言无一 live 成员（zombie 只记录不算存活泄漏）；Windows 断言**后代在 controller 返回时确实仍 live**——证明 PID-only 降级是真实发生而非空泛声明，随后 finally 由测试自行 reap，且证据只称 Windows 降级、不称 Linux 实测。POSIX mock（`_fake_group_signal`、各 fake-clock 测试）继续明确标注非真实 Linux 证据。

## 7. CPU 验证证据（事实；命令、exit、平台边界）

全部在仓库根目录、`D:/miniconda/python.exe -B`（Python 3.13.5，Windows win32）执行，针对已提交 `e8ef0cd`（worktree 干净）：

```
D:/miniconda/python.exe -B <checkout外>/repro_prefix_r2.py    # 修复前基线 eea509e
=> exit 0；两个 Root 场景复现与 Root JSON 一致（见 §2）
D:/miniconda/python.exe -B <checkout外>/repro_postfix_r2.py   # 修复后
=> exit 0；TERM@1599.0≤deadline；returned/persisted 均 failed 且错误可见
D:/miniconda/python.exe -B experiments/tests/test_text2cad_g4_launch.py
=> exit 0；Ran 72 tests OK in 45.361s        # 62+10 新，stdlib 路径不依赖 pytest
D:/miniconda/python.exe -B -m pytest experiments/tests/test_text2cad_g4_launch.py -q
=> exit 0；72 passed, 6 subtests passed in 44.65s
D:/miniconda/python.exe -B experiments/tests/test_text2cad_g4_launch.py -v DescendantDeadlineTest
=> exit 0；Ran 9 tests OK                    # 目标类可单独选择（待审远端命令同款）
D:/miniconda/python.exe -B <checkout外>/g4_r2_cpu_witness_20261002.py
=> exit 0；repo_head=e8ef0cdd…、worktree_clean=true（witness JSON 见镜像）
```

witness JSON 要点：末事件注入 returned/persisted 双 failed、真实 exit [0,0]、无残留进程；Root late-parent TERM@1599.0、signals_after_deadline=[]；near-deadline TERM@1594/KILL@1595；after-deadline 立即 TERM+KILL@1631 且 breach=true；probe_error 如实记；两个真实树在 controller 返回时（清理前）观察：Windows 上后代 state="live"、group_cleanup="unsupported_platform"——声明的降级，非 POSIX 证据。

**平台边界表（报告义务）**：

| 断言 | Windows 本机（已执行） | 说明 |
| --- | --- | --- |
| 末事件写失败双端诚实 failed（真实 CPU 孩子+真实临时 git 仓库） | **真实** | 平台无关逻辑，本机即真实证据 |
| drain 调度数学：settle 缩短/跳过、TERM/KILL 截止、超时立即+breach、probe 错误 | **mock**（fake clock + 注入 POSIX 语义，已标注非 Linux 证据） | DescendantDeadlineTest 9 项 + witness |
| 父近 deadline/父先走 peer 仍活/最后一片残留的 launch 级编排 | **mock**（同上；退出/reap/事件顺序为真实代码路径） | 3 项 end-to-end |
| 真实进程组信号与后代收尾、live/zombie 区分 | **未执行**（win32 无 killpg；本机只验证降级如实记录且后代在返回时确为 live） | 待远端零 NPU CPU 见证，见 §8 |
| 正常双卡成功/一卡失败 peer/共同 deadline/post-spawn 五类故障/full argv/七核心 AST/npu-smi 解析 | **真实回归**（既有全部保持通过） | 未重跑 G1/G2/G3 或无关套件 |

AST 稳定性（套件内 `AstStabilityTest` 通过）：七个已审定义与 `487d02b9ac92f5e2338718e45b1f023c308dd332` 的 `ast.dump` 完全一致——本轮未触碰 scorer 文件；full 固定 argv 测试（`FixedArgvTest`）原样通过。

## 8. 待审远端 POSIX 见证命令（**本轮未执行**；须 Codex 审核完整 SHA 后另行授权，零 NPU/零模型/CPU-only）

Git 内见证即已提交的 `experiments/tests/test_text2cad_g4_launch.py`（stdlib unittest，不依赖 pytest/安装包）：Linux 上 `_PROCESS_GROUP_SUPPORTED=True`，`ProcessGroupTest` 两树测试自动走 POSIX 分支（真实 killpg、live/zombie 区分、**断言先于 finally 清理**），`DescendantDeadlineTest`/`PostSpawnFaultTest`/`CommonDeadlineTest` 为平台无关逻辑加编排，`LaunchEngineTest` 覆盖必要正常双卡/一卡失败路径。禁止 SCP 源码；只按第 3 节协议检出已审完整 SHA；缓存/字节码/临时目录均在 checkout 外：

```bash
# yunnan-share2；$REPO=/home/ma-user/work/CADRetrieval，$RT=/home/ma-user/work/CADRetrieval_runtime
# 操作者前置（人工）：cd $REPO && git status --porcelain（空）&& git fetch origin &&
#   git checkout --detach <Codex批准的完整SHA> && git rev-parse HEAD（逐字符核对）
cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR="$RT/tmp" HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" \
  /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python -B \
  experiments/tests/test_text2cad_g4_launch.py -v \
  ProcessGroupTest DescendantDeadlineTest PostSpawnFaultTest \
  CommonDeadlineTest LaunchEngineTest \
  > "$RT/runs/g4_r2_posix_witness_<SHA>.log" 2>&1
# 预期 exit 0、"OK"；日志含真实 PID/进程组、controller 返回前后状态与共同 deadline 断言；
# 不产生/不触碰任何 NPU、模型、账本或 G4 run 产物；结束后 git status --porcelain 仍为空
```

## 9. 解释（与事实分开）

- **界内调度而非界后授权**：R1 的 `hard_end=deadline+kill_wait_s` 把"记账硬界"误用为"信号调度界"。R2 把两个界显式分开：信号（settle 结束/TERM/KILL）一律以原 deadline 为界推导截止（term 截止=deadline−(grace+wait)、kill 截止=deadline−wait），+30 只剩 KILL 后确认等待一个用途。这样父在 deadline−1 退出时，settle 被跳过、TERM 在 deadline−1 发出——宽限被压缩的是后代（异常情形，run 本就 failed），而不是预算被延长。
- **已超时仍立即收尾**：drain 启动即逾时时选择立即 TERM+KILL 并记 breach，而不是为了"不留晚信号记录"而放弃活作业——安全收尾优先，诚实记账其次，预算否认（failed+breach）兜底；该情形不得宣称预算通过，证据里也无法通过。
- **末事件前置而非追加兜底**：把 controller_exit 事件移到持久化之前，使"最后一条必需事件 I/O"与普通事件失败走同一条 `event_errors`→failed 路径；代价是该事件自带 status 字段在极端写失败时不会落盘——但失败本身已完整进入返回与持久 JSON，无信息被隐藏。
- **probe 错误只局部收紧**：非 ESRCH 只影响 drain 自身的判空，不扩散到 deadline 主信号路径（`_signal_card` 既有 PID 回退保留），符合"别扩框架"。
- **见证顺序是被测属性的一部分**：树测试把"采集+断言"放进 try、`finally` 仅清理自建 PID，使"controller 返回时无活树"成为可独立失败的断言；Windows 分支反向断言后代在返回时仍 live，是为了让"降级"本身成为可证伪证据而非措辞。

## 10. 限制与已知缺口（事实）

1. **真实 POSIX 进程组证据仍缺**：本机 win32 无 killpg；组 TERM/KILL、后代 drain、live/zombie 区分在本机只有 mock 与降级记录；须 §8 远端见证（Codex 另授权）通过后方可视为 Linux 生命周期已验证。
2. drain 的界内压缩意味着父极近 deadline 退出时后代几乎无宽限（TERM/KILL 背靠背）——这是预算优先的有意取舍，且该 run 因 leftover_descendants 必然 failed，不影响正常路径（正常 scorer 不派生持久子进程，首探即 empty）。
3. pgid 复用竞态（R1 已声明）不变：父退出到 drain 探测之间旧 pgid 理论上可被复用；signal-0 只读、TERM 仅当组非空且紧随 reap，概率极低，代码注释保留该限制，不扩大成普遍保证。
4. `survivors_after_kill`（如 D 态）如实 failed 且不悬挂；需人工在服务器处置。
5. 本轮未触碰远端 checkout、未创建任何远端 run/账本、未安装包、未跑任何模型/BERT/NPU 前向；witness/测试只创建并清理自己的 CPU 子进程与临时 git 仓库。

## 11. 下一轮建议（非授权）

Codex 复审重点建议：`_drain_card_group` 的 deadline/confirm_end 两界推导与 breach 记录、末事件前置顺序、probe 错误捕获范围、树测试先断言后清理的编排；随后另行授权 §8 远端零 NPU POSIX 见证（干净 detached 已审完整 SHA 上跑同一 unittest 文件的目标类）。见证通过前不发布新完整 SHA、不进入 pilot_authorized；任何源代码/配置改动使本轮审核失效，须重新提交。

## 12. 状态与回填

**review_required**。实现提交 `e8ef0cdd0349f255d8a299ae1b32fbf3fcdd1263`（git status 干净、未 push）；本报告提交 SHA：`PENDING_BACKFILL`（本行为 SHA 回填，单独提交）。
