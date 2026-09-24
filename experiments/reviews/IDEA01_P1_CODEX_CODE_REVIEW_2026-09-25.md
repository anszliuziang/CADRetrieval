# Idea 01 P1 第一门代码审核（2026-09-25）

对象：Kimi 的实现提交 `10c734451ee22e608eaf0d119521c092fc7b8f62`，交接提交 `eb50aec1bedac3b8cf56ef8b51daa84931eaead7`。本审查只覆盖实现、CPU 选择清单与拟定的第二门运行路径；**尚无 P1 模型分数，NPU pilot 未获授权**。

## 裁决

**第一门 CPU 选择结果可作为待用清单；第二门调度代码暂不通过。** `ACTIVE_TASK.md` 保持 `review_required`、`approved_sha` 为空、NPU/评分预算为 0。Kimi 须在本地修复并经 Git 提交/推送；Codex 复审新完整 SHA 后才决定是否进入 `pilot_authorized`。

## 已独立核过的证据

1. 本地在 Kimi 的 P1 worktree 执行 `python -m pytest experiments/tests/ -q`：**89 passed**。服务器 `ssh yunnan-share2` 的 checkout 为 `10c734451ee22e608eaf0d119521c092fc7b8f62`，`git status --porcelain` 为空；未见 P1 NPU 结果。
2. 服务器 CPU 清单 `p1_selection.json`、`p1_eligible_pool.json`、`p1_pair_tokenization.jsonl` 的 SHA256 分别为 `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964`、`023f52820666b71daa4bc7677df22e35ef4b7672ebff10b4db0d14e5c6e6d5c1`、`650dfa816aad701791b2b6849c99d25524f26a0119883e8f3453adc39bb71a36`；本地取回的选择清单 SHA 与服务器相同。
3. 我独立遍历选择清单：32 个 query，各 8 个不同 UID，来源恰为 1 self / 3 共享循环偏移 / 2 方法近邻 / 2 长度近邻，414 个唯一 pair ID（256 条件 + 158 空 query），偏移关系与 `[27,8,23]` 一致。CPU gate 报 414/414 通过、最长 697 token。使用 pinned validation 原文逐对比较，选中正例与其 7 个候选之间**没有逐字相同程序**。这仍不证明几何家族或语义负例成立。
4. 910C 的 `/data/data_dev` 仍有 P0 补证复制进程 PID 2602998 处于不可中断 `D` 状态。未再次在该挂载发起哈希读取；第二门所要求的 OBS 核验尚未重新完成。P0 完整副本在共享 OBS 与本地已通过 78 项 SHA 校验。

## 必须修复后再审

### R1（阻塞）：分阶段运行会重复评分并突破全程预算

`run_p1_matrix.py` 用追加模式打开 `pairs.jsonl`，但每次调用都把 `n_forwards=0` 重新开始，完全不检查已有 `pair_id`、selection SHA、环境或累计时长。交接文档建议在同一输出目录依次执行 `--pair-ids 0`、`0-9`、`all`；这会产生重复记录，实际尝试 **1+10+414=425** 次前向，超过 ≤416 的全程上限，且后续分析因重复 pair ID 失败。每次调用的 45 分钟计时也会重置。

修复要求：明确 1 → 新增 9 → 剩余 404 的阶段方案（或安全的断点续跑）；同一 run 逐次调用必须验证已有记录的 pair ID 唯一、状态成功、相同 selection SHA / Git SHA / 模型数据 / 精度与环境，并把已完成的前向及时间计入全程预算。任何重复、混版或失败记录都应在载入权重前拒绝；不能以覆盖、静默跳过或另起目录掩盖累计成本。运行后一律能得到恰好 414 条唯一原始记录。更新交接命令与相应测试。

### R2（阻塞）：逐 token 验证器会把错误分数判为通过

`verify_p1_run.py` 计算 `worst = max(abs(token_sum - pair_sum))`，却**没有把差异加入失败计数**，也没核对 `token_id == input_ids[position]` 或所有目标位置连续。独立构造的单 pair：逐 token 和 `-2`、记录 sum `-1`，运行该脚本仍返回 **exit 0**，同时打印 `max |diff|=1.000e+00; failures: 0`。这会使第二门审计对原始分数失效。

修复要求：以明确数值容差逐对核对 sum/mean/EOS、完整目标位置/shift/token ID 与 `input_ids`，并把 `all_finite` 为假、缺 token 文件、空 token 列表、重复/缺失 pair 等作为非零退出；加入能复现上述篡改的反向测试。分析前须先通过验证器。

### R3（阻塞）：正式运行的授权与资产/清单绑定不足

调度器目前只检查 Git 工作树干净、清单自报 `n_gate_fail=0` 和 pair 数 ≤416，未验证当前 `ACTIVE_TASK` 已是 `pilot_authorized`、运行 SHA 等于 `approved_sha`、清单 SHA 等于 Codex 审核的固定清单、模型与数据副本的 SHA256。`--model-dir`/`--data-dir` 还可覆盖配置；任意自造的 gate summary 为 0 的清单可被执行。需在载入权重前有可审计的强制前置校验，或由 Codex 审核的启动脚本严格执行并记录所有检查；在 `ACTIVE_TASK` 中钉住获批清单 SHA。任何不符都停止，不能只把 SHA 写进输出作为事后信息。现在 `/data/data_dev` 挂载仍有 D 状态复制，先解决或由 Codex 明确修订资产核验路径，不能把不可核验的 910C OBS 写成已通过。

**实现注意：**`ACTIVE_TASK.md` 含 `approved_sha`，不能让该字段等于包含该字段的同一 Git 提交 SHA；这是自引用。运行已审核代码的 detached HEAD 时，应在 `git fetch` 后从 Codex 控制的 `origin/master:experiments/ACTIVE_TASK.md` 读取最新授权（或使用等价的仓库外授权证明），验证 `pilot_authorized`、`approved_sha == HEAD` 与获批选择清单 SHA。不得从 detached HEAD 中的旧 `ACTIVE_TASK.md` 误判为已授权；当前 master 仍是 `review_required`。

### R4（应修）：运行失败证据与输出路径

模型载入、数据读取、环境写入等发生在逐 pair 的 `try` 之外；异常可能不留下失败记录。环境文件名只有秒级时间戳，快速两次调用可能覆盖；输出目录也未检查在 Git checkout 外。请保证异常非零且留存原因、每次调用的环境文件不覆盖、所有输出在 checkout 外。预算参数不能被命令行放宽。已授权的三个全程上限（416 前向、1 卡小时、45 分钟）应在多次调用中统一约束，达到任一条件即非零停止。

## 复审门槛

Kimi 在 `experiments/submissions/idea01_p1_conditional_sensitivity_20260925/IMPLEMENTATION.md` 追加修复记录：新完整代码 SHA、改动说明、反向测试命令/结果、全程预算与断点规则、远端拟运行命令。不要覆盖原始 CPU 清单；若必须重生，另存新路径并给出逐字段差异与 SHA。Codex 将复核代码、测试和清单后，再决定是否授权单卡的 1 对验证。任何 P1 NPU 前向仍须等待 `ACTIVE_TASK.md` 的明确授权。
