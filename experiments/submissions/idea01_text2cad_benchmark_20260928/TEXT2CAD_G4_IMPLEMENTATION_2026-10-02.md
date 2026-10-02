# Text2CAD G4：最小双卡全量启动控制器实现（2026-10-02）

- 任务：`idea01_text2cad_benchmark_20260928`；合同 `experiments/plans/IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md` §5；入口 `experiments/ACTIVE_TASK.md`（state=implementation，gpu_budget=0，max_scored_pairs=0）
- **实现提交（代码+测试）：`0bba6a7acb573aefbe1797a043b6dafe974e61e8`**；本报告提交 SHA 见文末回填
- 本轮零模型/BERT/NPU 前向、零网络、零安装包、零远端改动；不改任何既有 scorer/loader/math/ledger/worker/config/env/Codex 文档/旧报告
- 状态：**review_required**。完整 SHA 经 Codex 审码并另写 ACTIVE（pilot_authorized + 预算）前，真实评分额度始终 0；Kimi 不自行启动 G4

## 1. 新增文件（事实；仅此三个，git 证据）

| 文件 | 行数 | 用途 |
| --- | --- | --- |
| `experiments/src/idea01/text2cad_g4_launch.py` | ~1037 | 启动控制器模块（纯 CPU、import 不拉 torch；只读 git 白名单；无 qrels/UID/CSV） |
| `experiments/scripts/run_text2cad_g4_launch.py` | 33 | 薄入口（同 run_text2cad_score_matrix.py 模式） |
| `experiments/tests/test_text2cad_g4_launch.py` | ~1122 | 52 测试 + 6 子测试（真实临时 Git 仓库/真实子进程/假时钟/真实 worker） |

`git diff --cached --stat`（提交 0bba6a7）：3 files changed, 2192 insertions(+), 0 deletions；无既有文件改动。CPU 证据镜像（git 忽略）：`experiments/runs/m1c_g4_cpu_launch_20261002/`（README 索引 + 各命令原始输出与真实 exit 文件）。

## 2. 控制器职责与合同 §5 的映射（事实）

1. **校验**（全部只读、全部先于任何产物创建）：`--approved-sha` CLI 必填且必须 40-hex（脚本绝不猜测、绝不用 HEAD/任何 ref 作批准值）；HEAD==approved、detached（`symbolic-ref`）、`git status --porcelain` 空；origin URL 归一化比对；从**已 fetch 的** `origin/master:experiments/ACTIVE_TASK.md`（blob SHA256 与 ref SHA 落盘）经已审 `parse_active_task_authorization` 解析，并逐项钉死 G4 范围：`task_id=idea01_text2cad_benchmark_20260928`、`approved_sha==HEAD`、`max_scored_pairs==29319624`、`gpu_budget==336` 卡小时、`run_id=idea01_text2cad_g4_full_20261002`。git 命令白名单仅 `rev-parse/symbolic-ref/status/remote get-url/show`——控制器**绝不 fetch/detach/config/修改源码**（fetch+detach 是操作者启动前的人工步骤，scorer 自身门还会独立 fresh fetch 终审）。读 ACTIVE 用二进制 `git show`（SHA 按原始字节计算）。
2. **fresh 检查**：`$RT/runs/idea01_text2cad_g4_full_20261002`、两 card 目录、`$RT/budgets/text2cad_m1c_<approved_sha>.json` 及 `.lock` 均不得存在；G3 目录/账本属于不同 run_id/批准，永不复用（有专测）。run 根用 `os.mkdir`（无 parents）创建——该 mkdir 即单次启动锁：第二次控制器调用在零 spawn 下拒绝。只读 `npu-smi info` 预检要求 ≥2 芯片全 OK 且无进程行、无"无进程"标记即视为格式漂移拒绝（原始输出落盘；绝不终止他人进程）；runtime 分区 ≥100GiB。
3. **两条固定 full CLI + safe env**：与已审 G3 生产模板同形、去 probe 旗标（见 §4）；模块自检禁用旗标（`--probe-selection/--token-audit/--resume/--text-cache/--scan-candidates`）绝不在 argv 中；dtype/trim 靠 runner 默认（其 score 门已拒绝任何非 FP32/none）；无任何 extra scorer args 透传（CLI 仅三项；源码无 `parse_known_args`/`REMAINDER`）。env=会话继承−`ASCEND_RT_VISIBLE_DEVICES`（移除并记录）+固定 offline/线程覆盖；HF_HOME/XDG/TMP 强制在 checkout 外。逐卡落盘 `launch_command_cardX.txt`（shlex.join 原文）、`cardX.pid.txt`、`cardX.launch_time.txt`、`cardX.stdout/stderr.log`、`controller_launch.json`（git/授权来源/plan/env/PID/共同起止）与 `controller_events.jsonl`（逐事件 fsync）。
4. **后台持久与共同 deadline**：`start_new_session=True`（Linux setsid，SSH 断开不杀作业；控制器自身由操作者 nohup 启动，见 §4），stdin=/dev/null。`common_start` 在 card0 spawn 前取一次，`deadline=start+604800s` 两卡共用——card1 错开 3 秒**不**增加其墙钟；TERM@deadline−90s、KILL@deadline−30s（60+30s 宽限在七天总限内），硬回收界 deadline+30s 防控制器悬挂。每卡退出码在 reaped 当下立即写 `cardX.exit_code.txt`（裸整数；spawn 失败写 NONE），不等同伴、无管道掩盖。
5. **失败语义**：单卡失败即实记录并交接，同伴保留已授权运行；绝不自动重启/重试；`controller_exit.json` 的 `status=ok` 仅当两卡真实 exit0 且无 deadline 强制且 spawn 无误且启动后 git 复核（HEAD/干净）通过——spawn 成功绝不记为实验成功。控制器 exit：ok→0，否则 2（含全部拒绝）。

固定 plan 直接取自**真实生产数学** `text2cad_pair_index.PairSpace(3644,8046)`：shard0 [0,1822)、shard1 [1822,3644)；各 1822 chunk/14659812 对/916466 批（8046=502×16+14 尾）；总 3644 chunk/29319624 对/1832932 批。账本路径公式与已审 `text2cad_frozen_runner.canonical_budget_path` 有测试钉死一致。

## 3. CPU 测试证据（事实；真实 exit，全在仓库根目录执行）

```
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_g4_launch.py -v
=> exit 0；52 passed, 6 subtests passed in 27.04s
D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_frozen_runner.py -q   # 受影响绑定模块回归
=> exit 0；48 passed, 3 subtests passed in 25.73s
D:/miniconda/python.exe experiments/scripts/run_text2cad_g4_launch.py --help
=> exit 0（CLI 仅 --approved-sha 必填 + 两个生产路径项）
```

原始输出与 exit 文件：`experiments/runs/m1c_g4_cpu_launch_20261002/`（首次误在证据目录内以相对路径执行为 pytest exit 4 用法错误、零测试运行，已重跑并在 README 如实记录）。关键绑定证据（均非 mock 同名返回）：

- **真实 full plan/producer 绑定**：`ProducerBindingTest` 取控制器精确固定 argv、仅替换三个路径值（--repo-root→临时仓库、--pair-manifest-dir→本地真实 worker `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker`、--out-dir→临时目录），驱动**真实已审 scorer `main()`**：通过 committed 指纹×真实冻结 worker 四文件的 Gate 1、固定参数门（FP32/none/两卡 pin），exit 2 恰好停在授权门（无 unrecognized/two-card/fingerprint 类错误）；worker 真实计数 3644/8046/29319624==固定计划。
- **真实临时双版本 Git 仓库**（本地 bare origin、零网络）：通过/HEAD 不符/短 SHA/HEAD 字样/dirty/attached/非 pilot/错 task_id/错 run_id(=G3)/max_pairs=804600/card_hours=12/错 origin 逐项拒绝；白名单拒绝 fetch/checkout/config/reset/clean/push/pull；**真实项目仓库当前状态（attached master + 非 pilot ACTIVE）实时拒绝**。
- **真实子进程**：两卡 exit0（真实 PID/stdout/stderr 捕获/事件序列）；card0 exit3+card1 exit0→failed 且两卡裸码各自及时落盘不误报；`sleep 60` 假 scorer 在 0.8s 共同墙钟下被真实 TERM/KILL（deadline_enforced、非零码）；card1 spawn 失败→card0 保留完成、card1 写 NONE、恰两次尝试零重试；二次启动拒绝且零 spawn；preflight/低盘拒绝后 run 根不存在（可人工修复后再启动）。
- **确定性假时钟**：common_start=1000/deadline=1100，card1 在 1003 启动仍共用 1100；card0@1050 exit0 无强制；card1 TERM@1090→KILL@1096→exit −9（宽限全在预算内）。
- **AST 稳定性**：七个已审定义 `assemble_batch/forward_and_score_batch/score_pair_token_diagnostic/load_gated_model_and_store/SharedBudgetLedger/authorize_text2cad_run/ZipCandidateStore` 与 `487d02b9ac92f5e2338718e45b1f023c308dd332` 的 `ast.dump` 完全一致（该文件自 G3 后无提交，`git log 487d02b9..HEAD` 为空）。
- **卫生**：控制器+脚本 AST 活记号无 `qrels`/`evaluation_map`/`.csv`；子进程验证 import 不拉 torch；常量交叉核对（TASK_ID/三 revision/expected-keys rel/3644/8046/29319624/batch16/SEVEN_DAY_SECONDS=604800/SHARD_BY_DEVICE/账本公式）与已审模块一致；npu-smi 解析器接受**真实 G3 pre-launch 证据**输出。

## 4. 待审的未来生产命令（**当前未授权**；须 Codex 审码 + ACTIVE 填 pilot_authorized/完整 SHA/max_scored_pairs=29319624/gpu_budget=336/run_id 后）

操作者前置（人工、只读或按 §3 协议；控制器不代做）：远端 `git status --porcelain` 为空→`git fetch origin`→`git checkout --detach <批准完整SHA>`→`git rev-parse HEAD` 逐字符核对；`npu-smi info` 两卡 OK 无进程；`$RT` 分区 ≥100GB（cvteam14 OBS 侧容量另由操作者确认）；`$RT/runs/idea01_text2cad_g4_full_20261002` 与 `$RT/budgets/text2cad_m1c_<SHA>.json` 不存在。

```bash
# yunnan-share2；$REPO=/home/ma-user/work/CADRetrieval，$RT=/home/ma-user/work/CADRetrieval_runtime
nohup /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  "$REPO/experiments/scripts/run_text2cad_g4_launch.py" \
  --approved-sha <Codex批准的完整40hex> \
  > "$RT/runs/g4_controller.nohup.log" 2>&1 &
```

控制器随后生成的两条固定 scorer 命令（逐字、仅 device/shard/out-dir 三处不同；card0 示）：

```bash
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HOME="$RT/hf" XDG_CACHE_HOME="$RT/xdg" TMPDIR="$RT/tmp" \
  /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  "$REPO/experiments/scripts/run_text2cad_score_matrix.py" --mode score \
  --checkpoint "$RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth" \
  --bert-dir "$RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b" \
  --cad-seq-zip "$RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip" \
  --pair-manifest-dir "$RT/pair_manifest_v4" \
  --expected-keys-manifest "$REPO/experiments/configs/text2cad_checkpoint_expected_keys_20260930.json" \
  --batch-size 16 --texts-per-chunk 1 --cards-used 1 \
  --device npu:0 --shard-id 0 --n-shards 2 \
  --out-dir "$RT/runs/idea01_text2cad_g4_full_20261002/card0" --repo-root "$REPO"
# card1：--device npu:1 --shard-id 1 --out-dir .../card1；无 probe-selection/token-audit/resume/cache/trim
```

预计资源（沿用既有实测，不变评分路径）：两卡各 1822 chunk/14659812 对/916466 批；G3 实测 ETA 28.85179 小时、G2 保守 38.28394 小时（估计，非上界）；正式硬上限共同 604800 秒墙钟/336 合计卡小时（1209600 卡秒）；G3 实测峰值 HBM ~1.9GB/卡；原始 JSONL 预计 20–25GB。仅两次 fresh scorer 调用，无预热/重试/空查询。

## 5. 解释（与事实分开）

- 控制器不 fetch/detach：授权的新鲜终审属于 scorer 自身门（已审、fresh fetch）；控制器只做 launch 前只读核对，职责单一、可重入（拒绝零残留），避免两个写 Git 的组件。
- `os.mkdir(run_root)` 作为单次启动锁：原子、跨进程；控制器中途死亡留下目录即拒绝后续启动，强迫人工审查——这正是"失败保留记录、不自动 retry"的语义。
- 共同 deadline 从 card0 spawn 前起算、TERM/KILL 提前 90/30 秒：保证两卡进程在七天界内死亡，宽限不超限；`deadline+30s` 硬回收界防控制器自身悬挂（KILL 后 poll 必然返回，否则如实记 unreaped 并 failed）。
- exit_code.txt 在 reap 当下逐卡写（不等同伴）：满足"一片失败及时记录"；最终 ok 还需启动后 git 复核，防止运行期间 checkout 被改动后仍报成功。
- npu-smi 解析保守化：只接受"≥2 芯片全 OK + 有进程表无进程标记 + 零进程行"，任何格式漂移即拒绝（fail-closed），原始输出始终落盘供人工判读。

## 6. 限制与已知缺口（事实）

1. 控制器本轮**未在真实服务器运行**（无授权）；其 git/npu-smi/生产路径只在本地真实临时仓库、真实子进程与真实 G3 证据上验证。nohup 由操作者命令提供（§4），控制器不自我 daemonize。
2. npu-smi 解析器按该机 25.2.1 真实输出格式编写；驱动升级导致格式变化会拒绝启动（fail-closed，需人工复核，不自动放宽）。
3. 控制器的 ACTIVE 读自已 fetch 的 origin/master（不 fetch）；若批准 push 后操作者忘记 fetch，控制器拒绝——这是预期的 fail-closed，操作者 fetch 后重跑即可（拒绝零残留）。
4. 生产 argv 在 Linux 上为 POSIX 路径；本地 Windows 测试以 Path 归一化断言同一结构，非逐字节 POSIX 字符串。
5. OBS（cvteam14）侧 ≥100GB 与挂载健康不由控制器检查（控制器只查本地 runtime 分区），列入操作者前置。
6. 控制器不改/不替代 scorer 任何门；worker 资产 SHA 终审仍在 scorer Gate 1/5。本轮未触碰远端 checkout、未创建任何远端 run/账本、未安装包、未跑任何模型前向。

## 7. 下一轮建议（非授权）

Codex 审码重点建议：wait/kill 循环（`launch()` 第 8 节）、spawn 失败分支、只读 git 白名单、`_validate_config` 与 `_task_md` 解析钉死；审后另写 ACTIVE（pilot_authorized + 完整批准 SHA + 29319624 对 + 336 卡小时 + run_id）→ push → 远端按 §4 前置检出并 nohup 启动 → 运行期 Kimi 只读 heartbeat 监控 → 两卡 exit0/finalized 后矩阵/账本/归档审核，再单独激活 CPU 评价。任何源代码/配置改动使本实现审核失效，须重新提交。

## 8. 状态与回填

**review_required**。实现提交 `0bba6a7acb573aefbe1797a043b6dafe974e61e8`（git status 干净、未 push）；本报告提交 SHA：（回填处）。
