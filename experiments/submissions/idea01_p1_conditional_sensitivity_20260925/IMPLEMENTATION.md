# IMPLEMENTATION — idea01_p1_conditional_sensitivity_20260925（第一门：仅实现 + CPU 选择预核）

状态：**review_required**。P1 第一门（代码实现 + CPU-only 选择预核）已完成，等待 Codex 代码审查。本门 **0 卡小时、0 scored pairs**：任何机器上都未载入模型权重、未做模型前向、未评分任何配对、未计算 R@K、未启动 P2/P3。第二门（NPU 受限真实运行）**未获授权**，须待 Codex 在 `ACTIVE_TASK.md` 写入 `pilot_authorized` 与完整审核 SHA。

## 1. 任务与版本

- 任务合同：`experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md`（上位计划 `IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md`），入口 `experiments/ACTIVE_TASK.md` state=`implementation`；P0 审核 `experiments/reviews/IDEA01_P0_CODEX_REVIEW_2026-09-25.md`。
- **代码 Git SHA（已推送 origin，服务器已检出运行 CPU 预核）**：`10c734451ee22e608eaf0d119521c092fc7b8f62`（分支 `worktree-idea01-p1-impl`，基 master `fdd9adf`）。本文件为交接文档，提交在其后；**待审实现代码以此 SHA 为准**。
- 模型：`ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`——本门仅读取 tokenizer（slow `Qwen2Tokenizer`，`eos_id=151645`），未触碰权重。
- 数据：`ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，仅 `data_val.jsonl` + CSV/ZIP 映射与既有 P0 审计清单。
- 评分核心：`experiments/src/idea01/scoring.py` 原样复用 P0 已审核版本（本提交未改它、prompts.py、coverage.py、data_audit.py、envspec.py）。

## 2. 改动文件（全部本地编写，Git 同步，服务器未编辑任何源码）

| 文件 | 用途 |
| --- | --- |
| `experiments/src/idea01/p1_selection.py` | 合格池判定、固定抽样（seed 20260925）、3 个共享循环偏移、方法名 Jaccard/长度近邻、8 候选唯一性、256+≤160 配对计划、≤416 预算硬断言；`SelectionError` 即停语义 |
| `experiments/scripts/run_p1_selection.py` | CPU-only 驱动：独立复算合格池（对照 4,949 参考）、与 P0 清单交叉核验、全配对 tokenize 门（边界/EOS/≤1024），失败非零退出 |
| `experiments/src/idea01/p1_runner.py` | 第二门纯辅助：pair-id 解析（≤416 去重硬上限）、状态与停止策略（overlong 也停，因 CPU 门已验证 ≤1024）、退出码 |
| `experiments/scripts/run_p1_matrix.py` | 第二门 NPU 评分调度（**本门未运行**）：复用已审核 scorer；逐对完整 input_ids、模型/数据路径、双环境哈希、选择清单 sha256、每次调用独立 environment 文件、异常即停非零、45 分钟/416 前向预算 |
| `experiments/src/idea01/p1_analysis.py` | 预定分析：同候选边际（候选 UID bootstrap CI）、8 候选配对 UID 名次（sum/mean/corrected 三臂，query bootstrap）、随机/长度/空 query/字符 3-gram TF-IDF 基线、分层诊断、近重复 token Jaccard 粗筛 |
| `experiments/scripts/run_p1_analysis.py` | 分析驱动：只从原始 pairs.jsonl 复算；记录不全/失败一律非零退出（拒绝只报成功子集） |
| `experiments/scripts/verify_p1_run.py` | 只读复核：从逐 token 文件重算 sum、校验 shift/边界/EOS/计数、原始记录与清单配对集合一致性 |
| `experiments/configs/idea01_p1_conditional_sensitivity_20260925.json` | 任务配置（revision、路径、种子、预算、分析参数） |
| `experiments/tests/test_p1_selection.py`、`test_p1_analysis.py`、`test_p1_matrix_runner.py` | 47 项新测试（合计 89 项） |

## 3. 实现方法与关键决定（对应计划 §2/§3，含解释性决定）

1. **合格池独立复算**：不读 P0 覆盖表做判定，而是从 pinned 原始数据重算：JSONL `input` == CSV `intermediate` 逐字映射唯一 UID、ZIP `CQ/{uid}.py` 与 JSONL `output` 逐字相同、slow tokenizer 逐行 prefix/full 编码得边界/EOS/token 数；再按 `n_prefix_tokens<=250`、`n_response_tokens<=700`（含 EOS）过滤。结果 **4,949/5,513 = 89.77%**，与 Codex 只读参考值一致；与 P0 `uid_manifest_val.jsonl`、`coverage_val.jsonl` 逐行交叉核验 **0 不符**（本地与服务器两次均如此）。排除计数：prefix 超长 118、response 超长 521（两条件可重叠；无 UID/边界/EOS/ZIP 排除）。
2. **固定抽样**：池按 UID 升序后 `random.Random(20260925)` 先 `sample(pool,32)`（保留抽样顺序为固定 query 序），**同一 RNG 实例**再 `sample(range(1,32),3)` 得偏移 **[27, 8, 23]**，对所有 query 复用；每个被选中程序恰好作为 3 个其他 query 的候选（`offset_coverage_check` 通过）。
3. **近邻规则**：方法名 = 原样程序正则 `\.([A-Za-z_][A-Za-z0-9_]*)\(`，排除 `{export, exporters}`；两空集 Jaccard 定义为 0.0。方法近邻 = 池−32 个 UID，按 (Jaccard 降序， |response 长度差| 升序， UID 升序) 取 2；长度近邻 = **池−32 个 UID−本 query 的 2 个方法近邻**（解释性决定：保证 8 候选互异；计划未明说“剩余合格池”是否再减方法近邻，此不放宽规则的最严读法），按 (|长度差| 升序， UID 升序) 取 2。任何 query 凑不满 8 个不同候选即 `SelectionError` 停止，不放宽。
4. **配对计划**：256 条件对 + 158 空 query 对 = **414 ≤ 416**（空对按去重候选各一次；两个近邻 UID 各被 2 个 query 共享，故 158 而非 160）。`pair_id`：`q{ii}_{c}_{self|off|method|length}` / `e{nnn}`。
5. **CPU tokenize 门**：414 对全部 slow-tokenize 并断言严格前缀边界、id 级 EOS、`len(input_ids)≤1024`：**414/414 通过，最大实测 697 token，全部以 EOS 结尾**（资格上限 250+700 只约束自身配对，交叉配对逐对实测是必要的，本门已做）。
6. **评分/分析隔离**：选择与分析只用数据与种子，**从不读模型分数**；评分路径只接收 query 文本与候选 `output`；UID/行号/来源仅写入输出记录簿记。manifest 记录全部规则、正则、排除词、seed、偏移、每候选来源/相似度/前缀与 response token 长度、合格池与全集长度差异（池 response 均值 299.3 vs 全集 367.0，池偏短，已如实记录）、逐 query 近邻可用池大小（如 q0 方法池 4,917、长度池 4,915）。
7. **第二门调度（仅交付代码）**：`run_p1_matrix.py` 继承 P0 审核修复（逐对完整 input_ids、双环境哈希、按调用 environment 文件、异常记录后立即停并非零、墙钟预算非零）；P1 特有：配对计划 >416 或清单报告 CPU 门失败则拒绝运行；超长计划配对视为异常停止（CPU 门已验证 ≤1024，运行期超长 = 环境漂移）；记录选择清单文件 sha256 绑定分数与已审清单；`eos_logprob` 直接进 pairs.jsonl 供分层诊断。

## 4. 测试证据（Windows 本地，Python 3.14.0 + numpy 2.5.2 + transformers 4.53.1 tokenizer-only）

- `python -m pytest experiments/tests/ -q` → **89 passed**（P0 既有 42 + 新增 47）。
  - 选择：正则/排除词、Jaccard 空并集、逐准则排除计数、池 <32 停、UID 重复停、同种子可复现、偏移 1..31 互异、偏移恰好 3 覆盖、近邻排除 32 个 UID、长度近邻排除方法近邻、tie-break（Jaccard 同分→长度差→UID）、凑不满即停、配对计数/唯一 id/字段、>416 预算停。
  - 分析：bootstrap 同种子确定且换种子变 CI；`rank_of_self` 并列语义；合成分数下同候选边际精确等于注入值、篡改一个偏移分数边际恰好移动 1/3、三臂 top-1/胜率、corrected 臂确实使用空 query 分数（sum 臂不受影响）；基线（随机定种子可复现、长度升序、TF-IDF 文本匹配时 self top-1、cosine 数学）；近重复 exact 同文触发数据准入错误；`run_p1_analysis.main()` 合成端到端（产出 metrics.json/逐 query 表/近重复清单）+ 缺对拒绝（退出 2）。
  - 调度辅助：pair-id 解析/去重/上限、计划 >416 拒绝、全异常状态即停、退出码互异。
- **本地全流程彩排**（`.cache/idea01_assets` 的 SHA 已核验 pinned 数据与 tokenizer，输出 `experiments/runs/idea01_p1_20260925_plans/local_rehearsal/`）：4,949/5,513、P0 交叉 0 不符、414 对门全过、44.2 s、退出 0。
- **服务器正式 CPU 预核**（下节）：与本地彩排语义逐字段一致（pool/tokenize 文件仅换行符差异，内容全等）——同一种子跨 Windows/Linux 复现。

## 5. 远端执行（yunnan-share2，CPU-only，输出全部在 checkout 外）

```bash
cd /home/ma-user/work/CADRetrieval
git status --porcelain                      # 为空（干净）
GIT_TERMINAL_PROMPT=0 git fetch origin
git checkout --detach 10c734451ee22e608eaf0d119521c092fc7b8f62
git rev-parse HEAD                          # 与所检 SHA 完全一致
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=.../hf_cache XDG_CACHE_HOME=.../xdg_cache TMPDIR=.../tmp
/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_p1_selection.py \
  --data-dir /home/ma-user/work/CADRetrieval_runtime/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3 \
  --model-dir /home/ma-user/work/CADRetrieval_runtime/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe \
  --out-dir /home/ma-user/work/CADRetrieval_runtime/plans/idea01_p1_20260925 \
  --config experiments/configs/idea01_p1_conditional_sensitivity_20260925.json \
  --p0-uid-manifest .../runs/idea01_p0_20260924_r01/audit/uid_manifest_val.jsonl \
  --p0-coverage     .../runs/idea01_p0_20260924_r01/coverage/coverage_val.jsonl
# 退出 0；运行后 git status --porcelain 仍为空
```

**权威选择清单**（服务器，checkout 外）：`/home/ma-user/work/CADRetrieval_runtime/plans/idea01_p1_20260925/`
- `p1_selection.json` sha256 `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964`
- `p1_eligible_pool.json` sha256 `023f52820666b71daa4bc7677df22e35ef4b7672ebff10b4db0d14e5c6e6d5c1`
- `p1_pair_tokenization.jsonl` sha256 `650dfa816aad701791b2b6849c99d25524f26a0119883e8f3453adc39bb71a36`

本地镜像（Git 忽略）：`experiments/runs/idea01_p1_20260925_plans/server/`（逐字节取回）与 `.../local_rehearsal/`。

**第二门准备命令（未授权，仅供 Codex 审核计划；先 1 对、再 10 对、后全量，输出 `.../runs/idea01_p1_20260925_r01/`）**：

```bash
# 前置：模型/数据两处副本 sha256sum -c SHA256SUMS；checkout 干净且 HEAD == 审核 SHA
python experiments/scripts/run_p1_matrix.py --config experiments/configs/idea01_p1_conditional_sensitivity_20260925.json \
  --selection .../plans/idea01_p1_20260925/p1_selection.json --out-dir .../runs/idea01_p1_20260925_r01 \
  --pair-ids "0" --device npu:0          # 审逐 token/有限值/HBM/耗时后再 "0-9"，最后 "all"
python experiments/scripts/verify_p1_run.py --run-dir .../runs/idea01_p1_20260925_r01 --selection .../p1_selection.json
python experiments/scripts/run_p1_analysis.py --pairs .../runs/idea01_p1_20260925_r01/pairs.jsonl \
  --selection .../p1_selection.json --data-dir <数据副本> --out-dir .../runs/idea01_p1_20260925_r01/analysis
```

## 6. 资源实测与第二门预计成本

- 本门实际：**0 NPU、0 前向**。CPU 预核远端 43.6 s（含 CSV 索引 + 5,513 行 slow tokenize + 414 对门），checkout 外输出 460 KB。
- 第二门预计（按 P0 实测外推：单对 0.18–0.87 s、最长 745 token、峰值 6.99 GB、加载 ~19 s）：414 对、最大输入 697 token → 前向约 **1.5–6 分钟**，含加载与逐对落盘总墙钟 **<10 分钟**，峰值 HBM **≈7 GB**，卡时 **≈0.03–0.12 卡小时**——三项预算（416 前向 / 45 分钟 / 1 卡小时）余量充足。

## 7. 选择清单摘要（权威件同上）

- 合格池 **4,949/5,513（89.77%）**；排除：prefix>250 共 118、response>700 共 521（有重叠）；UID/ZIP/边界/EOS 排除均 0。
- seed 20260925；偏移 **[27, 8, 23]**；32 query；每 query 8 候选（self/3 偏移/2 方法/2 长度）；每被选中程序恰为 3 个他 query 的偏移候选。
- 数据事实（非问题，供解读）：方法名集合相似度普遍很高（64 个方法近邻 Jaccard 最小 0.917、多为 1.0——CadQuery 方法词汇高度共享，tie-break 常落到长度差与 UID）；长度近邻 |长度差| 仅 0–1 token；2 个近邻 UID 各被 2 个 query 共享（故去重后 158 个空对）。近重复/几何等价不在本门判断，第二门后由分数无关 token Jaccard 粗筛量化。
- 配对：256 条件 + 158 空 = **414 ≤ 416**；CPU 门 414/414 通过，输入最长 697 ≤ 1024，全部 EOS 结尾。

## 8. 已知缺口与声明

- 本门不含任何模型分数；P1 一切排名结论在第二门后才存在，且全部为 **exact-UID 配对 sanity**，不是独立语义检索 R@K。UID ≠ 设计家族；近重复/几何等价未闭合前不启动 P2/P3。
- `run_p1_matrix.py`、`run_p1_analysis.py`、`verify_p1_run.py` 已交付并单测/合成端到端验证，但**未在真实模型/真实分数上运行**；第二门须在新审核 SHA 上先做 1 对/10 对受限验证。
- 无 BLOCKERS（首版交付时未受阻；补正阶段新增 R3 相关 OBS 阻塞，见文末补正记录与 `BLOCKERS.md` B1）。
- 本地为跑 tokenizer 彩排新装了 `transformers==4.53.1`（tokenizer-only，与服务器版本一致；不进入仓库依赖声明，仅本地工具）。

---

# 补正记录（2026-09-25，响应 Codex 代码审核 R1–R4 及追加阻塞项）

状态：**review_required**（补正后复审）。上文 §1–§8 保留为首版交付原貌；本节及以后的命令/规则**取代** §5 的第二门准备命令。

- **补正代码 SHA（已推送 origin，服务器已检出做 CPU 复核）**：`0a513a7810604b7e1e5c62acda10bdef78db2072`（分支 `worktree-idea01-p1-impl`；含 R1–R4 修复 `c55b71c`、origin/master 授权修复 `ccab7b5`、顺序与解析修复本提交；并已合入 origin/master `f54af46` 的审核补充说明）。
- 选择逻辑（`p1_selection.py`、`run_p1_selection.py`）**零改动**；权威 CPU 清单 `p1_selection.json` sha256 `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964` 未被覆盖，仍是待评清单（审核要求不覆盖原件；补正 SHA 的复跑见下方证据，写在新路径）。

## R1（分阶段重复评分/预算重置）— 已修复

- `run_p1_matrix.py` 在**载入权重前**核验已有 `pairs.jsonl`：pair_id 唯一、status ok 且有限、selection/git/model/data/dtype/双环境哈希逐字段一致（`verify_existing_records`）；任何重复、失败或混版记录直接拒绝（非零）。
- 续跑必须显式 `--resume`：跳过已验证完成的配对并逐条打印（非静默）；不带 `--resume` 且目录已有记录即拒绝。
- 新增 `budget_ledger.json`（输出目录内）：逐次调用登记，累计前向数与墙钟；三项全程上限（**416 前向 / 2700 s / 1 卡时**，配置只能收紧不能放宽，CLI 已删 `--max-wallclock-s`）跨全部调用统一约束，触顶即非零（`EXIT_BUDGET=3`）；台账有未收尾调用（崩溃）即拒绝继续并提示报告 Codex，不自动续跑。
- 阶段方案固定为 **1 → 新增 9 → 剩余 404**：`--pair-ids "0"`（新目录）→ `--pair-ids "0-9" --resume`（新 9 对）→ `--pair-ids "all" --resume`（新 404 对）；结束后恰好 **414 条唯一原始记录**，累计前向 414 ≤ 416。

## R2（验证器放过篡改分数）— 已修复

- 新模块 `experiments/src/idea01/p1_verify.py`：逐对核对 token 和与记录 sum（相对容差 1e-6）、mean==sum/n、eos_logprob==末 token logprob、目标位置恰为 `n_prefix..n_prefix+n_target-1` 连续、`pred_row==position-1`、`token_id==input_ids[position]`、`input_ids` 长度一致且末位为 EOS；缺 token 文件、空列表、`all_finite` 为假、status 非 ok、重复/缺失/多余 pair 全部计入失败并非零退出。
- `verify_p1_run.py` 重写调用该模块；`run_p1_analysis.py` **先验证后分析**，验证不过即非零拒绝（分析不再接受未验证或成功子集记录）。
- **反向复现**：审核构造的篡改例（逐 token 和 -2 vs 记录 sum -1）旧脚本 exit 0；新脚本 **exit 2** 并打印 `FAIL: p000: token-sum -2.0 != record sum_logprob -1.0`（脚本级复现，命令与输出留存于测试记录，见 §证据）。

## R3（授权与资产/清单绑定）— 已修复（含追加阻塞项）

- 载入权重前强制前置校验（任一不符即非零停止，不只记录）：
  1. **授权来源（追加阻塞项修复）**：校验 `origin` URL 为预期远端 `https://cnb.cool/lzacode/CADRetrieval` → `git fetch origin`（`GIT_TERMINAL_PROMPT=0`）→ 读 **`origin/master:experiments/ACTIVE_TASK.md`**（**不再读 detached HEAD 工作树**——提交不可能包含自身 approved_sha）。要求 `state=pilot_authorized`、`approved_sha == 当前 HEAD`、`approved_selection_sha256 == 清单 sha256`、`task_id` 匹配；授权来源（ref、origin/master SHA、blob sha256、fetch 返回码）写入 environment 文件备审。
  2. **资产核验**：模型与数据运行副本逐文件 Python 版 `sha256sum -c SHA256SUMS`（授权通过后才做，避免无谓哈希 8 GB）。
  3. 已删除 `--model-dir`/`--data-dir` 覆盖；清单自报 `n_gate_fail!=0` 或配对 >416 即拒绝。
- **给 Codex 的授权格式约定**：授权提交在 master 的 `ACTIVE_TASK.md` 写入 `` `state`: `pilot_authorized` ``、`` `approved_sha`: `<已审核代码完整SHA>` ``、`` `approved_selection_sha256`: `c86fb56993af62f08079ff3f8f88af3f2d1ad33ac28cd2273ae0d7bb03028964` ``；runner 每次调用重新 fetch 校验。
- **OBS 阻塞**：910C `/data/data_dev` 挂载仍有 D 状态进程（PID 2602998，2026-09-25 只读 `ps` 复核仍在）；按指示未在该挂载启动任何新校验进程，**910C OBS 补证副本仍记为未核验**。详见同目录 **`BLOCKERS.md`（B1）**；等待挂载恢复或 Codex 修订资产核验路径。

## R4（失败证据与输出路径）— 已修复

- 环境文件名 = 时间戳+uuid8（`environment_<id>.json` 不再可能覆盖）；`--out-dir` 必须先通过"在 Git checkout 外"检查，否则在写任何文件前非零退出。
- 模型加载/数据读取等全部纳入全局异常处理：任何异常写 `run_failure_<id>.json`（含 traceback）并非零退出；前置校验失败写 `preflight_failure_<id>.json`；台账在 `finally` 中收尾，异常也按非零记账。

## 补正后的测试与服务器 CPU 证据

- 本地 Windows `python -m pytest experiments/tests/ -q` → **126 passed**（89 → 126；新增含：1→9→404 三次调用台账模拟恰好 414 唯一记录且累计 414≤416、篡改/混版/崩溃台账拒绝、R2 全部篡改路径、fabricated bare-remote 两段式授权往返、错误远端拒绝、SHA256SUMS 核验、预算钳制、checkout 外守卫）。
- 脚本级负例（本地）：篡改分验证器 exit 2（原为 0）；`run_p1_matrix.py` 在脏树 exit 2；干净树 + 真实 `git fetch origin` 后因 `review_required` **exit 2 且未载入权重**。
- 服务器 `yunnan-share2`（checkout `--detach 0a513a7`，全程 checkout 外输出，运行后 `git status --porcelain` 为空）：
  1. `python -m pytest experiments/tests/ -q -p no:cacheprovider` → **126 passed**（CPU，3.5 s）。
  2. 选择复跑到**新路径** `.../plans/idea01_p1_20260925/recheck_0a513a7/`（权威清单未触碰）：4,949/5,513、414/414 门过；复跑清单与权威清单**语义逐字段一致**（仅时间戳/路径/耗时等易变字段不同）。
  3. `run_p1_matrix.py --pair-ids "0"` 负例：**在哈希资产之前**因 `review_required` 拒绝，**exit 2**，`preflight_failure` 证据写在 checkout 外，未载入权重。

## 补正后的第二门拟定命令（取代 §5；仍须 Codex 授权，当前未授权）

```bash
cd /home/ma-user/work/CADRetrieval && git status --porcelain        # 必须为空
GIT_TERMINAL_PROMPT=0 git fetch origin
git checkout --detach <Codex 审核完整 SHA> && git rev-parse HEAD     # 与批准 SHA 一致
# origin/master 的 ACTIVE_TASK 须已为 pilot_authorized 且 approved_sha/approved_selection_sha256 匹配（runner 会再 fetch 校验）
RUN=/home/ma-user/work/CADRetrieval_runtime/runs/idea01_p1_20260925_r01
SEL=/home/ma-user/work/CADRetrieval_runtime/plans/idea01_p1_20260925/p1_selection.json
CFG=experiments/configs/idea01_p1_conditional_sensitivity_20260925.json
python experiments/scripts/run_p1_matrix.py --config $CFG --selection $SEL --out-dir $RUN --pair-ids "0"
python experiments/scripts/run_p1_matrix.py --config $CFG --selection $SEL --out-dir $RUN --pair-ids "0-9" --resume   # 新增 9 对，估时
python experiments/scripts/run_p1_matrix.py --config $CFG --selection $SEL --out-dir $RUN --pair-ids "all" --resume   # 剩余 404 对；累计恰 414
python experiments/scripts/verify_p1_run.py --run-dir $RUN --selection $SEL                                           # 必须先通过
python experiments/scripts/run_p1_analysis.py --pairs $RUN/pairs.jsonl --selection $SEL --data-dir <数据副本> --out-dir $RUN/analysis
```

任何一步非零即停并保留证据；台账未收尾（崩溃）不自动续跑，报告 Codex。

