# Text2CAD M1c G1 批一致性与优化等价门结果报告（2026-10-01）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 子计划：`experiments/plans/IDEA01_M1C_G1_BATCH_2026-10-01.md`；上位合同 `IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`；前置审查 `experiments/reviews/IDEA01_M1C_G1_SPOT_CODEX_REVIEW_2026-10-01.md`、`IDEA01_M1C_STAGE_B2_CODEX_REVIEW_2026-10-01.md`
- 授权依据：origin/master `ba58e66c531e702c926c78d27bc4496b715ef17e` 的 `ACTIVE_TASK.md`（`pilot_authorized`，G1 固定十对 batch/trim/cache 三门，90 对 / 1 卡小时 / 仅 npu:0）
- 运行代码 SHA（approved == actual）：`f0d348568f60ac2956deeb058aca7f4873afad79`（本地核对：相对已审 `24570dd…` 源码/脚本/配置/测试零差异，仅文档提交；相对 `ba58e66` 同样仅文档差异）
- 固定 run_id：`idea01_text2cad_g1_batch_20261001`；三个 invocation：`20261001T042016Z-7363f0a8`（batch-consistency）、`20261001T042206Z-8dfb3e97`（trim-check）、`20261001T042313Z-08bdb3fa`（cache-check）

本文件是本轮唯一新增交付文档，未改动任何源码、配置、计划、审查或既有报告；未做 git commit/push；未修改 `ACTIVE_TASK.md`。

## 1. 事实（Facts）

### 1.1 前置核验（全部通过后才启动）

- 远端 checkout `/home/ma-user/work/CADRetrieval` 工作树干净（`git status --porcelain` 0 行），fetch 后 `origin/master = ba58e66…`（其 ACTIVE_TASK 授权 90 对 / 1 卡小时 / 仅 npu:0），随后 `git checkout --detach f0d348568f60ac2956deeb058aca7f4873afad79`，`git rev-parse HEAD` 逐项一致。
- `npu-smi info`：两张 Ascend910 均 OK、无进程、HBM 0 占用。
- 目标输出 `runtime/runs/idea01_text2cad_g1_batch_20261001*` 不存在（无同 stage 既往结果）；预算目录仅有旧 SHA `24570dd…` 的 spot 账本，本 SHA 账本尚未创建。
- `pair_manifest_v4/pair_manifest.json` SHA256 == `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`（复核值）。未读 qrels/CSV/UID 映射。
- 逐门前置快照见 `budget_snapshot_before_*.txt`；运行后核验（HEAD 未变、工作树干净、NPU 无残留、资产 SHA 复算）见 `post_run_verification.txt`。

### 1.2 执行与退出状态

- 服务器 `yunnan-share2`，环境 `/home/ma-user/work/envs/cadretrieval-idea01-torch271`（Python 3.11.10、torch 2.7.1+cpu、torch_npu 2.7.1.post1、transformers 4.53.1，Linux aarch64），`PYTHONDONTWRITEBYTECODE/HF_HOME/XDG_CACHE_HOME/TMPDIR/OMP/MKL` 均按 B2 review 固定参数指向 checkout 外。
- 三门命令与合同逐字一致（`--device npu:0 --cards-used 1 --batch-size 1`，FP32；batch-consistency 另带 `--consistency-batches 2,4,8,10`），完整命令逐字保存于 `*_command.txt`。
- 退出码：**batch-consistency 0、trim-check 0、cache-check 0**；各自 stdout 仅一行状态 JSON（`status: ok`）；stderr 仅 torch_npu `collect_env` owner UserWarning 及一条驱动版本 W1001 提示，无错误（逐行核对，非警告行数为 0）。
- 顺序门控执行：先 batch-consistency，确认 `passed=true` 后才依次启动 trim-check、cache-check。未重试任何已扣账前向。

### 1.3 固定十对与各门数值

十对（text_index/cad_index，文本字符数/CAD 真实 token 数）：text 2218(71)/271(153)/3016(303) × cad 3311(23)/1531(34)/2537(87) 的 3×3 全网格 + (793,2537)（469 字符 × 87 token），覆盖短/中/长文本与短/中/长 CAD；其中 (271,1531) 即 G1 spot 对。锁定容差：logits rtol=1e-4/atol=1e-5，scores rtol=1e-6/atol=1e-4（三份结果 JSON 记录值与此逐项相等；看到结果后未做任何调整）。

**门 1：batch-consistency**（参考=batch=1 逐对，对照=batch 2/4/8/10，共 50 对扣账）`passed=true`

| 对照档 | max abs logit diff | logit 违规数 | 计数不一致对数 | 分数超容差对数 | max score_total diff |
| --- | --- | --- | --- | --- | --- |
| batch=2 | 1.252e-05 | 0 | 0 | 0 | 1.189e-05 |
| batch=4 | 1.252e-05 | 0 | 0 | 0 | 1.283e-05 |
| batch=8 | 1.645e-05 | 0 | 0 | 0 | 1.285e-05 |
| batch=10 | 1.657e-05 | 0 | 0 | 0 | 1.706e-05 |

- 比较口径：仅目标为真实 token 的预测行（`n_rows_compared = n_real_tokens-1`，十对为 22/33/86 行），padding 行不参与；`n_valid_x/y/total`、`valid_seq_len`、`n_real_tokens` 全部精确相等。
- 冷启动 32.85 s，测量窗口 34.97 s（2026-10-01T04:20:15–04:20:50Z UTC），invocation 计 34.48 卡秒。

**门 2：trim-check**（全长参考 vs 裁到 batch 最大长度的同 10 对 batch=10，共 20 对扣账）`passed=true`

- max abs logit diff 1.657e-05，logit 违规 0，计数全精确相等，分数超容差 0，max score_total diff 1.706e-05。
- 冷启动 26.90 s，窗口 28.54 s（04:22:06Z 起），计 28.06 卡秒。

**门 3：cache-check**（逐对文本编码参考 vs 唯一文本 BERT 缓存的同 10 对 batch=10，共 20 对扣账）`passed=true`

- max abs logit diff 5.722e-06，logit 违规 0，计数全精确相等，分数超容差 0，max score_total diff 1.198e-05。
- 冷启动 25.87 s，窗口 27.59 s（04:23:13Z 起），计 27.07 卡秒。

### 1.4 预算（同 SHA 共享账本）

- 账本 `runtime/budgets/text2cad_m1c_f0d348568f60ac2956deeb058aca7f4873afad79.json`（快照 `budget_ledger_final_f0d3485.json`）：**pairs_charged=90 / max_pairs=90，card_seconds_spent=89.61 / 3600**（约 0.0249 卡小时，上限 1 卡小时）。
- 3 个 invocation 分别扣 50/20/20 对、34.48/28.06/27.07 卡秒，全部 exit 0 且 finalized；无 stale 自动回收。旧 SHA `24570dd…` 的 spot 账本（1 对、33.64 卡秒）未动。
- 失败前向 0 次，重试 0 次。

### 1.5 代码与资产绑定（三份 bench JSON 内记录值，逐门一致）

- `authorization`：`approved_sha == actual_sha == f0d3485…`，授权来源 `origin/master:experiments/ACTIVE_TASK.md @ ba58e66…`，`task_state=pilot_authorized`，`max_pairs=90`，`card_hours=1.0`，`cards_used=1`。
- checkpoint SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`；BERT 五文件 SHA 逐项在案（model.safetensors `28349a5c…444a4c4`）；`pair_manifest_sha256=62086beb…`；`selection_sha256=dd07ec91782fa8156fa490f176d0fe0e3fa118488e687ce7bf14a04af29d834a`；score 名 `two_head_factorized_logprob_sum`，spec `text2cad-vector-score-v1`；FP32、`npu:0`、CLI batch_size=1。

### 1.6 本地独立复核（非远端程序自证）

- `verify_batch_local.py`（只读，本地运行）对三份 JSON + 账本做 **88 项检查全部通过（exit 0）**，输出 `verification_local.txt`：容差锁定值逐项相等；三门 `passed=true`；十对跨三门一致；档位恰为 2/4/8/10；计数精确、零 logit 违规；**三次独立 invocation 的 batch=1 参考分数两两逐对完全相等**；授权/资产/选择 SHA 与批准值逐项相等；账本 90/90、卡秒 ≤3600、invocation ID 与 JSON 对应。
- 跨阶段复核：batch-consistency 参考对 (271,1531) 的 `score_ref = -43.10905569791794`，与 2026-10-01 早些 G1 spot（代码 SHA `24570dd…`）的 `score_total` **逐位相等（abs diff = 0.0）**。

### 1.7 产物位置与哈希

- 本地镜像：`experiments/runs/idea01_text2cad_g1_batch_20261001/`（23 个文件：三份 bench JSON、逐门 command/stdout/stderr/exitcode、四个预算快照、post_run_verification、复核脚本与输出、`SHA256SUMS`）。
- 共享 OBS 归档：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g1_batch_20261001/`（目标路径事先确认不存在，未覆盖任何既有资产；上传后远端 `sha256sum -c` 23 项全部通过）。本地与远端同一文件哈希逐项一致（本地 `SHA256SUMS` 生成前已与服务器原件逐文件比对相等）。
- 关键 SHA256：batch-consistency JSON `7b98b93980f5d8df76262370ac5c1616e72ce157c9ae3cd7acbff79fadf18948`；trim-check JSON `609816327a77fd1f80834c4545f0dbb55ec859645b70756f282ec67fc2747827`；cache-check JSON `d2b85df935b0705f699106b1f16d5c7ad634f4019dba7f29b0ddcdaf331a0a7a`；预算账本快照 `be1491db31cc40edab67df779b87f27d41f23623971ea6838589ce62b9ef41ea`。完整清单见两地 `SHA256SUMS`。
- 远端原始输出仍留于 `yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_batch_20261001/`（checkout 外）；远端 checkout 保持干净 detached `f0d3485…`。

### 1.8 失败与偏离

- 无失败样本、无 OOM、无超时、无非有限值、无 qrels 接触、无计划外运行；失败比例 0/90；未触发任何 stop_conditions；未修改容差；预算恰用尽 90 对额度、卡时远低于上限。

## 2. 解释（Interpretation，非新增证据）

- 三门等价性在锁定容差内成立：batch=1 与 batch 2/4/8/10 的有效位置 logits 最大绝对差 ≤1.66e-05（ atol 1e-5 + rtol 1e-4·|ref| 的事前门内，零违规），总分差 ≤1.71e-05（门限 atol 1e-4）；trim 到 batch 最大长度与唯一文本 BERT 缓存在同 10 对上同样通过。差异量级与 NPU 上 batch 归约顺序变化的预期一致，不是系统性偏移。
- trim-check 的最大误差值与 batch=10 档完全相同（同一组 10 对、同一 batch 形状），属意料之中；cache-check 的 logit 最大差更小（5.7e-06），与缓存只替换文本编码路径一致。
- 三次独立进程（各自重载模型、重读 ZIP）的 batch=1 参考分数逐位相等，加上与昨日 spot 对跨代码 SHA 逐位相等，说明该评分实现在 npu:0 上按批=1 口径是可复现的确定性计算；但 batch>1 分数与 batch=1 并非逐位相等，只是在容差内有界。
- 按合同，trim/cache 等价门通过**不等于**优化已启用：本阶段生产口径仍是全长、无缓存、逐对/逐档显式 batch；是否把裁剪或缓存计入 G2 吞吐由 Codex 决定。
- 90 对 / 89.61 卡秒的实测不能外推全量 29,319,624 对的七天 ETA：冷启动占比高（每次约 26–33 s），稳态吞吐、测量窗口与开销下限仍是 B2 review 记下的 G2 未放行项。
- 本证据不支持任何检索精度或语义结论（未读 qrels、未算 R@K），也不构成 G2/全量授权。

## 3. 建议（Suggestions，待 Codex 决定，非授权）

- 建议 Codex 复核本报告、两地归档与账本后，决定是否在 G2 测速中启用 trim 和/或 text-cache 优化（其等价门本轮已在固定十对上通过；G2 的两项既有阻塞——测量窗口与 overhead 下限——仍需按 B2 review 处理）。
- 若启用优化计入吞吐，建议保留“同 pair 逐项等价门先于生产使用”的口径，并对更大 batch 档维持 batch-consistency 抽查。
- 建议沿用“失败也计预算、不自动重试”与逐 SHA 独立账本的做法到 G2/全量阶段。

results_submitted
