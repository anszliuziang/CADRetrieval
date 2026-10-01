# Text2CAD M1c G1 单对 spot 结果报告（2026-10-01）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 子计划：`experiments/plans/IDEA01_M1C_G1_SPOT_2026-10-01.md`；B2 审核：`experiments/reviews/IDEA01_M1C_STAGE_B2_CODEX_REVIEW_2026-10-01.md`
- 授权依据：origin/master `f0d348568f60ac2956deeb058aca7f4873afad79` 的 `ACTIVE_TASK.md`（`pilot_authorized`，1 对 / 0.5 卡小时 / 仅 npu:0）
- 运行代码 SHA（approved == actual）：`24570dd892202be4b5e31a4bdc12cb23dc258d57`
- 固定 run_id：`idea01_text2cad_g1_spot_20261001`；invocation_id：`20261001T040327Z-65da1389`

本文件是本轮唯一新增交付文档，未改动任何源码、配置、计划、审查或既有实施报告；未做 git commit/push。

## 1. 事实（Facts）

### 1.1 执行与退出状态

- 完整命令、逐条 pre/post 检查与退出码证据：`experiments/runs/idea01_text2cad_g1_spot_20261001/command_and_exit_code.txt`。
- 运行于 `yunnan-share2`，checkout `/home/ma-user/work/CADRetrieval`，命令与 review G1 段逐字一致（`--mode spot --device npu:0 --cards-used 1 --batch-size 1`，FP32、`trim_mode=none`、`text_cache=false`）。
- 退出码 **0**；stdout 仅一行状态 JSON；stderr 仅 4 条 torch_npu `collect_env` 关于 Ascend toolkit 目录 owner 的 UserWarning（环境采集提示，与评分数学无关）。
- 运行前：远端 HEAD == 批准 SHA、`git status --porcelain` 为空、origin/master == 授权提交、两张 Ascend910 OK 且无进程、runtime/budgets/tmp/hf/xdg 齐备、`pair_manifest.json` SHA256 == `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`、UID 映射仅在 checkout 外 `evaluation_inputs/m0_v3/`、输出目录不存在（无既往 forward 结果）。
- 运行后：远端工作树仍干净、HEAD 未变、NPU 无残留进程。

### 1.2 评分对象与数值（bench JSON 原值）

- spot 对：`pair_index=2181997`，`text_index=271`（153 字符），`cad_index=1531`（34 个真实 token）。未读取 qrels/CSV/UID 映射。
- logits：shape `[271, 2, 267]`，`all_finite=true`，min `-4.7618`，max `9.0116`。
- 分数（`two_head_factorized_logprob_sum`，spec `text2cad-vector-score-v1`）：
  - `score_x = -19.92079045623541`（`n_valid_x=33`）
  - `score_y = -23.18826524168253`（`n_valid_y=8`）
  - `score_total = -43.10905569791794`（`n_valid_total=41`，`valid_seq_len=33`）
  - 诊断均值：`mean_x=-0.60366`、`mean_y=-2.89853`、`mean_total=-1.05144`
- terminal/padding：终止行 `token_detail.x[-1]`（position 32，label START=1）被计入 x；其 y 因子被 mask；y 仅 8 个有效项（position 0/2/4/6/10/12/14/16）；271 个预测行中仅 41 项计入，padding 零贡献。
- 冷启动 32.64 s；测量窗口约 34.2 s（04:03:47–04:04:21 UTC）。

### 1.3 资产与代码绑定（bench JSON 记录值）

- checkpoint SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`（== ACTIVE_TASK 固定值）；BERT 五文件 SHA 逐项记录在案。
- selection SHA256 `dd07ec91782fa8156fa490f176d0fe0e3fa118488e687ce7bf14a04af29d834a`（== review 记录的重新生成版）。
- 环境：Python 3.11.10、torch 2.7.1+cpu、torch_npu 2.7.1.post1、transformers 4.53.1，Linux aarch64；`HF_HOME/XDG_CACHE_HOME/TMPDIR/PYTHONDONTWRITEBYTECODE` 均指向 checkout 外。

### 1.4 预算

- ledger（`budgets/text2cad_m1c_24570dd....json` 快照已取回）：恰好 **1 个 invocation，`pairs_charged=1`，`remaining_pairs=0`**，`card_seconds_spent=33.64` ≤ 1800（0.5 卡小时），exit 0、已 finalize。本轮未运行第二对、十对、G2、CPU 参考或任何评价。

### 1.5 本地独立复核（非远端程序自证）

- `verify_spot_local.py` 对取回的 bench JSON 做 26 项只读复核，**全部 PASS（exit 0）**，输出在 `verification_local.txt`。要点：token detail 逐条相加精确复算 `score_x`/`score_y`/`score_total`（误差 <1e-9）；x 位置 0..32 稠密；终止行 label==1 计入且无对应 y；有效计数与均值关系一致；预算恰扣 1 对；代码/资产/选择 SHA 与批准值逐项相等。

### 1.6 产物位置与哈希

- 本地镜像：`experiments/runs/idea01_text2cad_g1_spot_20261001/`（bench JSON、stdout/stderr、预算快照、命令与退出码证据、复核脚本与输出、`SHA256SUMS`）。
- 共享 OBS 归档：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g1_spot_20261001/`（仅存储操作，未用 GPU；上传后 `sha256sum -c SHA256SUMS` 远端全部通过；目标路径事先确认不存在，未覆盖任何既有资产）。
- 关键 SHA256：bench JSON `c5e08075c834273562b0b4ed5326f356f0f59d7ceadf5317813c1609e687e07c`；stdout `1cb43bf5b33e50898329311e6c61cc8b5deec168dac4f2a604b6423c4fd60506`；stderr `1ee8090f4ed15e1399485ef9f16234342912cb3bdd13869c4f7039a2067a0f7c`；预算快照 `9818d7ac873871df30881e4161b6b26c515027b0839019e47661f60c502c8d39`。完整清单见两地 `SHA256SUMS`。
- 远端原始输出仍留于 `yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_spot_20261001*`（checkout 外）。

### 1.7 失败与偏离

- 无失败样本、无 OOM、无超时、无 qrels 接触、无计划外运行。失败比例 0/1。未触发任何 stop_conditions。

## 2. 解释（Interpretation，非新增证据）

- 本轮验收口径全部满足：原始两路 logits 全有限；shift 后 x/y 有效计数与 token detail 一致；terminal `[START]` 的 x 因子计入、y 与 padding 排除；sum 可由 token detail 独立复算；代码-资产-选择-授权绑定逐项吻合；ledger 恰扣 1 对。
- 分数命名为 `two_head_factorized_logprob_sum`，是冻结模型两路原始 logits 上的条件排序分数；按 ACTIVE_TASK 要求，不称其为 CAD 几何的精确 NLL，也与含 label_smoothing=0.1 的训练 loss 无关。
- y 头均值（-2.90）明显低于 x 头（-0.60），与 y 因子为连续量离散化后高熵通道的预期一致；单个对不能支持任何语义结论。
- 33.64 卡秒含约 32.6 s 冷启动；单对 spot 的耗时结构不能外推全量吞吐，G2 测速的两项已知缺口（测量窗口、开销下限）仍按 review 处于未放行状态。
- 本证据不支持 R@K、语义检索能力或七天耗时结论；它只证明该 SHA 在 npu:0 上的评分实现按合同正确执行了一次。

## 3. 建议（Suggestions，待 Codex 决定，非授权）

- 建议 Codex 复核本报告与两地归档后，按既定路线决定是否放行 G2 测速（其两项阻塞修复与本轮无冲突）。
- 若后续需要 CPU 参考比对，建议作为单独授权阶段，明确样本与预算。
- 建议保持“失败也计预算、不自动重试”的口径沿用到 G2/全量阶段。

results_submitted
