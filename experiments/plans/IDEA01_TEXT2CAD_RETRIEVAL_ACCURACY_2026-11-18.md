# 冻结 Text2CAD 的检索提精度实验

2026-11-18 夜间。用户授权 Codex 总控，GLM‑5.3 先审方案、实施、汇报。Codex 分阶段验收方案、代码、小批真实前向、正式原始结果。

## 要回答的问题

在相同文本和完整图库的登记 UID 任务上，将 Text2CAD 的分数拆为不同字段，再学习低维权重，能否让正确 CAD 战胜近邻？原生 SUM 是主对照，只扣候选背景是第二对照。分组加权须超过背景校正才有额外价值。检查 R@1/5/10/20、MedR、配对区间、命中数和分层。早先 G5/E2 用于提出假设。生成模型参数固定。 [G5](../submissions/idea01_text2cad_benchmark_20260928/CODEX_TEXT2CAD_FULL_RETRIEVAL_RESULTS_2026-10-03.md) · [E2](../submissions/idea01_next_phase_20261112/E2_RESULTS_2026-11-14.md)

## 输入、拟合与留出

- 使用现有 `Text2CAD_1.0.pth`、BERT、`cad_seq.zip`、`text2cad_v1.0.csv` 和 `train_test_val.json`。验证文本从 v1.0 的 `uid,abstract` 读取。G5 测试从既有认证工件恢复，历史构建涉及 v1.1 UID 关联。文件和 SHA-256 见下方输入约定。只用 `yunnao-share4` 昇腾。服务器从 Git 检出经审代码的独立 clean detached worktree，保留原 checkout 中的未跟踪内容。运行、数据与日志均落到 checkout 外。
- **验证图库**为 v1.0 官方 validation 中全部有可核 CAD 的对象，按 UID 升序核对 ZIP 成员唯一。L0 取 `abstract.strip()`，缺失和空文本查询剔除，分组后保留同文对应每个原 UID。排除与 G5 测试逐字相同的验证文本组，按 `SHA256("cad-rank-dev-20261118\n" + text)` 的字典序取前 48 组作背景、接下来 256 组拟合、再接下来 256 组选型。实际条数、重合及选择顺序落盘。
- **测试图库**为原 G5 的全 8,046 CAD，按认证 `cad_store` 逐项对齐。文本和原查询从已有 G5 query/gallery/qrels 恢复。只 `strip()` 后逐字去重，按 `SHA256("cad-rank-test-20261118\n" + text)` 的字典序固定前 1,024 个测试文本组，组内保留全部原查询。若夜间实测吞吐限制规模，在看测试分数前取固定顺序的更小前缀并记录，图库保持完整。G5 曾用于历史诊断，新的读出从 validation 选定且冻结后作一次 G5 同协议比较。设计家族外推仍需独立基准。
- 两个图库用相同 **48 个验证背景文本**估计候选背景。验证与 G5 测试图库 UID 交集必须为零。记录验证可选文本中与 G5 测试逐字相同的组数，以及用 Unicode NFKC、casefold、标点转空格并合并空白后的撞组数，后一个计数仅作诊断。拟合、标准化和选型使用验证阶段资产。测试 qrels 仅在参数冻结并逐 SHA 核验后交给评价。评分入口只读当前文本、候选向量与固定模型。每条原查询只以自己的登记 UID 名次计分。G5 既往确认量化向量有跨分片重复。结果支持在此协议下的精确 UID 检索比较。
- `inputs.json` 固定 UTF-8、LF 和 canonical JSON。字段含 schema、axis、按索引排列的 texts 字符串及候选 members 字符串。`eval.json` 单独存 cad_uids、queries、text_roles。queries 每行至少有 query_id、text_index、cad_index。背景无 qrels，validation 的顺序为背景/拟合/选型，test 为背景/测试。准备器记录资产 SHA、候选轴 SHA、所有输入 SHA 及剔除原因。执行参数记录完整路径与 digest。读取器使用显式字段、shape、数量检查。

## 共享前向与正确性门

1. 复用 `load_frozen_text2cad`、既有候选向量验证、`assemble_batch`、`teacher_forced_logits`。固定 FP32/full272/b16，尾批按真实末批，trim 与文本缓存关闭。首选 share4 现成 `/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python`，已核它有 Transformers 4.53.1。实际运行用小批确认完整栈与模型。另一个 GLM 环境的 Transformers 已变为 5.23.0。无需新增数据下载。
2. 一对做一次前向。在历史 FP32 log_softmax 后取真实有效字段，将 logprob 转 float64 归约。六个互斥组的**固定列序**是 sketch_x、sketch_y、extrude_numeric、extrude_boolean、delimiter、terminal。x≥11 且 y≥11 的两头分别计 sketch_x/sketch_y。x≥11、y=0 计 extrude_numeric。x∈7..10、y=0 计 extrude_boolean。x∈2..6、y=0 计 delimiter。目标行 x=1、y=0 计 terminal，每个 CAD 恰一个且在最后有效行。行 (0,0) 跳过。先用历史结构校验，非法或未归组的非零字段拒绝。组和之和在绝对误差 1e−8 内等于历史式 SUM，计数之和精确等于 n_valid_total。
3. 每个文本写完整图库的 NPZ，scores 为 float64、(n_cads,7)，第 0 列原生 SUM，后六列是组和。候选有效字段计数单独存 (n_cads,6)。文件绑定文本索引、自身文本 SHA、候选轴/输入 SHA，进程原子发布。读出检查全部索引、有限性、计数和组和，拒绝错位、缺口、重复。候选原始 UID 和文本 qrels 分离。只对小型探针留逐位置诊断。模型使用非标签、合法旧协议输入。
4. **先验收真实 NPU 小批。** 复用已认证 E2 REPLAY 中两条非空文本各至少一个原满批与原真实尾批 14，保持成员顺序、形状，并核原始 replay SHA。用 `max(0.001,1e-6*abs(reference))` 比较 SUM，有效字段数精确相等，分组加总与有限性通过。记录实际运行栈、退出、耗时、设备。可先用同形合成检测逻辑。只有 Codex 验收小批，正式矩阵才准入。
5. 空闲时四 NPU 各负责独占的文本索引分片，按索引取模。每阶段单一控制器，保留每 worker 日志、实际退出、已发布清单和准确 pair budget。失败保留证据。SSH 由 Codex 单一调度，每次与上一连接结束间隔至少 65 秒。所有远端源码均经 Git 推送后检出。矩阵和正式测试后做独立 payload SHA 和排名审核。

## 每臂可执行定义与优先级

令 S 为原生 SUM，G_g 为某字段组和，n_g 为相应有效字段数，A_g=G_g/max(1,n_g)，缺席置零。B(c) 和 B_g(c) 是 48 个独立背景文本在该候选上的 S 和 G_g 均值。

| 顺位 | 臂 | 定义及对照 |
|---|---|---|
| 0 | 原生与简单基线 | SUM、MEAN=S/有效计数、有效计数升序。报解析随机 K/N。 |
| 1 | 背景残差 | S−λB。固定 λ=1，并在验证选 λ∈{0.25,0.5,0.75,1,1.25}。固定报告前 24 个背景估计对所选 λ 的敏感性。 |
| 2 | 对数混合背景 | 将 B 改为 logsumexp(48 个背景 S)−log(48)，同 λ 网格。作为经验背景。 |
| 3 | 固定字段权重 | 对六个 (G_g−B_g)/max(1,n_g) 加权，按固定列序比较均等、[1,1,1,2,2,1]、[1,1,1,3,3,1]。后两条加重结构和布尔。 |
| 4 | 非负线性学习排序 | 原始组均值 A 与背景校正组均值各拟合全局六个权重。在所有验证选型查询上比较。 |
| 5 | 文本条件调权 | 用臂 4 背景版的选型权重。casefold 后匹配 `\b(?:holes?|cuts?|slots?|subtract|subtracts|subtracted|subtracting)\b` 时 bool 权重×2。命中 ASCII `[0-9]` 时前三组权重×2。分别报告仅词门、仅数字门、双门。 |
| 6 | 负例来源 | 背景组版在等量随机负例上重拟合，与同特征困难负例版对照。 |
| 7 | 遮组、融合及机制 | 报每组正负分差的中位数、严格胜率与同分数。对已选背景组模型逐组去掉一项。固定取验证 R@20 最佳的两个不同家族，用平均倒数排名 1/(60+rank) 融合。 |

拟合时同文所有登记 UID 都是正例，从负例池移除。每个正例取同一查询 SUM 最靠前 16 个剩余 CAD，再取有效字段数最近的 16 个剩余 CAD，同分按真实索引升序。随机对照 seed 20261118 取等数。每个正例负例数一致，按原查询加权。这些近负例按登记 UID 定义，可能仍符合粗略描述。不得向评分器传标签。

对拟合正负特征差 D 用各维 RMS 标准化，常零维取尺度 1。仅在拟合集计算 RMS，选型和测试直接使用冻结值。拟合 mean(logaddexp(0,−Dw))+α||w||²，w≥0，α∈{0.01,0.1,1}。投影梯度，步长用逻辑损失曲率上界，固定迭代数并记录收敛。选型依次比较 validation R@20、R@1、R@5、较小 MedR、固定枚举顺序。每家族保留一个冻结代表，所有固定基线与代表在同一测试条件下评价。掩码和融合由验证定义。验证选型和测试各用 5,000 次配对文本组 bootstrap，seed 20261118，恢复原查询等权，选型的统计量仅作探索性描述。输出全部逐查询 rank、命中数、分层、区间、权重和拟合输入指纹。候选同分按 cad_index 升序。

## 输入约定

准备器生成 `<prepared>/{validation,test}/inputs.json` 与同级 `eval.json`。前者为 `schema: text2cad-accuracy-inputs-v1`、`axis: validation|test`、`texts` 字符串列表、`members` 字符串列表。后者为 `schema: text2cad-accuracy-eval-v1`、`axis`、`inputs_sha256`、`cad_uids` 列表、`text_roles` 列表（background/fit/select/test）、`queries` 对象列表（query_id/text_index/cad_index）。每轴的评分目录内 `counts.npy` 是 shard 0 独占发布的 int64 候选组计数，其它 shard 可保留自己的计数收据。每个 `chunks/t{index:05d}.npz` 含 `scores`、`text_index`、`text_sha256`、`inputs_sha256`、`counts_sha256`，除 scores 外均为零维标量。counts SHA 使用 int64 小端 C-order 数据字节。阶段前缀包含全部背景文本。

| 资产 | SHA-256 |
|---|---|
| text2cad_v1.0.csv | `32427ca7169f8b93411b64a025c387cdb0b55d9b9ce0e9cf9755ac8a3528f4a3` |
| train_test_val.json | `4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b` |
| cad_seq.zip | `9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa` |
| Text2CAD_1.0.pth | `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598` |
| G5 query.jsonl | `a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd` |
| G5 gallery.jsonl | `09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30` |
| G5 qrels.jsonl | `f0677928ae2412ca4bcd8952c95d89f0568c5bb214c7bb53e82e0059555c87d6` |
| G5 cad_store.jsonl | `2f441110f013bb614b0bd2fdb7a3b49569a231bbf3f8eaa2c03ac1d725cb1380` |
| E2 replay_gate_record.json | `59f3e8c1469d5f2056dd0579de687f328b8f7b37357a2f3e3276773e27b5bcc2` |

## 运行顺序与验收收据

采用新增 `experiments/scripts/run_text2cad_accuracy_night.py` 负责 prepare/replay/score，`experiments/scripts/eval_text2cad_accuracy_night.py` 负责 fit/evaluate。实现须交付准确 argv 与针对性回归。

| 阶段 | 操作者与交付 | 进下一阶段的证据 |
|---|---|---|
| 方案 | GLM‑5.3 max 审准确性、可行性、共享前向和冻结边界 | Codex 逐条裁决、修订并记录 |
| 实现 | GLM 本地实现和 CPU 测试 | Codex 核对 diff、数据身份、掩码、拟合、预算和 CLI 后批准完整 SHA |
| 小批 | 远端零前向准备和固定 NPU Replay | 真退出码、旧分逐项通过、数值有限、设备/吞吐可用。Codex 接受后开始正式评分 |
| 验证 | 四卡按共享矩阵打分，CPU 拟合选型 | 覆盖、分组算术、配对时序、参数文件原始记录。GLM 报告，Codex 核验冻结 |
| 测试 | 同库全量候选，加载冻结方案，评价 G5 固定查询 | 逐查询名次、实际重排和差分、配对 CI、配置与日志逐 SHA。GLM 报告，Codex 独立复核 |

最大主评分量 560×8,925 + 1,072×8,046 = **13,623,312 对**，另有 60 对回放。按 share4 E2 的约 150 pair/s/卡，4 卡约 6.3 小时。验证阶段完毕后用实测四卡速度估算测试，优先在次日上午交付前完成。在写入测试评分前，在 1,024、768、512、256、128 的固定前缀中选有足够时间的最大值，保留约 20% 吞吐余量和至少 25 分钟作评价与复核。记录时间估计和选择。任何超时或资产、数值、运行异常保留证据。测试结果不回流选型。

## 方案验收

GLM‑5.3 `[1m]` 以 max 推理完成只读审查，实际退出 0，回执 SHA `25fff05d91b0179c9d0e8a87b1355ad74a790ede7c3b6e0502632cd98c4e6816`。采纳 share4 吞吐、固定文件指纹、UID 交集、字段穷尽、词门、RMS 冻结、选型区间及文本碰撞八条意见。历史测试 UID 关联合同沿用 G5，验证 L0 资产以本计划的 v1.0 CSV 为准。Codex 本地预检看到验证图库 8,925 UID、非空 L0 查询 8,829 条、独特文本 4,023 组，排除与 G5 逐字重合后 3,143 组，满足本轮取样。
