# Text2CAD G2 生产批测速与 ETA 修复实施计划

> 执行者为本地 Claude Code / Kimi K3，按任务顺序实施和自检；Codex做独立审核。使用可用的 executing-plans 流程，项目既有分工优先，不自行派发其它执行者。本文件仅授权实现与CPU假数据测试，不授权任何真实模型/NPU前向。

**Goal：** 修正可信双卡全量ETA的协议，并提供真正16/32批及优化组合的等价验证入口。

**Architecture：** 保留原版模型、score和既有G1输入；新增CPU协议模块按正式矩阵的真实批界生成测速清单。测速记录真实批大小和逐rep时间窗口，ETA按两张卡的固定分片取最大完成时间。优化仍关闭，后续证据与配置须另审。

**Tech Stack：** 当前Python/torch/torch_npu环境不变，CPU测试用本地 `D:\miniconda\python.exe`。不安装新依赖。参考 `CLAUDE.md`、M1c总合同、两卡路线、G1 batch Codex审查。

## 全局约束

- 当前仅implementation，真实模型评分、BERT模型编码、NPU预算均0；不得沿用已耗尽的f0d3485账本。不要在本轮占用NPU。
- 源码、配置与文档只在本地编辑，经Git提交；服务器不修改源码。当前无需部署或切换服务器HEAD。
- 固定8,046 CAD、3,644文本、29,319,624对；qrels/UID评价映射/CSV不进入selector/worker。
- 模型、双头评分、shift/mask、FP32和固定容差不变；不要为速度改注意力计算或做BF16。
- 原G1 selection配置及原报告不修改；新G2配置在后续CPU资产预处理和Codex审核后生成，此阶段不提交空配置或虚构哈希。
- 正式矩阵仍仅reference（float32/full-length/uncached）；新增优化入口只能用于独立G1等价门，不能绕过现有production拒绝逻辑。
- 不修改CLAUDE、ACTIVE_TASK、plans、reviews；只改src/scripts/tests和本轮submission。不得自行进入G2或全量。

## 文件责任

| 文件 | 工作 |
|---|---|
| 新增 `experiments/src/idea01/text2cad_bench_protocol.py` | 纯CPU生产批清单、真实分片权重、实际窗口核验、较慢卡ETA；可放小型同步barrier实现，必要时单独模块 |
| 修改 `text2cad_speed_bench.py`、`scripts/run_text2cad_speed_bench.py` | 新selector/CPU token长度审计入口、G1扩展、生产批测速与协议集成；旧G1继续可复现 |
| 必要时修改 `text2cad_frozen_runner.py` | 仅增加cache统计/公开共享纯函数，不能改变数学或生产授权范围 |
| 新增/修改 `tests/test_text2cad_bench_protocol.py`、`test_text2cad_speed_bench.py` | 症状回归、预算与真实batch/并发/分片一致性；不执行真实权重 |
| `submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_G2_PROTOCOL_IMPLEMENTATION_2026-10-01.md` | 实现/红绿测试/差异/限制/下一阶段命令与完整提交SHA，末尾review_required |

## Review Focus

1. nominal32必须真实满32，不能由20对冒充；末批也要单独建模。
2. 不能把冷启动重叠当测速并发，拒绝无真实窗口或单卡冒充双卡。
3. 卡速/文本分布不平衡时必须以较慢分片为墙钟。
4. 生产混长批、token padding、缓存命中须有相同构造；优化等价通过不等于已生产批准。
5. peer退出、OOM、NaN、预算/输入不符时保存失败证据并停，不混旧schema、重试或修改容差。

## Task 1：CPU生成真实生产批与精确分片权重

**Interfaces：** `build_g2_selection(text_records, cad_length_records, text_token_lengths: dict[int,int], *, batch_tiers=(1,2,4,8,16,32), texts_per_chunk=1, n_shards=2) -> dict`。生产尺寸在CLI固定，纯算法测试可用小尺寸fixture。

- [ ] 先写回归测试：37个CAD、batch32产生真实32满批和5尾批；两个文本分片互斥且完整；跨所有cell的pair计数精确等于query×CAD，尾批不遗漏、不补重复CAD。
- [ ] 实现与正式 `PairSpace.shard_text_range(2,k)` 相同的文本分界。合同固定 `texts_per_chunk=1`，每文本的CAD仍按原cad_index顺序，每batch是相邻CAD索引切片，不能把长度相近但不相邻的CAD拼为生产批。
- [ ] 文本按实际固定BERT tokenizer的token长度（含special tokens、无截断）排名分三层；full batch按批最大真实CAD长度排名分十层，tail另列。逐tier、逐shard、逐cell保存实际batch数、pair数、实际batch尺寸、batch最大长度；完整矩阵权重可隐式算，无需写2,932万行。
- [ ] 每个非空cell取其对应shard内2个覆盖token长度端点的文本、2个覆盖该batch层端点的真实相邻CAD批，去重不足就全部取，记录shortfall；每一档必须实测足量真实满batch。两卡用不同文本分片的样本，对数计实际forward，允许不同tier重复相同pair但要如实计预算。
- [ ] 提供纯CPU `audit-text-lengths` 入口：仅加载固定BERT tokenizer（不能加载BERT模型），记录全部text_index/token长度、输入文本SHA和五文件资产SHA；输出checkout外。提供本地纯CPU `select-g2` 入口读取该派生数据生成新配置，LF输出、拒绝覆盖、绑定输入SHA/规则/schema。**本轮只实现与假数据测试，不在服务器运行；真实token长度数据与新配置随后由Codex逐步核验。**
- [ ] 保持原spot/ten_pair不变；新扩展G1固定64对=8个文本长度分位×8个CAD长度分位（0/10/25/50/75/90/99/100），包含真实最长文本和最长CAD，不读qrels。新schema与旧schema不互认用于G2 ETA。

## Task 2：真实批大小、预算与实际并发测量

**Interfaces：** 沿用 `score_batch_fn(pairs)->(scores,timings)`，新增驱动消费Task1的**已分好的批**，不能重新flatten成另一种组批。预算计划必须与驱动同源枚举，计warmup/reps所有pair。

- [ ] 回归测试先复现旧20对假32、名义窗口重叠实际rep不重叠；修复后拒绝或生成真实满批，不接受旧数据。
- [ ] 每档/每cell至少2次warmup，至少3次计时rep；记录每batch的真实尺寸、pair顺序、最大CAD长度、实际text token/padding宽度，记录epoch start/end、preprocess/forward-score/write/E2E、峰值HBM、失败/拒绝、缓存统计（若有）。真实模型CLI默认仍reference。
- [ ] 同一host两进程用checkout外的新 `--sync-dir` 与 `--sync-group`，按tier/cell/rep同步；barrier绑定代码/模型/清单/选择/实际profile，拒绝旧文件/不同绑定/同设备。timeout=180秒；等待计卡时，不混进纯测量区间；peer错误/OOM标记使对端停止，保存已有证据，不删除旧marker或自动重跑。
- [ ] 双卡ETA对使用的每个cell/tier/rep核实际窗口重叠至少较短窗口的80%；反向、缺失、非有限或实际错开一律拒绝。不使用原全局window作为并发凭据。
- [ ] `--allow-large-batch`只允许明确请求16/32，不自动追加未请求档。所有实际评分pair在forward前扣共享账本，模型加载和barrier时间计卡时。

## Task 3：按固定生产分片取较慢卡ETA

**Interfaces：** `estimate_sharded_eta(selection, results_by_device, tier, *, overhead_seconds=None, safety_factor=1.30) -> dict`，保留旧估算器供旧CPU测试/历史解释，但新G2入口不调用旧池化公式。

- [ ] 红绿测试：卡0速40、卡1速20、两卡负载各半时，必须是慢卡时间（不能按40+20计算）；文本层分布不等时以对应shard×cell实际权重算，不理想2倍。
- [ ] 对每张卡计算 `sum(cell真实batch数 × 该cell实测每批E2E秒) + 该卡冷启动`；实测每批成本用完整rep的E2E/真实batch数，保守值取3次rep中最大值。全量ETA取两张卡完成时间max，加适用额外固定开销，再乘>=1.30。
- [ ] `overhead_seconds`未给时保留全部实测冷启动；显式值必须有限且>=两卡cold_start之和，额外部分作为两卡之后的附加墙钟开销；不能通过0绕过。拒绝NaN/Inf、负数、低安全系数、缺cell、错代码/模型/配置、重复文件/invocation/device或未完成档。
- [ ] 明列每卡负载/计算/冷启动/附加开销/保守ETA。写盘和ledger若已在E2E，不重复外加；其它I/O/故障余量说明实际来源。完整pair数严格29,319,624；七天门仍604,800秒。

## Task 4：真正大批及优化组合的G1入口

**Interfaces：** 新 `profile-consistency` mode，固定Task1的64对，使用原 `score_pairs_detailed/compare_gate_results` 和原容差。

- [ ] 假数据测试确认实际64对参考batch1；基线batch16/32各64；trim32/cache32各64；trim+cache组合batch1/2/4/8/16/32各64。总共**704评分对**，预扣/计时与实际一致，不让CLI更改固定候选或容差。
- [ ] 单独报告每个variant及真实batch大小、trim/cache状态、逐对两头/总分和有效计数、最大绝对差及按固定容差归一后的最大差；有限性强检。保存各variant有效行logits张量（checkout外，按需CPU，weights_only可读）供Codex独立复算，避免只交布尔值。
- [ ] 任一比较失败写原始证据、非零退出并停止。此入口不批准production优化；throughput和matrix仍拒绝非reference，之后由Codex根据结果提交明确配置再扩授权。

## Task 5：本地验证与交付

- [ ] 每项症状先有失败测试，再做最小修复和重跑；保留红绿日志到checkout外。
- [ ] 运行协议/速度/runner受影响测试；原G1输入、分数定义和资产指纹不得变化。不得用真实模型跑测试。
- [ ] 检查diff/新增输出路径/生产拒绝门/预算失败路径，写IMPLEMENTATION含准确CPU预处理、704对G1与后续G2示例命令（都注明未授权）。说明剩余问题，不自行推算实测吞吐。
- [ ] 仅提交本任务实现/测试/轻量submission，记录完整SHA；不带数据/权重/缓存，不改Codex文件。完成后停止，等待Codex审码；Codex随后负责push/逐步CPU资产核验/完整SHA授权。用户已授权验证性实验，常规通过自动推进，数值/资产/资源或科学口径异常则停下交流。
