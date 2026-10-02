# Text2CAD G4：Codex 独立启动审核

2026-10-02。结论：**正式全量已启动，运行身份、预算、两卡映射和两个首 chunk 核验通过；全量结果及检索指标尚未完成。** 本审核不授权新 invocation、评分重试或源码改动。

## 1. 实际运行与交接

- run：`idea01_text2cad_g4_full_20261002`；完整 3644×8046=29319624 对，两片各 1822 个 query chunk。
- 实际运行 SHA：`56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；授权来源 `origin/master` 的 `67e89a20eb6fc3b35d27c3ba2363da4311d9113c`，ACTIVE blob SHA256 `53b305c2b4407b16b09b85f769e218b50fdac983a6b5a7d34c204ce170de8ef9`。
- Kimi LAUNCH 首次报告提交 `9dd46182d7a1cd6a0805fe6a25361ea58f0f82c7`，回填提交 `c9e499b627cb1c1dacf010a6f9f4251240af1e1a`。相对授权提交仅新增该报告，无源码、配置或环境改动。CLI session46405 已 exit0，stream result success/is_error=false；不再次派发启动。
- controller PID **3682036**，实际 PPID1。scorer0 PID **3682124**、invocation `20261002T131158Z-c3812433`；scorer1 PID **3682194**、invocation `20261002T131159Z-6ce644de`。ps 与 npu-smi 对应 Chip0/1，分别是 scorer 自有 PGID/SID。
- 共同起点 `2026-10-02T13:11:51.702770Z`（北京时间21:11:51）；共同截止 `2026-10-09T13:11:51.702770Z`，差值604800秒。TERM deadline−90，KILL deadline−30，均从真实 common_start 事件复核。

## 2. 独立只读证据

Codex 于 **13:18:09Z** 直接读取服务器，而非只接受报告：

- HEAD 为批准完整SHA、detached、porcelain空；未 fetch/detach、未改源码。
- controller/two scorers存活；两张Ascend健康OK，实际NPU进程仅上述两scorer；runtime剩余约4.30TB。controller外层日志无输出，两卡stderr仅既有torch_npu owner警告，无FATAL/Traceback。
- 两卡各有10个原子发布chunk。完整读取并审核的是两卡各首个chunk，共**16092条pair**；未据此声称20个chunk或全矩阵全部审核。
- 共享账本 `runtime/budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json` 恰两invocation/cards_used1，cap29319624/1209600卡秒，合计已扣167160对。心跳推进、两invocation尚未finalized，无退出/summary。
- settled `card_seconds_spent=0` 是运行中账本的结算语义。独立按 snapshot epoch−两 started_epoch 加 settled 复算 **741.277658700943卡秒**，在预算内；监控必须同时读pending，不能把0当作没占卡。
- controller实际argv与ps一致：完整gallery/full分片、FP32、full272、batch16尾14、无trim/cache/probe/token-audit/resume；完整plan各14659812对/916466批。

首两个chunk经服务器源SHA与本地逐SHA一致后，独立复算稠密索引、8046完整CAD、绑定、六个score有限性、双头sum、有效长度与三项mean，共**96593项检查通过**。长度来自本地固定 `cad_length_index.jsonl`，重新核其SHA `daac66a102e34fe45c9cb474922076b0d62345ab0c9796f917a0f8085c6ea0c3`。

| shard | 首chunk SHA256 | 已审pair |
| --- | --- | --- |
| 0 | `dec0ddd97d79251c4dd3e1f0b24d55757779a0cc66b660f33813d46f88fcd882` | 8046 |
| 1 | `b32ed9eacbdd4d96481e23b75b7a05d4f48e5badc5ebdd552416918dd1cafea5` | 8046 |

## 3. 计时与报告的两处限定

1. 上轮执行文档沿用的“首chunk/timing”表述不精确。`text2cad_frozen_runner.main` 在 **probe_selection不为None** 时才调用 `execute_probe_query` 并生成 `timing_sXX.jsonl`；正式full路径没有该sidecar，也没有G3特有的cold_start/peak字段。本次首chunk完成节奏只能据mtime观察，不能冒称测得逐query窗口、精确冷启动或并发比例。正式总耗时将核final ledger、controller/scorer实际退出；持续监控以真实文件mtime/心跳判断推进。修正文档，不修改健康作业或评分代码、不重跑。
2. Kimi报告把账本字段归于controller维护，责任表述有误：实际 `SharedBudgetLedger` 由scorer记录/结算，controller负责共同截止与真实进程退出。当前运行语义及预算数字不受此文字更正影响。完整资产strict-load/preload报告在最终summary中才发布；本次根据批准代码准入及首chunk绑定核运行身份，最终仍需审核完整summary，不以首chunk替代资产结项。

## 4. 原始证据与下一步

本地权威证据为 checkout外 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_actual_launch_codex_20261002/`：9个文件（snapshot/SSH状态、两首chunk、源SHA、运输记录及独立核验JSON），另有LF SHA256SUMS，manifest SHA256 **`26a0982224b49c4a59c38c4273f902d5aebf0b63b8e3934c2d75e0b8860d9787`**。共享OBS `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/m1c_g4_actual_launch_codex_20261002/` 同9项逐SHA通过，真实解包及sha256sum退出0；运输原始 stdout/stderr/exit 在本地 `g4_actual_launch_archive_transport_20261002/`，不以管道尾命令掩盖失败。

早先选用的 `g4_launch_codex_20261002/` 已有本地CPU故障fixture；第一次归档误含这些旧fixture，232文件解包/核SHA命令120秒超时。该混合目录与 `m1c_g4_codex_launch_check_20261002` **不是本次正式证据集**，未宣称OBS核验成功；新目录只选上述9个已知运行证据。没有重新评分或修改实验服务器源码。

每30分钟只读监控，不重新启动/fetch/detach。计划保守时长仍取 max(G2,G3)=**38.28394小时**；早期局部节奏不代替全量完成证明。待两片各1822完整chunk、两exit0/finalized、全29319624对/零拒绝后，收集并独立审核全矩阵、两summary、final ledger、实际退出和全量本地/OBS SHA，再激活CPU指标评价。UID指标、语义结论、纯MLE/几何NLL均不在此次启动审核结论内。
