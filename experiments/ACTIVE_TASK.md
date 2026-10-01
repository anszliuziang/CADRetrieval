# 当前实验任务：原版 Text2CAD M1c 固定64对扩展G1

- `state`: `pilot_authorized`（仅G1 profile-consistency单次调用；固定704对、单卡，不授权G2/全量）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G1_PROFILE_2026-10-01.md`（上位子合同仍为 M1c 2026-09-30）
- `phase`: R3代码与真实G2配置已审，独立CPU108项/25子测试及真实清单1187项通过；允许profile-consistency：64对reference_b1、batch16/32、trim32/cache32、trim+cache批1/2/4/8/16/32，共704评分对。固定容差，失败保留原始证据并停止，不自动重试或放大
- `reviewed_cpu_sha`: `598dd40b0cf6f8ce87003225c0ba3ee434abb450`（仅CPU资产入口审定；服务器只Git检出此SHA运行，不授权真实前向）
- `approved_sha`: `a2219e25b7f018aa12e820616466672fd8d70729`（代码、真实配置已审；远端仅干净detached此完整SHA运行）
- `gpu_budget`: 1 卡小时；`max_scored_pairs`: 704；仅逻辑npu:0、cards_used=1、单次固定profile-consistency调用；不批准额外warmup、另跑batch档、第二卡、G2或全量。新的per-SHA共享账本不改旧spot/batch账本，不自动重试
- `run_id`: `idea01_text2cad_g1_profile_20261001`；out `/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_profile_20261001/`，不覆盖已有结果；token审计SHA `9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8`；配置SHA `fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce`
- `data`: `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` 官方 v1.0；复用 M0 冻结的 8,046 CAD 图库、7,934 非空 L0 查询、3,644 唯一 `strip()` 文本和独立 qrels。官方 `cad_seq.zip` 是本臂唯一候选原生表示；评分输入不得含原 CSV、caption metadata、UID 正例映射或 qrels
- `model`: 原作者 `Text2CAD_1.0.pth` SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，固定 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`；官方源码本地镜像 SHA `669a81472ede2269837c22a4d00070eaac360c82`。代码许可 CC BY-NC-SA 4.0，若改编纳入仓库须保留许可/出处
- `server`: `yunnan-share2`（仅npu:0进行704门，启动前确认两卡健康且目标卡无其它进程）；checkout `/home/ma-user/work/CADRetrieval`，源码树保持干净。资产及日志在 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/`；`cvteam14` 仅用于共享 OBS 归档，不计计算资源
- `allowed_network`: 固定官方仓库/Revision 的源码与资产、只读元数据。大权重与数据由 Codex 当前选择性准备和校验；不要下载图像/深度图全库或其它模型
- `required_output`: 本地原始镜像与共享OBS归档、11份有效行logits、逐对两头分数/计数、失败证据、环境/完整SHA/命令、真实batch/trim/cache、预算账本和SHA256SUMS；新submission `TEXT2CAD_G1_PROFILE_RESULTS_2026-10-01.md`。仅Codex修改本文件/计划/审核；Kimi不改源码/配置/旧报告，不自行进入下一门
- `stop_conditions`: checkpoint 非 BERT 键无法严格对应、官方数据 CAD 向量/有效 mask 不自洽、文本或 CAD 被静默截断、qrels 泄漏、910 移植改变数学、资产 SHA/许可不明、远端 Git 不干净；在本地交接阻塞并停止受影响步骤

原版 Text2CAD 的训练 loss 使用 `label_smoothing=0.1`；只允许从冻结模型原始两路 logits 构造清楚命名的条件排序分数，不把训练 loss 或双头分数称为 CAD 几何的精确 NLL。此前 Qwen Text-to-CadQuery 与 A test 有 4,581 UID 训练交叉，不能作为 A 的无泄漏主结果。

用户要求完整 A 的全查询×全图库精确评分。先完成本地实现、Codex 审码和小规模真实正确性/吞吐验证；基于两张卡代表性实测，计冷启动/I/O/失败并乘至少 1.30 余量，保守 ETA ≤7 天才可另行批准全量，否则先与用户讨论。任何旧 P0/P1/CADmium 授权均不适用于本臂。

所有项目代码、配置、文档只在本地修改并 Git 推送；服务器只 fetch/detach 到 Codex 审核的完整 SHA 运行。结果和缓存留在 checkout 外，重要原始结果归档共享 OBS。
