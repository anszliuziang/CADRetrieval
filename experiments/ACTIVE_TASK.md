# 当前实验任务：原版 Text2CAD M1c G1 批一致性与优化等价门

- `state`: `pilot_authorized`（仅 G1 固定十对的 batch/trim/cache 三门；禁止扩大运行）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G1_BATCH_2026-10-01.md`（上位子合同仍为 M1c 2026-09-30）
- `phase`: 单对 spot 已通过并由 Codex 独立复算/核 OBS。现仅执行固定十对的 batch-consistency（1逐对参考与2/4/8/10档，共50对）、trim-check20对、cache-check20对。必须先 batch 通过再测优化；预设容差不改。G2、CPU真实参考与全量检索未授权
- `approved_sha`: `f0d348568f60ac2956deeb058aca7f4873afad79`
- `gpu_budget`: 1 卡小时；`max_scored_pairs`: 90；单进程 `cards_used=1`，三门共用本 SHA 的预算账本；失败也计预算，不自动重试真实前向。此前spot 1对及33.64卡秒保留在旧SHA账本，不删除或改额度
- `data`: `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` 官方 v1.0；复用 M0 冻结的 8,046 CAD 图库、7,934 非空 L0 查询、3,644 唯一 `strip()` 文本和独立 qrels。官方 `cad_seq.zip` 是本臂唯一候选原生表示；评分输入不得含原 CSV、caption metadata、UID 正例映射或 qrels
- `model`: 原作者 `Text2CAD_1.0.pth` SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，固定 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`；官方源码本地镜像 SHA `669a81472ede2269837c22a4d00070eaac360c82`。代码许可 CC BY-NC-SA 4.0，若改编纳入仓库须保留许可/出处
- `server`: `yunnan-share2`，本阶段仅 npu:0；checkout `/home/ma-user/work/CADRetrieval`，须保持干净 detached approved SHA。资产及日志在 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/`；`cvteam14` 仅用于共享 OBS 归档，不计计算资源
- `allowed_network`: 固定官方仓库/Revision 的源码与资产、只读元数据。大权重与数据由 Codex 当前选择性准备和校验；不要下载图像/深度图全库或其它模型
- `required_output`: `experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_G1_BATCH_RESULTS_2026-10-01.md`，三门原始 JSON/stdout/stderr/退出码/预算 snapshot/环境/命令/结果 SHA，在本地 runs 与 OBS 各保存可审计副本；末尾 results_submitted。Claude Code/Kimi 不能修改本文件、源码、配置、计划或 Codex 审查文件
- `stop_conditions`: checkpoint 非 BERT 键无法严格对应、官方数据 CAD 向量/有效 mask 不自洽、文本或 CAD 被静默截断、qrels 泄漏、910 移植改变数学、资产 SHA/许可不明、远端 Git 不干净；在本地交接阻塞并停止受影响步骤

原版 Text2CAD 的训练 loss 使用 `label_smoothing=0.1`；只允许从冻结模型原始两路 logits 构造清楚命名的条件排序分数，不把训练 loss 或双头分数称为 CAD 几何的精确 NLL。此前 Qwen Text-to-CadQuery 与 A test 有 4,581 UID 训练交叉，不能作为 A 的无泄漏主结果。

用户要求完整 A 的全查询×全图库精确评分。先完成本地实现、Codex 审码和小规模真实正确性/吞吐验证；基于两张卡代表性实测，计冷启动/I/O/失败并乘至少 1.30 余量，保守 ETA ≤7 天才可另行批准全量，否则先与用户讨论。任何旧 P0/P1/CADmium 授权均不适用于本臂。

所有项目代码、配置、文档只在本地修改并 Git 推送；服务器只 fetch/detach 到 Codex 审核的完整 SHA 运行。结果和缓存留在 checkout 外，重要原始结果归档共享 OBS。
