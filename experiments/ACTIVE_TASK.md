# 当前实验任务：原版 Text2CAD M1c 本地评分器与测速器实现

- `state`: `implementation`（仅本地源码/配置/测试、资产只读或传至 checkout 外；**真实模型前向预算 0**）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`
- `phase`: 用户已选原版 Text2CAD；先审官方两头训练目标、checkpoint/BERT、CAD 向量 mask 和 910 适配，再在本地实现 scorer、pair 构建器、固定 batch 基准与 CPU 测试。CADmium 四种公开 adapter 的 LoRA-B 全零，原 G1 作废；不得复用其代码授权或评分
- `approved_sha`: 无；须 Codex 审查 M1c 实现提交后填写完整 SHA
- `gpu_budget`: 0 卡小时；`max_scored_pairs`: 0；当前禁止 NPU/CPU 真实模型前向、批量评分和正式检索。无前向 checkpoint 结构加载诊断须保存证据，且不得借此运行打分
- `data`: `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` 官方 v1.0；复用 M0 冻结的 8,046 CAD 图库、7,934 非空 L0 查询、3,644 唯一 `strip()` 文本和独立 qrels。官方 `cad_seq.zip` 是本臂唯一候选原生表示；评分输入不得含原 CSV、caption metadata、UID 正例映射或 qrels
- `model`: 原作者 `Text2CAD_1.0.pth` SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，固定 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`；官方源码本地镜像 SHA `669a81472ede2269837c22a4d00070eaac360c82`。代码许可 CC BY-NC-SA 4.0，若改编纳入仓库须保留许可/出处
- `server`: `yunnan-share2`，仅两张 Ascend 910 可供后续获批验证；checkout `/home/ma-user/work/CADRetrieval`。此阶段仅 SSH 只读环境/资产核验或把数据权重复制到 `/home/ma-user/work/CADRetrieval_runtime/`；远端源码树不编辑，NPU 不运行。`cvteam14` 仅用于共享 OBS 资产核验，不计计算资源
- `allowed_network`: 固定官方仓库/Revision 的源码与资产、只读元数据。大权重与数据由 Codex 当前选择性准备和校验；不要下载图像/深度图全库或其它模型
- `required_output`: 本地实现与 CPU 测试、逐字段评分定义、资产/权重键/许可核验、固定 pair 清单与不含 qrels 的评分入口、batch 测速器、精确远端命令，以及 `experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_IMPLEMENTATION.md`，末尾写 `review_required`。Claude Code/Kimi 不能修改本文件、计划或 Codex 审查文件
- `stop_conditions`: checkpoint 非 BERT 键无法严格对应、官方数据 CAD 向量/有效 mask 不自洽、文本或 CAD 被静默截断、qrels 泄漏、910 移植改变数学、资产 SHA/许可不明、远端 Git 不干净；在本地交接阻塞并停止受影响步骤

原版 Text2CAD 的训练 loss 使用 `label_smoothing=0.1`；只允许从冻结模型原始两路 logits 构造清楚命名的条件排序分数，不把训练 loss 或双头分数称为 CAD 几何的精确 NLL。此前 Qwen Text-to-CadQuery 与 A test 有 4,581 UID 训练交叉，不能作为 A 的无泄漏主结果。

用户要求完整 A 的全查询×全图库精确评分。先完成本地实现、Codex 审码和小规模真实正确性/吞吐验证；基于两张卡代表性实测，计冷启动/I/O/失败并乘至少 1.30 余量，保守 ETA ≤7 天才可另行批准全量，否则先与用户讨论。任何旧 P0/P1/CADmium 授权均不适用于本臂。

所有项目代码、配置、文档只在本地修改并 Git 推送；服务器只 fetch/detach 到 Codex 审核的完整 SHA 运行。结果和缓存留在 checkout 外，重要原始结果归档共享 OBS。
