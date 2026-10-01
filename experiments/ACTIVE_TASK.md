# 当前实验任务：原版 Text2CAD M1c CPU资产预处理与扩展G1准备

- `state`: `reviewed`（R3代码已审；仅CPU tokenizer派生资产与本地配置冻结；真实模型/NPU预算0）
- `task_id`: `idea01_text2cad_benchmark_20260928`
- `plan`: `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- `active_subplan`: `experiments/plans/IDEA01_M1C_G1_PROFILE_2026-10-01.md`（上位子合同仍为 M1c 2026-09-30）
- `phase`: G1 spot与十对三门已过；G2协议R1/R2/R3已修并审查，最新CodexCPU108项/25子测试通过。只运行零前向tokenizer资产审计、取回SHA核验、本地生成并冻结G2配置；704对G1及G2需另写完整SHA和预算
- `reviewed_cpu_sha`: `598dd40b0cf6f8ce87003225c0ba3ee434abb450`（仅CPU资产入口审定；服务器只Git检出此SHA运行，不授权真实前向）
- `approved_sha`: 无（上一轮完整SHA及账本保留，已不授权新真实前向）
- `gpu_budget`: 0 卡小时；`max_scored_pairs`: 0；BERT模型编码、NPU、真实checkpoint forward、G2和全量均未授权。此前两轮spot/batch账本不删除或改额度；新实现须重新审查、Git推送和完整SHA逐门授权
- `data`: `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` 官方 v1.0；复用 M0 冻结的 8,046 CAD 图库、7,934 非空 L0 查询、3,644 唯一 `strip()` 文本和独立 qrels。官方 `cad_seq.zip` 是本臂唯一候选原生表示；评分输入不得含原 CSV、caption metadata、UID 正例映射或 qrels
- `model`: 原作者 `Text2CAD_1.0.pth` SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，固定 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`；官方源码本地镜像 SHA `669a81472ede2269837c22a4d00070eaac360c82`。代码许可 CC BY-NC-SA 4.0，若改编纳入仓库须保留许可/出处
- `server`: `yunnan-share2`（本阶段不占卡；允许Git fetch/detach上述已审CPU SHA并运行tokenizer审计）；checkout `/home/ma-user/work/CADRetrieval`，源码树保持干净。资产及日志在 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/`；`cvteam14` 仅用于共享 OBS 归档，不计计算资源
- `allowed_network`: 固定官方仓库/Revision 的源码与资产、只读元数据。大权重与数据由 Codex 当前选择性准备和校验；不要下载图像/深度图全库或其它模型
- `required_output`: checkout外真实token长度审计及SHA、本地G2配置候选、`experiments/assets/TEXT2CAD_G2_PREPROCESS_2026-10-01.md`；R1/R2/R3报告与CPU证据保留。仅Codex修改本文件/计划/审核；未获704门授权前不运行真实前向
- `stop_conditions`: checkpoint 非 BERT 键无法严格对应、官方数据 CAD 向量/有效 mask 不自洽、文本或 CAD 被静默截断、qrels 泄漏、910 移植改变数学、资产 SHA/许可不明、远端 Git 不干净；在本地交接阻塞并停止受影响步骤

原版 Text2CAD 的训练 loss 使用 `label_smoothing=0.1`；只允许从冻结模型原始两路 logits 构造清楚命名的条件排序分数，不把训练 loss 或双头分数称为 CAD 几何的精确 NLL。此前 Qwen Text-to-CadQuery 与 A test 有 4,581 UID 训练交叉，不能作为 A 的无泄漏主结果。

用户要求完整 A 的全查询×全图库精确评分。先完成本地实现、Codex 审码和小规模真实正确性/吞吐验证；基于两张卡代表性实测，计冷启动/I/O/失败并乘至少 1.30 余量，保守 ETA ≤7 天才可另行批准全量，否则先与用户讨论。任何旧 P0/P1/CADmium 授权均不适用于本臂。

所有项目代码、配置、文档只在本地修改并 Git 推送；服务器只 fetch/detach 到 Codex 审核的完整 SHA 运行。结果和缓存留在 checkout 外，重要原始结果归档共享 OBS。
