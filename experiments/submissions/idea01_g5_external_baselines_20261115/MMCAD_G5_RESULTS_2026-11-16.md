# MM-CAD G5 全量评测结果（2026-11-16）

计划 §3 顺序 2 的正式交付。用户授权序列（CLIP → MM-CAD → OpenShape）中的第二项。**零 NPU；share4 `cadretrieval-glm-torch271` 环境 CPU。**

## 1. 执行记录

| 项 | 值 |
|---|---|
| 命令 | share4 上 `run_mmcad_g5_20261116.py`（完整参数见 `runtime/mmcad_g5_run10.log` 启动段与本地脚本） |
| 实际退出 | 最终收尾运行 exit 0（complete），`final_summary.json` 落盘 |
| 总耗时 | 嵌入段 ~1.2h（8,032 件）+ 文本段 ~6 分钟 + 评价 ~3 分钟；多次续跑（故障链见 §4） |
| 覆盖 | **8,032/8,032（公共子库，与 CLIP 完全一致；14 个 B-Rep 无效 UID 同源缺失）→ 7,920 保留查询** |
| 权重 | `baseline_trimodal_v4.pth`（1,349.7MB，`datasets/exanos/MMCAD` 经用户 token 下载）+ 官方 `brep_norm_stats_v4.npz` + EmbeddingGemma-300m 本地快照（离线加载，`Config.TEXT_MODEL`+`__defaults__` 补丁） |
| 运行目录 | `runtime/idea01_mmcad_g5_20261116/`（本地镜像 `experiments/runs/idea01_mmcad_g5_20261116/`，SHA256SUMS 9+3 项全 OK） |

## 2. 首批门（并入正式运行，全部通过）

- 官方参考件回环（192610/192621/192623）：经我们的 STEP→npz→`prepare_brep`→`embed_brep` 路径，与官方发布 embedding **余弦 1.0 ×3**
- 16 件 gallery STEP：嵌入有限且非零 ✓

## 3. 同协议对比表（7,920 保留查询 × 8,032 公共子库；配对 bootstrap seed 20261112 × 5,000 共享抽样）

| 方法 | 表示 | R@1 | R@5 | R@10 | R@20 | MedR |
|---|---|---:|---:|---:|---:|---:|
| Text2CAD SUM（认证分同库重排） | 命令序列原生分 | 0.2020%（16） | 0.9217%（73） | 1.6414%（130） | 2.6136%（207） | 3405.5 |
| CLIP ViT-B/32 单视角（已审） | 渲染图 | 0.1515%（12） | 0.3914%（31） | 0.7449%（59） | 1.2374%（98） | 2057.5 |
| **MM-CAD baseline_trimodal_v4**（本轮） | B-Rep | **0.0505%**（4） | **0.2146%**（17） | **0.4545%**（36） | **0.8712%**（69） | **2632.5** |

**MM-CAD − SUM 配对差**：R@1 −0.152pp [−0.263, −0.043]；R@5 −0.707 [−0.964, −0.466]；R@10 −1.187 [−1.570, −0.838]；**R@20 −1.742pp [−2.346, −1.213]**；MedR −773。全部 CI 排除 0。

**SUM 子库命中 16/73/130/207 与全库认证一致**（与 CLIP 运行相同的一致性核验）。

## 4. 三方法初步排序（公共子库，正式口径）

**Text2CAD 原生 SUM（2.61%）> CLIP 单视角（1.24%）> MM-CAD（0.87%）**，R@20 全部配对差 CI 排除 0。

事实层读出：MM-CAD（文本-B-Rep 联合检索，768 维）在当前 G5 UID 集上的 Top-20 命中不足 SUM 的三分之一，也低于 CLIP。可解释方向（非结论）：①MM-CAD 的 B-Rep 塔训练分布（ABC/Onshape 工业件）与 DeepCAD 衍生序列的几何风格差异；②我们的 B-Rep 来自 minimal_json 重建的 STEP，与原训练的 CAD 内核 B-Rep 存在表达差异；③文本塔 EmbeddingGemma-300M 与 CLIP 文本系的能力差异。这些归因需要 §4 的分层与错例分析区分——本轮不做归因断言。

## 5. 工程故障链（持续授权，全部留痕）

run1 sidecar 结构（v1 扁平）→ run2 输出目录冲突 → run3 EmbeddingGemma 快照 tar 嵌套层级 → run4 路径笔误 → run5 目录残留 → run6 离线加载失败（`__defaults__` 补丁修复 def-time 绑定）→ run7 cad 轴解析（cad_store 无 cad_id，evaluation_map 兜底）→ run8 gate 无条件重跑转换（share4 无 OCC）→ STEP 补传输（本地 cadclip 转换 8,032 npz 全部有效，`as_completed(map)` bug 修复后校验 0 失败）→ run10 文本段 47 线程 OMP 死锁（MMCAD_TORCH_THREADS=8 修复）→ run11-13 resume 路径变量绑定三连修。所有日志 `runtime/mmcad_g5_run*.log`。

## 6. 证据路径

- 逐查询排名（MM-CAD 与 SUM 双臂）：`experiments/runs/idea01_mmcad_g5_20261116/entry_evaluation/query_ranks.jsonl`
- Top-20 / 汇总 / 配对统计：`entry_evaluation/{text_top20.jsonl, results.json}`、`final_summary.json`
- 分数工件：`mmcad_scores_subset_axis_f64.npy` + sidecar（cad 轴来自 sidecar `cad_indices`）
- SUM 臂核验：`sum_arm_certified_check.json`（2,044/7,920 与全库认证名次相同——受 14 缺失候选影响的查询按子轴重排，符合预期）

## 7. 边界

- MM-CAD 的 B-Rep 输入由 DeepCAD 衍生 JSON 重建（`CadSeqProc.save_stp`），非原训练管线直接产物——表示域差异属于对照的一部分（§4 计划要求的"同时改变表示与训练"维度），归因待分层分析。
- 文本提示模板（"title: none | text: …"）与 max_length=512 为官方 inference.py 原样；dim=768（全维）。
