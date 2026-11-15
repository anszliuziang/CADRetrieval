# CLIP G5 全量评测结果（2026-11-15 首版；同日数值复审通过 + 收尾订正）

计划 `IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md` §3 顺序 1 的正式交付。用户 2026-11-15 授权全量。**零 NPU；隔离 env `cadclip`（CPU）。**

**复审与订正（2026-11-15，Codex 数值复审通过）**：审查者从原始分数独立复算两种方法 15,840 个正例名次与 5,000 次配对重抽样，全部一致（审计 `CADRetrievalIdea1_runtime/clip_g5_codex_review_20261115/EXPERIMENT_AUDIT.md`）。收尾订正：①"一次启动 784 秒"订正为"**第五次续跑成功，耗时 784.4 秒**"（故障链见 §4）；②SUM 源矩阵与切片的逐位核验来自本次复审；③**权重版本回执补录**：日志只记录了模型仓库名，实际加载权重经本地缓存钉定——`openai/clip-vit-base-patch32` 权重快照 `c237dc49a33fc61debc9276459120b7eac67e7ef`，`model.safetensors` SHA256 `99d28a652e6ec466…`（605,157,884 字节），回执 `runtime/idea01_clip_g5_20261115/weights_version_receipt.json`；方法轮改进：`from_pretrained` 显式传 `revision` 钉定。

## 1. 执行记录

| 项 | 值 |
|---|---|
| 命令 | `D:/miniconda/envs/cadclip/python.exe experiments/scripts/run_clip_g5_20261115.py --out-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_clip_g5_20261115` |
| 实际退出 | **0**（`runtime/clip_g5_exit.txt`；日志 `runtime/clip_g5_run.log`→`clip_g5_run5.log`——工程故障链见 §4） |
| **执行口径** | **第五次续跑成功，耗时 784.4 秒**（渲染全复用 1.5s + CLIP 加载 4.3s + 图像编码 585.2s + 文本编码 83.7s + 评价含 5,000 次 bootstrap） |
| 冻结规格 | `experiments/configs/clip_g5_render_spec_20261115.json`（SHA `938b800c50f31130…`）：单视角等轴测 azim45/elev30、平面色、白底 448px、CLIP 224 预处理、77 token 截断、L2 归一化余弦 |
| 权重 | `openai/clip-vit-base-patch32`，加载快照 `c237dc49…`，`model.safetensors` SHA256 `99d28a652e6ec466…`（回执见上） |
| 运行目录 | `runtime/idea01_clip_g5_20261115/`（SHA256SUMS 24 项 OK；entry_evaluation 3 项 OK） |

## 3b. 分层读出（复审补充，证据 `clip_g5_codex_review_20261115/deterministic_checks.json`）

复审分层复算显示 Top-20 差距几乎全部来自**最短 CAD 四分位**：SUM 命中 144 条 vs CLIP 37 条；而**最长四分位 CLIP 反超**：CLIP 29 条 vs SUM 6 条。这与"CLIP 对短 CAD（渲染信息量小）判别弱、对长 CAD 的全局形状匹配反而占优"一致——该分层假设值得在下一个模型上验证。

## 2. 覆盖（诚实口径）

- 渲染覆盖 **8,032/8,046（99.826%）**；**14 个 UID 持续转换失败**（OCC "Invalid Model"，每件重试 2 次，失败 trace 单独存档 `render_failure_*.txt`）：`0001/00018369, 0007/00077685, 0010/00109798, 0023/00230494, 0023/00231710, 0029/00299643, 0034/00343787, 0045/00453026, 0058/00584135, 0065/00657352, 0065/00659246, 0067/00672858, 0074/00745875, 0095/00954235`（`coverage.json`）。
- 评价按**公共子库模式**：8,032 件覆盖轴，正例落在缺失 UID 的 **14 条查询冻结剔除 → 7,920 条保留查询**（入口 kept/dropped 记录）。

## 3. 首批门（并入正式运行，全部通过）

32 UID（四分位各 8，G5 边界规则）：渲染非空可解码 ✓；图像向量有限且非零 ✓；UID 与冻结轴一一对应 ✓；**截断普查：214/3,644 条唯一文本超 77 token（最长 102）**，截断编码有限非零 ✓。

## 4. 工程故障链（持续授权处理，全部留痕）

attempt1（`clip_g5_run.log`）：14 缺失 PNG 在全量编码处 FileNotFound → 崩溃。修复：覆盖检测 + 子库模式（编码已覆盖集、SUM 认证工件切片到同轴配对）。attempt2（`run2.log`）：输出目录已存在拒绝 → 断点续跑支持（渲染 skip-existing 复用，8032 件 1.5s）。attempt3（`run3.log`）：`log` 先用后定义。attempt4（`run4.log`）：`missing_cad_queries` 先用后定义。attempt5（`run5.log`）＝成功。首次运行现场保留于 `idea01_clip_g5_20261115_attempt1/`。

## 5. 同协议对比表（7,920 保留查询 × 8,032 公共子库；配对 bootstrap seed 20261112 × 5,000 共享抽样）

| 方法 | 表示 | R@1 | R@5 | R@10 | R@20 | MedR |
|---|---|---:|---:|---:|---:|---:|
| **CLIP ViT-B/32 单视角**（本轮） | 渲染图 | **0.1515%**（12） | **0.3914%**（31） | **0.7449%**（59） | **1.2374%**（98） | **2057.5** |
| Text2CAD SUM（认证分数同库重排） | 命令序列原生分 | 0.2020%（16） | 0.9217%（73） | 1.6414%（130） | 2.6136%（207） | 3405.5 |

**配对差（CLIP − SUM）**：R@20 = **−1.376 pp，95% CI [−1.975, −0.826]**（排除 0，SUM 显著更好）；MedR = −1348 [−1637, −888]。SUM 臂在子库上的命中数 16/73/130/207 与全库认证值完全一致（被剔 14 查询在 K≤20 无命中），MedR 3405.5 vs 认证 3411.5 的微差来自子轴重排（方向如预期）。

**读出（事实层）**：在当前冻结的单视角渲染规范下，CLIP ViT-B/32 的检索显著弱于冻结 Text2CAD 的原生条件分。R@20 命中 98 vs 207，且 MedR 更好（2057 vs 3405）说明 CLIP 把正例整体排得更靠前但顶尖命中率低——与"全局相似但局部判别不足"一致。单视角平色渲染是保守配置；多视角/光照增强属于方法轮变量，不属于本轮结论。

## 6. 证据路径

- 逐查询排名（CLIP 与 SUM 双臂）：`runtime/idea01_clip_g5_20261115/entry_evaluation/query_ranks.jsonl`
- 每文本 Top-20：`entry_evaluation/text_top20.jsonl`；汇总：`entry_evaluation/results.json`；SUM 臂核验：`sum_arm_certified_check.json`
- 分数工件：`clip_scores_subset_axis_f64.npy`（SHA 见 `SHA256SUMS`）+ `clip_scores_sidecar.json`（规格/权重/覆盖 pin）；SUM 切片工件 `sum_scores_subset_*.json`（sliced_from 指向已认证 v1 工件）
- 截断/覆盖统计：`encode_timing.json`、`coverage.json`、`final_summary.json`

## 7. 边界

- 探索性外部基线读出：单视角、平色、ViT-B/32 是保守下限配置；不外推到"CLIP 不能做 text→CAD 检索"的类级结论。
- 14 个缺失 UID 同时是 MM-CAD/未来渲染方法的候选覆盖缺口（同一几何源头）；公共子库口径已可对齐三方法。
