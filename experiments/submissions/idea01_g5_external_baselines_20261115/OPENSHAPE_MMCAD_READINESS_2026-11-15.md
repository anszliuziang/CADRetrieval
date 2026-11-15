# OpenShape 与 MM-CAD 启动准备清单（2026-11-15）

计划 §3 顺序 2/3 的并行准备。两条转换路径已在隔离环境 `cadclip` 用同一批 G5 UID 实测打通；权重尚未下载（体积/许可见下）。正式全量评分在 CLIP 之后按同一准入推进。

## 1. OpenShape PointBERT（顺序 2）——转换路径已验证 32/32

| 项 | 状态 |
|---|---|
| 转换路径验证 | **32/32 成功**（`runtime/openshape_prep_20261115/`）：minimal_json → CadSeqProc OCC B-Rep → mesh → `sample_points(10000)` → bbox 中心归一化到 [-1,1] → 10,000×3 float32 npz + 中性 RGB 0.5。脚本 `experiments/scripts/probe_openshape_pc_20261115.py` |
| 点云输入规范 | 官方要求 10,000×3，xyz∈[-1,1]，gravity **z 轴**对齐。缺口：DeepCAD 序列坐标系是否 z-up 未核——正式评分前须对少量 UID 做朝向检查（渲染侧可目测）；若非 z-up 需固定轴向变换并写入规格 |
| RGB 通道 | 官方 checkpoint 为 RGB 训练；CAD 点云无颜色 → **中性 0.5 灰**（与训练分布的差异是已知偏差，进结果解释，不静默） |
| 权重 | HF `OpenShape/openshape-pointbert-vitg14-rgb`（Apache-2.0）——**未下载**，PointBERT 权重数百 MB 级 |
| 文本塔 | **官方配对 OpenCLIP ViT-bigG-14（laion2b_s39b_b160k，≈2.5B 参数，权重 ~10GB 级）**——3,644 条文本 CPU 编码不现实（数小时～天级），需 GPU/NPU 或分批长时 CPU；share4 NPU 对 OpenCLIP bigG 的兼容性未验证。**这是该方法的最大启动缺口** |
| 运行代码 | OpenShape 官方 repo（Colin97/OpenShape_code）未克隆；推理依赖其 demo 库（PointBERT 骨干可避开 MinkowskiEngine） |
| 环境 | cadclip 已有 torch 2.14.1+cpu + trimesh；OpenShape 官方环境为 torch 1.12.1+CUDA——版本兼容性待克隆后核（CPU 路径预计可跑 PointBERT 塔本身） |
| 预计全量耗时 | 点云生成 ~0.1s/件 ×8,046 ≈ 15 分钟；PointBERT 图像塔 CPU 未知（数百 ms/件量级 → 0.5–1.5 小时）；**bigG 文本塔为决定性变量** |
| 下一步（启动前清单） | ①克隆 OpenShape repo 并核 checkpoint 加载 ②下载 PointBERT+bigG 权重 ③z-up 朝向核验 ④设备决策（NPU 预检或长时 CPU） |

## 2. MM-CAD 纯文本到 B-Rep（顺序 3）——转换路径已验证 1/1（等价类）

| 项 | 状态 |
|---|---|
| 转换路径验证 | **实测通过**：探针 STEP（`clip_probe_20261115/steps/`，51 faces/222 edges）→ 官方 `scripts/step_to_brep.py`（pythonocc 7.7.2，cadclip 环境直接可用）→ 输出张量与官方文档逐项吻合：`face_features(192,16)/face_centroids(192,3)/edge_to_faces(512,2)/face_mask(192,)/face_normals(192,3)/num_faces/num_edges`，全部有限（`runtime/mmcad_prep_20261115/test_brep.npz`）。官方 repo 已克隆至 `.cache/MMCAD-official/`（commit 见本地镜像） |
| B-Rep 归一化 | 必须用官方 `brep_norm_stats_v4.npz`（182,131 训练件统计）+ `inference.prepare_brep()`；**stats 文件随 checkpoint 下载**，未自算 |
| 权重 | HF `exanos/MMCAD`：`baseline_trimodal_v4.pth`（text=EmbeddingGemma-300M + BRepFormer + DGCNN）+ norm stats + EmbeddingGemma-300M——**未下载**；体量 GB 级（HF 页核）；代码 MIT，派生资产 CC BY-NC 4.0，权重许可 HF 页再核 |
| 文本塔 | EmbeddingGemma-300M——3,644 条文本 CPU 可行（分钟级） |
| STEP 全量生成 | CLIP 全量运行已在逐件落盘 STEP（`idea01_clip_g5_20261115/steps/`）——**MM-CAD 的 STEP 源可直接复用 CLIP 运行的副产物**；转换失败清单与 CLIP 渲染失败清单同源 |
| 预计全量耗时 | STEP 已由 CLIP 运行顺带完成（0.010s/件）；step_to_brep ~0.3s/件 ×8,046 ≈ 40 分钟；BRepFormer+EmbeddingGemma 编码 CPU 分钟级 |
| 下一步（启动前清单） | ①下载 checkpoint+stats+EmbeddingGemma ②`prepare_brep` + 编码的少量 UID 端到端冒烟 ③"官方示例以预计算向量作图库输入"的图库向量生成顺序确认 |

## 3. 共同事项

- 两个方法的分数矩阵都走统一评价入口的固定 binary artifact 通道（方向 higher），与 CLIP 同一协议、同一配对 bootstrap（seed 20261112 ×5000）对认证 SUM。
- 三个方法共享的几何失败集合（OCC "Invalid Model" 等）会在同一些 UID 上出现；首个方法确定覆盖后，其余方法按同一公共子库对齐（入口 `cad_subset` + 保留查询上限已修复）。
- 权重下载（OpenShape bigG ~10GB 级 + MM-CAD GB 级 + OpenShape 点云塔数百 MB）是两项启动的共同前置，需一次授权的下载清单与磁盘预算（HF 默认缓存盘余量已够）。
