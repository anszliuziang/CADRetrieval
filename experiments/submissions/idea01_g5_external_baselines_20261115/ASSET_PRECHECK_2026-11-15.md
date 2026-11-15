# 三方法接入盘点与资产预检（2026-11-15 首版；同日追加真实转换/渲染/CLIP 编码探针结果，见 §6）

计划 §2 第 3 项交付。首版为零模型前向、零大文件下载的桌面核查；**2026-11-15 追加**：按验收意见在隔离环境（checkout 外 conda env `cadclip`）完成 32 UID 的真实 **JSON→几何→固定视角渲染→CLIP 编码** 批，成功率 **100%（32/32）**（§6）。

## 1. CLIP + 固定视角渲染（顺序 1，首选）

| 项 | 状态 |
|---|---|
| 建议权重 | `openai/clip-vit-base-patch32`（HF，MIT 许可）。**已下载至 HF 默认缓存（2026-11-15，589MB 实测）** |
| 本地运行时 | 隔离 env `cadclip`（python 3.10 + pythonocc-core 7.7.2 + transformers + torch 2.14.1+cpu）已建并实测跑通；CPU 编码 8,046 图量级按探针外推约 8 分钟；NPU 路径未验证，不默认 |
| 渲染资产 | 数据集无现成渲染；**探针已实测渲染 32 件**（matplotlib 等轴测，`clip_probe_20261115/renders/`）。转换链：minimal_json → `CADSequence.from_minimal_json` → OCC B-Rep → mesh → 固定视角渲染 |
| 视角规范（建议，方法轮冻结） | 探针用单张固定等轴测（azim 45°/elev 30°，白底，网格几何归一化）；多视图聚合（4 视角均值池化）作为方法轮对照。生产规格（含光照）待冻结 |
| 预处理 | CLIP 官方 `CLIPImageProcessor`（224×224），文本侧 BPE ≤77 token（已实测文本编码 0.618s/条） |
| UID 对应 | 渲染文件名 = cad_id（`0000__00000915.png` 风格）；评价端经 gallery.jsonl 连接 |
| 缺项 | ①生产渲染规格与光照待冻结（含 normalize/bbox 集成缺口，见 §6）②多视图聚合决策 ③编码输出格式（f32 向量工件，经统一评价入口的 binary artifact 通道） |

## 2. OpenShape PointBERT（顺序 2）

| 项 | 状态 |
|---|---|
| 权重 | HF `OpenShape/openshape-pointbert-vitg14-rgb`（官方首选，ModelNet40 Top1 84.4；Apache-2.0）。**本地未缓存，需下载**（PointBERT 权重数百 MB 级） |
| 文本编码器 | **官方配对 OpenCLIP ViT-bigG-14（laion2b_s39b_b160k）≈2.5B 参数**——3,644 条文本编码在 CPU 上不现实，需 GPU/NPU 或长时 CPU；这是三个方法里最大的设备依赖 |
| 点云输入 | 10,000×3，xyz ∈ [-1,1]，gravity **z 轴**对齐；rgb 通道对 CAD 点云可置中性值（RGB 检查点期望 [0,1]；置 0.5 灰并在配置记录——与训练分布的差异是已知偏差，写入结果解释） |
| 转换链 | minimal_json → json2stl_skt3d（OCC）→ STL → trimesh `sample_surface` 10,000 点 → 归一化+z 朝向 |
| 设备依赖 | 官方环境 torch 1.12.1 + CUDA 11.3；spconv 变体需 MinkowskiEngine——**选 PointBERT 骨干可避开**；share4 NPU 兼容性未验证（LayerNorm/attention 类算子为主，PointBERT 主体可能可用，需预检） |
| UID 对应 | 点云文件名 = cad_id；同 gallery 连接 |
| 缺项 | ①权重+bigG 文本编码器下载（数 GB）②OCC 点云环境 ③NPU/CPU 推理路径验证 ④rgb 通道决策记录 |

## 3. MM-CAD 纯文本到 B-Rep（顺序 3）

| 项 | 状态 |
|---|---|
| 权重 | HF `exanos/MMCAD` `checkpoints/baseline_trimodal_v4.pth`（text=EmbeddingGemma-300M + BRepFormer + DGCNN，d=768 Matryoshka 空间）。代码 MIT；派生资产 CC BY-NC 4.0；权重许可未明确标注（HF 页核实） |
| 输入表示 | **B-Rep**：STEP → `scripts/step_to_brep.py`（**pythonocc-core 7.7**）→ npz（16 列面特征+质心+法向+edge_to_faces）；归一化必须用官方 `brep_norm_stats_v4.npy`（182,131 件训练件统计），不得自算 |
| 转换链 | minimal_json → CadSeqProc `json2step.py`（OCC）→ STEP → step_to_brep → B-Rep npz → embed_brep |
| 文本塔 | EmbeddingGemma-300M——3,644 条文本 CPU 可行（比 bigG 小两个数量级） |
| 设备依赖 | 未官方声明；BRepFormer/DGCNN 体量小，CPU 大概率可行，GPU/NPU 待预检 |
| UID 对应 | STEP/B-Rep npz 文件名 = cad_id |
| 缺项 | ①checkpoint+norm stats+EmbeddingGemma 下载 ②pythonocc 7.7 环境（与 CadSeqProc 的 OCC 版本兼容性要一起核）③双段转换（JSON→STEP→npz）的失败率待首轮测定 |

## 4. 本轮实测的资产 SHA 与探针记录

- `minimal_json_0000_0099.zip`：本地 `.cache/text2cad_official/text2cad_v1.0/misc/minimal_json/`，114,762,278 字节（云脑 `datasets/Text2CAD-v1.0/422d4c93…/text2cad_v1.0/misc/` 同源；正式使用前按整文件重算 SHA 入固定清单）。
- 结构探针（4 UID，覆盖长度四分位 Q1–Q4）：`0051/00517599`(L=26, 1 part, 2 curves, 1 extrusion)、`0050/00507628`(L=27, 4 curves)、`0047/00472622`(L=65, 2 parts, 12 curves, 2 extrusions)、`0049/00491022`(L=73, 16 curves, 1 extrusion)——全部 JSON 可解析、sketch/extrusion 结构完整、坐标平移非零样本存在。
- 覆盖交集：zip 内 176,019 UID ∩ gallery 8,046 = **8,046，missing=[]**（2026-11-15 云脑实测）。

## 5. 首轮评分（CLIP）所需资源汇总

| 资源/动作 | 量级 | 状态 |
|---|---|---|
| CLIP ViT-B/32 权重下载 | ~605MB（HF openai/clip-vit-base-patch32） | 待下载（需授权） |
| OCC 环境（OCP/pythonocc + trimesh + loguru/rich） | conda/pip 若干百 MB | 本地/服务器均未装；Windows OCP wheel 对 py3.13/3.14 的支持待核——**可能需要专用 conda env** |
| 渲染 8,046 张（4 UID 探针先扩到 32 个分层 UID 预检） | STL 化秒级/件，渲染毫秒级/件；全库估 1–3 小时 CPU | 工具在，脚本待写 |
| CLIP 编码 | 8,046 图 + 3,644 文本；CPU 1–2h 或单 NPU 更快 | 权重就绪后可跑 |
| 评价 | 统一入口（本轮已交付） | 就绪 |

三条方法共同的开工前置：**一次明确授权的依赖安装清单**（OCC 系 + 各方法权重下载），建议作为下一轮的第一步，与渲染脚本评审一并处理。

## 6. 真实探针批（2026-11-15 追加；验收意见授权的隔离环境批）

按验收意见：①补齐 Q2 探针（首版 4 个探针按正确边界规则实际落在 Q1/Q1/Q3/Q4——`np.searchsorted(cuts, L, 'left')` 边界归下四分位；本次按每四分位 8 个 UID 重新抽样，共 32 个，Q1–Q4 各 8）；②在隔离环境跑真实全链路。**结果：32/32 成功（100%）**。

- 环境：conda env `cadclip`（checkout 外）：python 3.10 + **pythonocc-core 7.7.2（conda-forge；CadSeqProc 绑定 `OCC.Core.*`，不兼容 pip 的 cadquery-ocp/OCP）** + pip：trimesh/pandas/scikit-learn/plyfile/joblib/networkx/matplotlib/transformers/**torch 2.14.1+cpu**。完整规格与已知缺口见 `experiments/configs/clip_probe_env_20261115.json`。
- 脚本：`experiments/scripts/probe_clip_render_20261115.py`（每 UID 分阶段计时与失败原因落盘）。
- 产物：`CADRetrievalIdea1_runtime/clip_probe_20261115/`（32 张固定等轴测渲染 PNG、32 个 STEP、summary/per_uid JSON）+ 冒烟 `clip_probe_smoke_20261115/`（4 UID，几何+渲染）。
- CLIP 权重：`openai/clip-vit-base-patch32` 已下载至 HF 默认缓存（589MB 实测缓存占用）。
- 每件平均耗时（32 件实测）：OCC B-Rep 0.025s、STEP 导出 0.010s、mesh 0.009s、渲染 0.056s、CLIP 图像编码 0.058s（CPU）——**全库 8,046 件外推约 21 分钟单进程**（仅作规划参考）；文本塔 3,644 条 0.618s/条 → 数分钟级。
- 新发现的集成缺口（方法轮冻结决策）：①minimal_json 不含 `properties.bounding_box`，CadSeqProc `CADSequence.normalize()` 依赖它（`from_minimal_json` 返回 bbox=None 即崩溃）——生产管线需改为从生成的 B-Rep 取 bbox（BRepBndLib）或复刻 json2vec 量化路径；②探针渲染为 matplotlib 平面色（无光照），生产渲染规格（光照/分辨率/视角数）待方法轮冻结；③本版 transformers `get_image_features` 返回 ModelOutput（投影向量在 `pooler_output`），已适配。
- CLIP 权重下载耗时实测 ~26 分钟（慢代理），一次性成本。
