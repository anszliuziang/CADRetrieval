# 三方法接入盘点与资产预检（2026-11-15）

计划 §2 第 3 项交付。本轮零模型前向、零大文件下载；以下"已核实"均指本地文件/官方仓库/官方 HF 页面级核查，运行覆盖由首轮评分测定。

## 0. 关键结论

1. **几何源头已备齐**：Text2CAD v1.0 数据集自带 `misc/minimal_json/minimal_json_0000_0099.zip`（110MB，176,019 个 DeepCAD 风格参数化 JSON；SHA 见 §4）。**8,046 个测试 UID 交集 = 8,046/8,046，缺失 0**（本地 zip 实测，2026-11-15）。gallery.jsonl 每行带 `minimal_json_member` 字段（`n_representations=3` 全库一致），UID→文件路径连接零成本。
2. **官方转换工具已具备**：Text2CAD 官方仓库（本地镜像 Git SHA `669a81472ede2269837c22a4d00070eaac360c82`，LICENSE CC BY-NC-SA 4.0）的 `CadSeqProc/` 提供 `json2step.py`（JSON→STEP B-Rep）、`json2stl_skt3d.py`（JSON→STL mesh）、`cad_sequence.create_mesh/sample_surface`（B-Rep→mesh→点云采样）、`OCCUtils/`（pythonocc 辅助）。依赖：pythonocc-core/OCP、trimesh、loguru/rich。
3. **结构探针通过**：按 G4 有效长度四分位（cuts 27/40/70）抽 4 个 UID 实测 JSON 解析——parts/sketch loop/`extrusion` 键结构完整（探针记录见 RESULTS 附录）。深度转换（STEP/STL/点云）需先安装 OCC 环境，见各方法"启动条件"。

## 1. CLIP + 固定视角渲染（顺序 1，首选）

| 项 | 状态 |
|---|---|
| 建议权重 | `openai/clip-vit-base-patch32`（HF，MIT 许可）。**本地 HF 缓存仅有 ref 指针（1KB），权重本体未下载**；需下载 model.safetensors ≈605MB |
| 本地运行时 | CPython 3.14 已有 torch（2.x CPU）+ transformers 4.x，`CLIPModel` 可直接用；CPU 编码 8,046 图 + 3,644 文本可行（估 1–2 小时量级）；NPU 路径未验证，不默认 |
| 渲染资产 | **不存在现成渲染**（数据集只有几何 JSON 与 token 序列）。转换链：minimal_json → CadSeqProc `json2stl_skt3d.py`（OCC）→ STL → 离屏渲染（matplotlib/trimesh+pyrender）→ 224×224 |
| 视角规范（建议，方法轮冻结） | 首选单张固定等轴测视图（azimuth 45°/elevation 30°，STL 归一化到单位立方后正交投影）做首轮；多视图聚合（4 视角均值池化）作为第二轮对照。两版都要在配置里写死并记 SHA |
| 预处理 | CLIP 官方 `CLIPImageProcessor`（224×224 resize+center crop+归一化），文本侧 BPE tokenize ≤77 token |
| UID 对应 | 渲染文件名 = cad_id（`0000/00000093.png` 风格）；评价端经 gallery.jsonl 连接 |
| 缺项 | ①CLIP 权重下载（~605MB）②OCC 渲染环境安装 ③渲染脚本本身需新写（CadSeqProc 无现成 render-to-image 入口） |

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
