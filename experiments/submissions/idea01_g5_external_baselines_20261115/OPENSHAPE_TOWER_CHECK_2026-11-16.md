# OpenShape 双塔验证（yunnao-share4，2026-11-16）

验收指示的"在目标设备验证点云与文本两塔"已完成。**双塔在 share4 上加载并产生有限嵌入，联合维度一致（1280）**。运行目录 `runtime/openshape_tower_check_20261116/`（本地镜像 `experiments/runs/openshape_tower_check_20261116/`）。

## 执行

| 项 | 值 |
|---|---|
| 机器 | `yunnao-share4`（用户指定/指示机器；NPU 空闲未占用，塔验证用 CPU） |
| 环境 | `cadretrieval-glm-torch271`（torch 2.7.1+cpu）+ huggingface_hub 0.36.0 + open_clip 4.2.2 + dgl/torch_redstone（见"部署缺口"） |
| 权重 | bigG：HF **数据集无关的模型仓** `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` 经 **hf-mirror**（share4 直连 HF 不通，`huggingface.co=000`；hf-mirror=200）下载 20.32GB 快照（含 `open_clip_pytorch_model.bin` 原生格式与 HF 分片两种）；PointBERT：`OpenShape/openshape-pointbert-vitg14-rgb/model.pt` 388MB 同路下载；推理库 `OpenShape/openshape-demo-support` 快照 |
| 数据 | 32 个点云 npz（10,000×3 [-1,1] + 中性 RGB 0.5，本地验证件转存）+ `pair_manifest_v4_worker/texts.jsonl` 冻结唯一文本 |
| 命令 | share4 上 `openshape_tower_check_share4.py`（本地 `experiments/scripts/`），exit 0 |

## 结果（事实）

| 塔 | 输出 | 耗时 | 有限性 |
|---|---|---|---|
| PointBERT G14 点云塔（4 件点云） | **1280 维** | 7.69s（≈1.9s/件，含 FPS shim） | 全部有限 ✓ |
| OpenCLIP ViT-bigG-14 文本塔（16 条冻结文本） | **1280 维** | 首条 1.07s，均值 1.12s/条 | 全部有限且非零 ✓ |

两塔嵌入维度一致（1280）确认进入同一 OpenShape 联合空间。4×16 交叉余弦网格（0.006–0.139）已记录于 summary.json（未配对样本的事实读数，不作结论）。点云塔嵌入范数 832–1073（非单位向量——评分时按协议 L2 归一化）。

**全库外推（仅规划参考）**：点云塔 8,032 件 × ~1.9s ≈ 4.2 小时；文本塔 3,644 × 1.12s ≈ 68 分钟；点云几何生成（B-Rep→mesh→采样）另计 ~15 分钟。**合计约 5.5 小时 CPU，单机可完成。**

## 部署缺口（正式全量前须处理）

1. **dgl 依赖**：pip dgl 2.x 无 torch 2.7.1 的 graphbolt 构建（`libgraphbolt_pytorch_2.7.1.so` 缺失），且 dgl 链式依赖 torchdata 0.11（无 datapipes）。本次探针以**显式记录的等价 FPS shim**（标准确定性最远点采样，`runtime/probe_shims/dgl/`）供 PointBERT 采样——**仅探针用**；正式评分的 FPS 语义须在方法轮冻结（方案 a：conda 装 torch 2.4 配套官方 dgl；方案 b：以 shim 的标准 FPS 冻结为规范并注明与官方 dgl 实现的差异核验）。
2. **NPU 兼容**：本轮全部 CPU。PointBERT（Transformer 系）与 bigG 文本塔在 torch_npu 的可运行性未验证——方法轮若要 NPU 需先做单件预检。
3. **点云朝向**：z-up 假设未核（探针件与 OpenShape 训练的 gravity 轴一致性目测待做）。

## 与 MM-CAD 的共同事项

- 公共子库：CLIP 覆盖 8,032 件（14 UID B-Rep 本身无效）——三方法统一 8,032 口径。
- MM-CAD 已达可全量状态（参考件回环余弦 1.0、全库估 ~2 小时 CPU），待启动令。
