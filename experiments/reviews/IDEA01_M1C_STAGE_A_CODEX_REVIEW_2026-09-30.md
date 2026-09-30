# Idea 01 M1c Stage A：Codex 代码与数据审查

日期：2026-09-30。结论：**接受静态评分定义和 CPU 数值核心，未授权真实模型前向或 NPU 评分。** `ACTIVE_TASK.md` 仍为 `implementation`、`approved_sha: 无`、`gpu_budget: 0`。

## 审查证据

- 独立对照官方 pinned 源码：`Cad_VLM/train.py:256-285` 对 input/target 逐位移位，`Cad_VLM/models/loss.py:13,28-43` 使用两头有效位 masked mean 与 `label_smoothing=0.1`，`CadSeqProc/json2vec.py:260-262` 定义逐列 mask，`CadSeqProc/cad_sequence.py:244,260` 将 `[START,0]` 同时用作首位和末位。Kimi 的 `TEXT2CAD_SOURCE_AUDIT.md` 与这些定义相符。
- 逐读 `experiments/src/idea01/text2cad_vector_scorer.py`：目标为 `vec[1:]`，有效位是目标非零项，原始 logits 先转 FP32 做逐头 `log_softmax`，x/y 有效项分别求和；没有从 scorer 读取 qrels、UID 映射或 CSV。末位 `[1,0]` 的 x 项计入，y 项排除，padding 两头排除。raw sum 是主分数，mean 仅诊断。
- Codex 在本机重新执行 `D:\miniconda\python.exe -m pytest experiments/tests/test_text2cad_vector_scorer.py -q`：**45 passed in 1.80s，exit 0**。其中手写 float64 log-sum-exp 参考、均匀 logits 闭式、末位双向扰动、masked logits 污染和 batch/单项一致性提供了非镜像验证。既有套件在此 Windows Python 中缺 `transformers`/`pyarrow`，与新增文件无关，不据此放行完整项目。
- Codex 执行 `experiments/scripts/audit_text2cad_gallery_vectors.py` 对官方 v1.0 test 全 8,046 UID 扫描：**8,046/8,046 通过，拒绝原因空**；脚本先独立核对 `[272,2]` 与 `mask == (vec == 0)`，再用待审 validator 核对首尾、尾部 padding、范围与 flag/index padding。未填充长度中位 34、P90 87、P99 142、最大 212；结果已登记于 `experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md`。它证明输入覆盖，不证明模型可前向。

## 审查中修正的两处表述

1. 子计划中的 BERT `model.safetensors` SHA256 曾漏末位 `4`；已按本地与远端整文件重算、资产清单的 64 位值修正。Kimi 报告保留当时发现的勘误记录。
2. scorer 原 docstring 声称 `flag/index` 与完整确定性构造一致，但实现仅检查形状、值域及 padding 值；已将 docstring 与 Kimi 交付说明改为实际保证。Stage B 须直接传官方固定向量，不能自行重建标记。

## Stage B 准入要求

- 原版官方模型路径需要适配 Ascend：`text_embed.py` 硬编码 CUDA；`decoder.py`/`adaptive_layer.py` 顶层导入 OCC 几何模块。最小前向适配必须保留 checkpoint 参数键、注意力数学、官方因果 mask 对 START 列的屏蔽和文本 tokenizer 行为，并注明 CC BY-NC-SA 4.0 来源与改动。
- 模型 checkpoint 的 172 个非 BERT `module.` 键与 pinned BERT 必须逐键严格加载并核对数值。测试所有 3,644 种查询的 BERT token 长度，超过 512 则显式拒绝；不得静默截断。真实前向须在代码审查与 `ACTIVE_TASK.md` 完整 SHA/预算批准后进行。
- 先做一对，再做跨长度的十对以及 batch=1/batch>1 分数等价。尾部 padding 裁剪、文本缓存、候选无交叉注意力前缀缓存，均须用同 pair 数值等价证明。两卡端到端吞吐的七天准入下界约 63 对/秒，须以长度分布、冷启动、I/O 和写盘完整外推。

本阶段没有真实模型打分，也没有检索准确率。后续实现仍由本地 Claude Code/Kimi 进行；Codex 再审 Stage B 代码后才开放 G1 预算。
