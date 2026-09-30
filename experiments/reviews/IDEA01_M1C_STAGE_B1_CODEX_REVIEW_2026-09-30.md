# Idea 01 M1c Stage B1：Codex 冻结模型移植审查

日期：2026-09-30。结论：**接受原版 Text2CAD 的 OCC-free、显式设备、零前向架构移植及严格加载器，作为 B2 runner 的基础；不授权真实模型 CPU/NPU 前向或检索评分。** `ACTIVE_TASK.md` 仍为 `implementation`、`approved_sha: 无`、`gpu_budget: 0`、`max_scored_pairs: 0`。

## 独立核验

- 对照固定的官方源码 Git SHA `669a81472ede2269837c22a4d00070eaac360c82`。Codex 用 AST 比较了官方与移植的 `multi_head_attention_forward`、`FeedForwardLayer.forward`、`MultiHeadAttention.forward`、`CrossAttention.forward`、`PositionalEncodingSinCos.forward`、`TextAdaptiveLayer.forward`、`CADDecoderLayer.forward`，这些核心函数的可执行 AST 相同。`CADSequenceEmbedder.forward` 仅删未使用的局部赋值，`CADDecoder.forward` 和 `Text2CAD.forward` 仅有类型注解差异。设备、构造及导入改动逐处记录在 `text2cad_port/NOTICE.md`。这支持前向数学保留，但不能证明 NPU 算子数值等价。
- Codex 独立构造不含真实 BERT 的模型：其非 BERT 键集、形状、dtype 与 172 键 manifest 全等。真实 checkpoint 经 SHA256 `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598` 校验后，172/172 子模块 `strict=True` 加载，加载后的张量与源张量逐个 `torch.equal`。两个全零张量恰为第 0、1 解码层未使用的 `norm_3.bias`，对应 `weight` 仍为初始化全 1；其他 170 张量非零。
- Codex 重新运行 `D:\miniconda\python.exe -m pytest experiments/tests/test_text2cad_model_adapter.py experiments/tests/test_text2cad_vector_scorer.py -q`：**101 passed，exit 0**。测试覆盖 mask、双头输出、shift、合成 batch、一致性及资产拒绝路径。Kimi 初版合成测试把 272 行直接送入期望 271 行 teacher forcing 的入口，已由实现者修正并复测通过。
- Codex 重新运行 `D:\miniconda\python.exe experiments/scripts/audit_text2cad_model_load.py --checkpoint .cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth --bert-dir .cache/bert_large_uncased --expected-keys experiments/configs/text2cad_checkpoint_expected_keys_20260930.json`：**exit 0**，状态 `structure_only_no_forward`，172/172、numeric equality、BERT 五文件 SHA 均通过。本地 Python 无 `transformers`/`safetensors`，所以完整 BERT 加载留待服务器以本次审核提交重跑；此前 Codex 独立服务器零前向审计记录为 391/391 BERT 张量加载、仅 7 个 `cls.*` 未使用、词嵌入与源 safetensors 逐位相等，见 `experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md`。
- `text2cad_port` 的代码导入未含 OCC 或 CUDA 设备字面量；移植 LICENSE 与 NOTICE 说明了官方 CC BY-NC-SA 4.0 来源。审查了加载器的文件 SHA、结构、键、形状、意外零张量和加载后数值门。B2 的正式入口**必须传入已提交的 expected-keys manifest 路径**；加载器当前把该参数设为可选，不能让正式 runner 省略。

## B2 与 G1 保留条件

1. 实现固定、UID-free 的 query/gallery 隐式笛卡尔积与分片；评分 worker 只接触当前文本和官方 CAD 向量，不接收 qrels 或原 CSV。结果逐对可复核、可恢复，批次与两卡分片不能漏算或重复。
2. 动态裁剪尾部 padding、冻结 BERT 查询缓存及任何候选侧缓存，均先用**同一真实 pair**与全长、逐对参考的有效 logits 和最终分数比较；未通过则退回参考路径。当前端到端合成测试不够证明真实模型等价。
3. 服务器在 checkout 外保存零前向完整 BERT/模型加载审计，远端 Git 必须干净，检出与本审查对应的完整 SHA。该审计不计真实前向预算。
4. B2 代码、CPU 测试、命令和资源边界由 Codex 再审；只有随后在 `ACTIVE_TASK.md` 写明完整批准 SHA、1 对或更小的明确 pair 上限及卡时，才可进入 G1 真实前向。七天门以两卡长度加权端到端实测估计，含至少 30% 余量；下界约 63 对/秒。

当前没有真实 Text2CAD logits、吞吐、检索指标。Kimi 对全套测试的 4 failed/9 errors 来自本机缺 `transformers`/`pyarrow` 的既有测试模块，本审查不以其声称的 307 passed 代替相关测试证据。
