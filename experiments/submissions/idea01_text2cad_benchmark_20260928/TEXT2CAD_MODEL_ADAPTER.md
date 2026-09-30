# 原版 Text2CAD 冻结前向架构移植与严格加载器（M1c Stage B1）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 子计划（本合同）： `experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`（G0.2/G0.4 的 Stage B1 部分）
- Stage A 审查： `experiments/reviews/IDEA01_M1C_STAGE_A_CODEX_REVIEW_2026-09-30.md`（Stage B 准入要求）
- 前置审计： `TEXT2CAD_SOURCE_AUDIT.md`（同目录）、`experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md`
- 本文件作者： Claude Code / Kimi K3（本地实现者），2026-09-30
- **本轮边界**：只做官方冻结前向架构的本地移植、严格加载器、CPU 合成测试与零前向真实权重结构/键/数值审计。**没有 SSH、没有 NPU、没有真实模型前向（CPU/NPU 均为 0）、没有下载资产、没有修改服务器、没有 git commit/push、没有修改 Codex-owned 文件（`ACTIVE_TASK.md`/`plans/`/`reviews/`/`CLAUDE.md`）、没有改动已接受的 Stage A scorer。** 正式 pair runner、批量 benchmark 属 B2/G2，本轮未写。
- 新增工作区文件（未提交）：
  - `experiments/src/idea01/text2cad_port/`：`model.py`（torch-only 架构）、`text_embed.py`（适配的 TextEmbedder）、`loader.py`（严格加载器）、`__init__.py`、`NOTICE.md`（逐处改动说明）、`LICENSE`（上游 CC BY-NC-SA 4.0 原文副本）
  - `experiments/scripts/dump_text2cad_expected_keys.py`（manifest 生成脚本）与 `experiments/configs/text2cad_checkpoint_expected_keys_20260930.json`（172 键独立 manifest）
  - `experiments/scripts/audit_text2cad_model_load.py`（零前向严格加载审计脚本）
  - `experiments/tests/test_text2cad_model_adapter.py`（56 项 CPU 合成测试）
  - 本文件

## 1. 来源、许可与移植范围

上游：Text2CAD（Mohammad Sadil Khan et al., NeurIPS 2024, DFKI GmbH），本地只读镜像 `.cache/Text2CAD-official`，Git SHA `669a81472ede2269837c22a4d00070eaac360c82`，许可 CC BY-NC-SA 4.0。按 ShareAlike 要求，移植目录内置上游 `LICENSE` 原文副本，逐文件标注出处，`NOTICE.md` 给出**逐处改动清单**（含上游行号）。移植只覆盖 teacher-forcing 前向：自注意力/交叉注意力/FFN、sin-cos 位置编码、因果 mask 的 START 列屏蔽、双头 logits、`Text2CAD.forward`、train.py:257-270 的输入切片。**有意不移植**：自回归 `decode()`/`test_decode()`（评分不需要，且会拖入 OCC 链的 `create_flag_vec`/`create_index_vec`/`top_p_sampling`）、训练循环、OCC 几何重建、参数打印工具。

两个必须适配的上游阻塞点（Stage A 审查已定位）：`decoder.py`/`adaptive_layer.py` 顶层导入经 `CadSeqProc.utility.utils`/`utils_decode` 拉入 OCC（服务器禁止安装）；`text_embed.py:24,42`、`decoder.py:162` 以 CUDA 可用性/字面量选设备（910 无 CUDA）。移植将所需三个辅助函数（`generate_attention_mask`、`prepare_cross_attention_mask_batch`，均逐字复制）与全部前向类收进 OCC-free 的 `model.py`，所有设备选择改为**显式参数**；源码卫生测试锁定不再出现 CUDA 字面量/OCC 导入。注意力数学未用通用 Transformer 替换：`multi_head_attention_forward` **逐字复制**（含 `torch.equal` 的融合/拆分 GEMM 分支选择——两分支末位 ulp 可区分），`CrossAttention` 刻意保留「key_padding_mask 不传给内部 MHA」的上游行为（文本 padding 仅由加性 attn_mask 抑制；交叉注意力 `num_heads=1` 使 3D mask 形状检查成立）。

**刻意保留并复核的官方语义**（改变任一都会改变冻结模型条件分布）：自注意力 key_padding_mask 为 `all(官方mask, axis=2)`——只有纯 `[0,0]` 行被屏蔽，`[v,0]` 行照常参与注意力；`attn_mask[1:,0] = -inf` 使 START 列对所有后续位置不可见；嵌入为 `sf(flag)+si(index)+cx(x)+cy(y)` 四表求和且 PAD 位置 x/y 冗余清零；`ca_level_start=2`（第 0-1 层仅自注意力）；每层都有 downsampler 参数（Linear 1024→256）；adaptive layer 配置 in=out=1024 **无** downsampler；位置编码是普通属性而非 buffer（checkpoint 无 PE 键）。

## 2. 键映射与零张量证据（零前向审计）

checkpoint `Text2CAD_1.0.pth`（SHA256 `235b…4598` 本地复核一致）：顶层恰 `{"model_state_dict"}`，172 个张量全部带 `module.` 前缀、无 BERT 键、全部 float32。剥前缀后键空间 = `adaptive_layer.*`(12) + `cad_decoder.cad_decoder_layers.*`(152) + `cad_decoder.seq_embed.*`(4) + `cad_decoder.seq_output_x/y.*`(4) = **172**，与移植模型非 BERT state_dict 键集**精确相等**（`text2cad.py:123-127 get_trainable_state_dict` 同款过滤，测试锁定）。

**170/172 非零的成因已定位**：两个全零张量是 `cad_decoder.cad_decoder_layers.{0,1}.norm_seq.norm_3.bias`。`CADDecoderLayer` 对每层都创建 `norm_3`，但仅在 `use_ca`（第 2-7 层）的前向分支内使用；第 0、1 层无交叉注意力，`norm_3.bias` 从未获得梯度，保持 LayerNorm 初始值 0。旁证：对应 `norm_3.weight` 在 checkpoint 中仍为全 1（未训练的初始化值），而已训练的第 2 层同位置权重均值≈0.135、偏置绝对值最大≈0.712。因此这**不是资产异常**，加载器把这两个键写成白名单 `EXPECTED_ZERO_TENSOR_KEYS`：任何**其它**全零张量拒绝（`unexpected_zero_tensor`），这两个键若不为零同样拒绝（`expected_zero_violation`）。

零前向严格加载审计（真实权重，CPU，`weights_only=True`，**未执行任何前向**）：

```
PYTHONDONTWRITEBYTECODE=1 D:\miniconda\python.exe experiments/scripts/audit_text2cad_model_load.py \
  --checkpoint .cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth \
  --bert-dir .cache/bert_large_uncased \
  --expected-keys experiments/configs/text2cad_checkpoint_expected_keys_20260930.json
```

退出码 0，stderr 为空，原始输出存于 `experiments/runs/m1c_model_load_20260930/{stdout.json,stderr.log}`（Git 忽略目录）。要点：checkpoint SHA 一致；172 张量经「模型↔checkpoint↔独立 manifest」三方键集/形状相等后，按子模块 `load_state_dict(strict=True)` 加载（**绝不**整模型 `strict=False`），随后逐张量 `torch.equal` 数值相等；零张量恰为白名单两键；BERT 五文件 SHA 全部一致；本机无 transformers/safetensors，BERT 加载门（391/391、`cls.*` 之外无 unexpected、词嵌入逐位相等）标注为「已有 2026-09-30 服务器零前向证据，G1 须以获批 SHA 在云脑重跑本脚本」。manifest 由 `experiments/scripts/dump_text2cad_expected_keys.py` 对真实 checkpoint 结构转储生成（含 provenance）。

## 3. 严格加载器设计（`loader.py`）

门顺序（任一不过即 `AssetVerificationError`，带稳定 reason，全部在触碰参数之前）：① checkpoint 文件 SHA256；② 结构（顶层恰 `{"model_state_dict"}`、全部 `module.` 前缀、拒绝任何 `base_text_embedder.*` 键——BERT 只能来自 pinned 基座）；③ 键集双向相等（missing/unexpected 全量列出）；④ 逐键形状相等；⑤ 可选独立 manifest 三方相等（键、形状、dtype）；⑥ 零张量白名单（双向）；⑦ 子模块级 strict load；⑧ 逐张量数值相等。BERT 路径：五文件 SHA → `from_pretrained(pinned_dir, local_files_only=True, output_loading_info=True)` → missing/mismatched/error 为空、unexpected 全在 `cls.*`、张量数 391、词嵌入与 safetensors `torch.equal`、`max_position_embeddings==512`。组装入口 `load_frozen_text2cad(...)` 返回 `(model, report)`：eval()、全参数冻结（BERT 冻结同 train.py:86-87；requires_grad 标志不改变前向数学）、显式 device。**该模块任何位置都不执行前向**。

tokenizer 截断门（Stage B 准入要求）：适配的 `TextEmbedder.get_embedding` 以 `truncation=False` 调 tokenizer，用 `attention_mask` 逐行求和得到真实 token 数，任何输入 >512 BERT token 以 `text_too_long` 拒绝（官方是静默截断，任务停止条件禁止）。对所有被接受输入，tokenize 结果与上游逐位一致（同 tokenizer、同 special tokens、同右 padding）。已审 3,644 种查询最长 98 token（Codex 服务器审计），门为防御将来查询更换。

## 4. 与官方的差异汇总（功能维度）

| 维度 | 官方 | 本移植 | 对数学的影响 |
| --- | --- | --- | --- |
| OCC 导入 | decoder/adaptive_layer 顶层拉入 OCC 链 | 所需 3 个辅助函数逐字复制到 `model.py`，不导入 OCC | 无（同一代码） |
| 设备 | `torch.cuda.is_available()`/`.to("cuda")` 字面量 | 全链路显式 device 参数 | 无（设备选择与算子语义无关） |
| BERT 加载 | hub id + `cache_dir`，`truncation=True` 静默截断 | pinned 本地目录 `local_files_only`，`truncation=False` + >512 拒绝 | 接受输入上逐位一致 |
| checkpoint 加载 | 剥 `module.` 前缀 + `strict=False` | SHA/结构/键/形状/零张量/数值全门 + 子模块 strict | 更严，加载结果相同 |
| 组合方式 | `Text2CAD(text_config, cad_config)` 内部建 TextEmbedder | 构造器接收预建子模块，属性名不变 | 无（state_dict 路径一致） |
| 生成路径 | `decode()`/`test_decode()` | 未移植 | teacher-forcing 不经过 |
| 输入切片 | train.py 原地重绑定 | `build_teacher_forcing_inputs` 返回新字典（视图+clone，不污染调用方） | 无（值与切片语义一致） |

行级差异见 `experiments/src/idea01/text2cad_port/NOTICE.md`。

## 5. 测试与运行记录

环境：本机 Windows，`D:\miniconda\python.exe` = Python 3.13.5、torch 2.12.0+cpu、pytest 9.1.1（无 transformers/safetensors/pyarrow）。

命令：`PYTHONDONTWRITEBYTECODE=1 /d/miniconda/python.exe -m pytest experiments/tests/test_text2cad_model_adapter.py -q`
结果：**56 passed, 0 failed**（约 10s）。非镜像验证点：

- **mask 语义**：因果 + START 列 0/-inf 模式逐元素断言；交叉注意力加性 mask 的 B=1/B>1 分支等价；`[v,0]` 行参与注意力（改动后后续位置 logits 变）、纯 `[0,0]` 行不可见（塞垃圾后真实位置 logits **逐位**不变）、位置 0 扰动不改变 ≥1 位置 logits（START 列屏蔽端到端**逐位**成立）；尾部 padding 对真实位置不可见（逐位）。
- **注意力数值核心**：ported MHA 对手写 float64 参考（逐头 QK^T/√d + mask + softmax @ V + out proj），自注意力与编码器-解码器两分支均 1e-4 内一致；masked 位置注意力权重**精确为 0**；3D mask 折叠与形状检查；`CrossAttention` 的 key_padding_mask 不入内层 MHA 的怪癖被锁定为逐位等价。
- **架构/键**：移植模型非 BERT 键集、形状与 172 键 manifest 全等；`get_trainable_state_dict` 键集一致；PE 非持久化；`use_ca`、交叉注意力 1 头、每层 downsampler、adaptive 无 downsampler、嵌入表尺寸 267/267/12/11。
- **双头**：输出 (B,271,2,267)；扰动 `seq_output_x` 仅头 0 变、头 1 **逐位**不变；eval 两次前向逐位一致、train 模式 dropout 使输出不同。
- **teacher forcing 胶水**：切片与 train.py:257-270 逐项一致、目标为 clone、输入字典不被修改、`target != 0 == ~shifted_kpm` 与官方 loss 的 mask 对齐。
- **TextEmbedder（假件）**：显式 device 路由、tokenizer 调用契约（`padding=True`、`truncation=False`）、>512 拒绝且 512 边界通过、裸字符串包装、未加载拒绝；`from_local` 经注入假 transformers 模块验证 `local_files_only=True`、loading_info 留存、eval+冻结。
- **严格加载器**：合成 checkpoint 全门往返 + 12 条拒绝路径（SHA、缺文件、顶层结构、无前缀键、checkpoint 内 BERT 张量、缺键、多键、形状、意外零张量、预期零不为零、manifest 漂移、BERT 文件 SHA/缺失）。
- **源码卫生**：移植文件无 CUDA 字面量、model.py 无 transformers 导入、全包无 OCC 导入行。
- **与 Stage A scorer 集成**（随机权重）：合成合法 272 候选（circle+extrusion，过 `validate_cad_vec`）经 `teacher_forced_logits` → `two_head_factorized_score`，`n_valid_x/y` 与诊断一致；**batch 一致性**：同一 (text, candidate) 单独评分与在混合 batch 中评分，logits 1e-4 容差一致、`score_total` 三位小数一致；不同 query 分数显著不同。

回归：`pytest experiments/tests -q` → **307 passed**（Stage A 251 + 新增 56）、4 failed + 9 error 为 Stage A 已登记的既有环境性失败（`test_cadmium_scorer.py::TestRealTokenizerCPU` 4 项缺 transformers、`test_m0_manifest.py` 9 项缺 pyarrow），与本次新增文件无关；本次改动为纯新增，未触碰任何既有文件。

## 6. 剩余风险与未验证项（如实登记）

1. **真实模型前向零次**：上述全部正确性证据是「官方语义的合成复现 + 真实权重的结构/键/数值审计」。真实 logits 层面的 batch=1 vs B 等价、终止/padding 行为须待 G1（Codex 审码 + `ACTIVE_TASK.md` 完整 SHA/预算）后验证。
2. **BERT 加载门未在本机执行**：本机无 transformers；`verify_bert_model_load` 的逻辑由假件单测覆盖，真实 BERT 的 391/391/`cls.*`/词嵌入证据来自 Codex 2026-09-30 服务器审计；G1 须在云脑以获批 SHA 重跑 `audit_text2cad_model_load.py`（脚本已内置该路径）。
3. **NPU 适配未验证**：移植消灭了 CUDA 硬编码（显式 device 贯通，源码卫生测试锁定），但 torch_npu 上的 dtype 策略（是否 FP32 前向、`attn_mask` -inf 加性 mask 在 BF16 下的行为）属 G1 决策项；评分核已在 Stage A 固化 FP32 log_softmax，前向 dtype 须在 G1 与 CPU 参考对照。
4. **transformers 版本指纹**：官方 environment.yml 未钉 transformers 版；云脑环境为 4.53.1。`from_local` 行为（fast/slow tokenizer 选择、`output_loading_info` 字段）以云脑实测为准。
5. **假件测试的边界**：TextEmbedder 的设备路由/长度门用假件验证；真实 BatchEncoding 的 `.to(device)` 行为与 fake 一致属常规假设，G1 真实加载时复核。
6. `EXPECTED_ZERO_TENSOR_KEYS` 白名单与 manifest 绑定 pinned checkpoint SHA；若未来换 checkpoint，两处须同步重审。

## 7. 停止条件自查（对照 `ACTIVE_TASK.md`）

- checkpoint 非 BERT 键无法严格对应 → 172/172 三方相等 + 数值逐位相等，两个全零张量成因定位并白名单化；未触发。
- 官方数据 CAD 向量/有效 mask 不自洽 → 本轮未新增数据扫描（Stage A/Codex 已 8,046/8,046 覆盖）；移植 mask 语义与官方逐字一致并有端到端测试；未触发。
- 文本或 CAD 被静默截断 → 文本侧 `truncation=False` + >512 硬拒绝（测试含边界 512 通过/513 拒绝）；CAD 侧沿用 Stage A 核心；未触发。
- qrels 泄漏 → 移植包与加载器无 qrels/UID/文件语义输入（loader 只读资产路径）；scorer 未改动；未触发。
- 910 移植改变数学 → 逐字复制注意力/嵌入/mask，设备改显式参数；合成层面双头、因果、padding、batch 行为均有测试；真实 NPU 前向未授权未执行；未触发。
- 资产 SHA/许可不明 → checkpoint 与 BERT 五文件 SHA 复核一致；CC BY-NC-SA 4.0 LICENSE 副本 + NOTICE 改动清单已入移植目录；未触发。
- 远端 Git 不干净 → 本轮无 SSH；未触发。

未发现无法解释的关键不一致（172 键、两个零张量、mask/shift 语义均有源码+数据双重证据），故**未创建 BLOCKERS.md**。

`review_required`
