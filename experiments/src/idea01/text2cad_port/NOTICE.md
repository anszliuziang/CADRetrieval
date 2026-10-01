# NOTICE — Text2CAD 前向移植的来源与逐处改动说明

本目录（`experiments/src/idea01/text2cad_port/`）是对上游 **Text2CAD**（Mohammad Sadil Khan et al., NeurIPS 2024, DFKI GmbH）冻结前向架构的改编移植。上游代码来自本地只读镜像 `.cache/Text2CAD-official`，Git SHA **`669a81472ede2269837c22a4d00070eaac360c82`**，许可 **CC BY-NC-SA 4.0**（完整文本见本目录 `LICENSE`，为上游原文件的逐字节副本）。按 ShareAlike 要求，本目录中的移植文件同样以 CC BY-NC-SA 4.0 提供。移植目的：在 Ascend 910 / CPU 上以**与官方训练完全一致的 teacher-forcing 前向**计算冻结模型的两路原始 logits，用于文本到 CAD 检索评分；不包含自回归生成、训练循环或几何后处理。

下游任务合同：`experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`（G0.2）与 `experiments/reviews/IDEA01_M1C_STAGE_A_CODEX_REVIEW_2026-09-30.md`（Stage B 准入）。

## 为什么需要移植（不改变数学的两个阻塞点）

1. `Cad_VLM/models/decoder.py:7-13` 与 `Cad_VLM/models/layers/adaptive_layer.py:13` 的顶层导入会拉入 `CadSeqProc.utility.utils` / `Cad_VLM.models.layers.utils_decode`，前者在顶层 `import OCC.*`（pythonocc-core）。teacher-forcing 前向不使用这些几何函数；在服务器安装 OCC 被任务明确禁止。
2. `Cad_VLM/models/layers/text_embed.py:24,42`、`Cad_VLM/models/decoder.py:162` 等处以 `torch.cuda.is_available()` 和 `.to("cuda")` 字面量选择设备；在 CPU-only 与 Ascend 910（torch_npu、无 CUDA）主机上都会崩溃或落错设备。

## 逐文件改动清单（上游行号基于上述 pinned SHA）

### `model.py`

| 上游来源 | 移植内容 | 改动 |
| --- | --- | --- |
| `CadSeqProc/utility/macro.py:6-14, 40-44` | `MAX_CAD_SEQUENCE_LENGTH`、`CAD_CLASS_INFO` 常量 | 仅取值，无逻辑 |
| `CadSeqProc/utility/utils.py:1790-1803` | `generate_attention_mask`（含 `mask[1:,0] = -inf` 的 START 列屏蔽） | **逐字复制**，使其可用而不导入 OCC 链 |
| `Cad_VLM/models/layers/text_embed.py:8-18` | `prepare_cross_attention_mask_batch` | **逐字复制**（从 text_embed.py 移到本文件，避免 model.py 依赖 transformers） |
| `Cad_VLM/models/layers/embedder.py:11-33` | `PositionalEncodingSinCos` | 逻辑逐字；上游调用方用 `torch.cuda.is_available()` 选 device（decoder.py:162），移植后 device 为**显式必选参数** |
| `Cad_VLM/models/layers/embedder.py:58-64` | `Embedder` | 逐字 |
| `Cad_VLM/models/layers/embedder.py:67-103` | `CADSequenceEmbedder` | 逻辑逐字；删除一行被注释掉的死代码（原 L82）；删除 `loguru` 与 `sys.path` 操作导入 |
| `Cad_VLM/models/layers/functional.py:9-283` | `multi_head_attention_forward` | **逐字复制**（含 `torch.equal(query, key)` 分支选择——自注意力融合 GEMM 与编码器-解码器分支在末位 ulp 上可区分，剪枝会改变冻结模型数学） |
| `Cad_VLM/models/layers/functional.py:286-309` | `FeedForwardLayer` | 逐字 |
| `Cad_VLM/models/layers/attention.py:15-218` | `MultiHeadAttention` | 逻辑逐字；删除未用导入（`itemgetter`、`layer_utils` 星导入、`os/sys` 路径操作）；保留 `__constants__`、`__setstate__`、`use_3d` 约定 |
| `Cad_VLM/models/layers/attention.py:220-272` | `CrossAttention` | 逻辑逐字；**刻意保留** `forward` 不把 `key_padding_mask` 传给内部 MHA 的上游行为（文本 padding 仅由 `prepare_cross_attention_mask_batch` 的加性 attn_mask 抑制；交叉注意力 `num_heads=1` 使 3D mask 形状检查 `B*num_heads==B` 成立） |
| `Cad_VLM/models/decoder.py:294-421` | `CADDecoderLayer` | 前向逐字；构造函数 `device` 参数改为必选显式值（原默认 `"cuda"`）；删除 `exit()` 调试残行（原 L385）与 rich 导入。保留 `norm_3`/`dropout_2` 对所有层创建但仅在 `use_ca` 时使用——这是 checkpoint 两个全零张量（`layers.0/1.norm_seq.norm_3.bias`）的成因 |
| `Cad_VLM/models/decoder.py:22-164` | `CADDecoder`（`__init__`/`forward`/`from_config`） | 前向逐字；`from_config` 的 `device="cuda" if torch.cuda.is_available() else "cpu"`（原 L162）改为必选显式参数；层堆栈用直接逐层构造替代 `copy.deepcopy`（原 L82-97；对新构造对象 deepcopy 结构完全等价，仅省一次拷贝）；`count_parameters`/prettytable/rich 未移植 |
| `Cad_VLM/models/decoder.py:166-282` | `CADDecoder.decode`（自回归生成） | **未移植**：teacher-forcing 评分不调用；其依赖 `create_flag_vec`/`create_index_vec`/`top_p_sampling` 位于 OCC 链 |
| `Cad_VLM/models/layers/adaptive_layer.py:18-109` | `TextAdaptiveLayer` | 前向逐字；删除顶层 `utils_decode`（OCC 链，仅上游 `__main__` 演示用）与 `models.utils`（prettytable）导入；`total_parameters` 未移植 |
| `Cad_VLM/models/text2cad.py:14-127` | `Text2CAD` | `forward` 逐字；构造函数改为**接收预建子模块**（`base_text_embedder`/`adaptive_layer`/`cad_decoder` 属性名不变，state_dict 路径与上游一致），组合根移至 `loader.build_text2cad`；`test_decode`（原 L35-69，生成路径）未移植；`count_parameters` 未移植；`get_trainable_state_dict` 保留。**Stage B2 追加（2026-09-30，待 Codex 重审）**：`forward` 新增可选参数 `cad_seq_len`，默认 `None` 时使用构造值 271，与已审 B1 路径逐位一致；仅尾部 padding 裁剪门在显式传入 `L-1` 时使用，且输入三向量、`key_padding_mask` 与官方 `attn_mask` 均已同步切到 `L`/`L-1`。越界值（<1 或 >271）以 `bad_cad_seq_len` 拒绝 |
| `Cad_VLM/train.py:80` 等 | `OFFICIAL_INFERENCE_CONFIG` | 值取自 `Cad_VLM/config/inference.yaml`；`cad_seq_len` 按 train.py:80 / test.py:76 / test_user_input.py:76 注入的 `MAX_CAD_SEQUENCE_LENGTH` 镜像；`text_embedder.cache_dir`（"YOUR DIR"）弃用 |
| `Cad_VLM/train.py:257-270, 273-278` | `build_teacher_forcing_inputs` / `teacher_forced_logits` | 新增函数，精确复刻官方 teacher-forcing 切片与调用（输入 `[:, :-1]`、目标 `[:, 1:].clone()`、shifted mask、`attn_mask` 不切、`metadata=False`）；差异：**不就地修改**调用方字典（官方原地重绑定；切片视图与克隆语义一致）。**Stage B2 追加**：`teacher_forced_logits` 新增可选 `cad_seq_len` 透传参数，默认 `None` 与已审路径一致 |

### `text_embed.py`

| 上游来源 | 改动 |
| --- | --- |
| `Cad_VLM/models/layers/text_embed.py:21-30`（`__init__`） | `BertTokenizer`/`BertModel.from_pretrained("google-bert/bert-large-uncased", cache_dir=...)` 改为 `from_local()`：`from_pretrained(<本地 pinned 目录>, local_files_only=True)`；`torch.cuda.is_available()` 设备选择（L24）改为显式构造参数；冗余的 `max_position_embeddings=max_seq_len`（L28-29，等于 pinned config 默认 512）移除，改由加载器断言 config 值；加载后 `eval()` + `requires_grad_(False)`（与 train.py:86-87 冻结一致，前向数学不变）；保留 `output_loading_info` 供严格门控 |
| `Cad_VLM/models/layers/text_embed.py:32-50`（`get_embedding`） | 硬编码 `.to("cuda")`（L42）改为 `.to(self.device)`；`truncation=True`（静默截断，L40）改为 `truncation=False` + 逐输入长度门：任何 >`max_seq_len`(512) BERT token 的输入以 `text_too_long` **拒绝**（任务停止条件禁止静默截断）。对所有被接受的输入，tokenize 结果与上游逐位一致（同 tokenizer、同 special tokens、同右 padding；上限内截断本不触发）。返回 `last_hidden_state` 全序列与 `attention_mask == 0` 的语义不变 |

### `loader.py`（新增，项目胶水，无上游对应）

实现任务合同 G0.2/G0.4 与 Stage B 准入的严格加载门：资产 SHA256 校验 → checkpoint 结构（顶层恰 `{"model_state_dict"}`、全部 `module.` 前缀、无 BERT 张量）→ 键集/形状双向相等（模型↔checkpoint↔独立 manifest 三方）→ 零张量白名单（仅 `cad_decoder.cad_decoder_layers.{0,1}.norm_seq.norm_3.bias`，证据见该文件 docstring 与 `experiments/runs/m1c_structure_20260930/`）→ 子模块级 `load_state_dict(strict=True)`（**绝不**对整个模型 `strict=False`）→ 逐张量 `torch.equal` 数值相等。BERT：五文件 SHA、`from_pretrained(local_files_only=True)` 的 missing/mismatched/error 为空、unexpected 仅限 `cls.*`、391/391 张量、词嵌入与 safetensors 逐位相等、`max_position_embeddings == 512`。本文件任何位置都不执行前向。

## 已知有意不移植项

自回归生成（`decode`/`test_decode` 及其 `create_flag_vec`/`create_index_vec`/`top_p_sampling`）、训练循环与损失、OCC 几何重建、attention score 可视化的参数打印工具。teacher-forcing 评分均不需要；`flag_vec`/`index_vec` 在评分路径直接来自官方数据文件（Stage A 审计 A3）。
