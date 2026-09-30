# Text2CAD 官方实现静态精读与评分定义（M1c 阶段 A）

- `task_id`: `idea01_text2cad_benchmark_20260928`
- 总计划： `experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md`
- 子计划（本合同）： `experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`（G0.1–G0.4 的静态部分）
- 当前 `experiments/ACTIVE_TASK.md` 状态（2026-09-30 读取）： `state: implementation`，`approved_sha: 无`，`gpu_budget: 0`，`max_scored_pairs: 0`，真实模型前向预算 NPU/CPU 均为 0
- 本文件作者： Claude Code / Kimi K3（本地实现者）
- 本轮边界： **只做静态源码精读、真实数据只读抽验、无权重 CPU 数学核心与合成 logits 测试。** 没有 SSH、没有加载任何模型权重、没有真实模型前向、没有 NPU、没有下载新资产、没有修改 `.cache` 官方文件、没有修改 Codex-owned 文件（`ACTIVE_TASK.md`/`plans/`/`reviews/`/`CLAUDE.md`）、没有 git commit/push。checkpoint 键结构审计、runner、pair 构建器、测速器属后续阶段，本文只在 §9 列出接口要求。
- 新增工作区文件（未提交）：
  - `experiments/src/idea01/text2cad_vector_scorer.py`（CPU 数值评分核心）
  - `experiments/tests/test_text2cad_vector_scorer.py`（45 项合成测试）
  - 本文件

## 1. 审计对象与资产指纹（全部本地只读复核）

| 资产 | 本地路径 | 复核结果 |
| --- | --- | --- |
| 官方源码镜像 | `.cache/Text2CAD-official` | `git rev-parse HEAD` = `669a81472ede2269837c22a4d00070eaac360c82`，与 `ACTIVE_TASK.md` pinned 一致；许可 `LICENSE` 为 CC BY-NC-SA 4.0 |
| 官方 checkpoint | `.cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth`（91,601,866 B） | 文件级 SHA256 = `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598`，与 pinned **一致**（仅哈希，未加载权重） |
| 官方 CAD 向量 | `.cache/text2cad_official/cad_seq.zip`（606,626,050 B，176,020 个 `.pth`） | ZIP SHA256 = `9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa`，与 pinned **一致** |
| 官方 split | `.cache/text2cad_official/text2cad_v1.0/train_test_val.json` | train 159,049 / validation 8,925 / test 8,046，与 M0 冻结数一致 |
| BERT 权重 | `.cache/bert_large_uncased/model.safetensors` | SHA256 = `28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4` |
| BERT 配置 | `.cache/bert_large_uncased/config.json` | `BertForMaskedLM`，hidden 1024，24 层，`max_position_embeddings` 512，vocab 30,522；同目录有 `tokenizer.json`/`tokenizer_config.json`/`vocab.txt` |

**文档勘误（不影响本阶段，提请 Codex 修订计划文件）**：子计划 `IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md` 第 7 行所记 BERT 权重 SHA256 为 `…f444a4c`（63 个十六进制字符），本地实算为 `…f444a4c4`（64 字符），前 63 位逐字一致，疑为转写漏末位。本阶段以实算值为准并记录于此；`ACTIVE_TASK.md` 未含该值。

官方运行环境（`environment.yml`）：Python 3.9、pytorch 2.2.1 + CUDA、pythonocc-core 7.7.2、transformers（pip，未钉版）。本机测试环境与之不同，见 §7。

## 2. CAD 向量与 mask 的官方构造（证据：源码行号 + 真实数据）

### 2.1 词表与常量（`CadSeqProc/utility/macro.py`）

- `N_BIT=8`（L6）；`END_TOKEN=["PADDING","START","END_SKETCH","END_FACE","END_LOOP","END_CURVE","END_EXTRUSION"]`（L8-9），即 PADDING=0、START=1、END_SKETCH=2、END_FACE=3、END_LOOP=4、END_CURVE=5、END_EXTRUSION=6；`END_PAD=7`、`BOOLEAN_PAD=4`（L11-12）；`MAX_CAD_SEQUENCE_LENGTH=272`（L14）。
- `CAD_CLASS_INFO`（L40-44）：`one_hot_size = 7+4+2**8 = 267`（两个输出头共享的 267 类词表：0-6 特殊 token、7-10 布尔值、11-266 量化坐标/参数）；`index_size = 11`；`flag_size = 12`。

### 2.2 每个 token 是两列整数 `[x, y]`（`CadSeqProc/cad_sequence.py::CADSequence.to_vec` L242-274）

- 首行固定 `[START, 0]` 即 `[1, 0]`（L244）；每个 sketch 后接其 extrusion；**序列以第二个 `[START, 0]` 收尾（L260）——没有独立 EOS，终止 token 就是 START**。`cad_vec` 转 `torch.int32`（L264）。
- padding（L266-270）：`num_pad = 272 - len`；`add_padding` 以 0 填充（`utils.py` L833-857，**`M < 0` 直接 `ValueError`**，故官方数据不存在超过 272 的序列）；padding 行 `flag=11`、`index=max(index)+1`。
- sketch 内部（`sketchsequence.py` L114-123、`face.py` L66-74、`loop.py` L130-138、`line.py` L57-68、`arc.py` L101-113、`circle.py` L113-125、`curve.py` L59-60）：
  - Line → `[sx+11, sy+11]` + `[5, 0]`；Arc → 起点/中点两个坐标 token + `[5, 0]`；Circle → 圆心/pt1 两个坐标 token + `[5, 0]`。**三种曲线的类型 token 都是 END_CURVE=5**，曲线类型靠 token 计数隐式区分（`loop.py::from_vec` L141-176：loop 内单曲线判 Circle、2 token 判 Line、3 token 判 Arc）；坐标 token 两列都 ≥11（两列都有效）；随后 END_LOOP=4、END_FACE=3、END_SKETCH=2 各一行，均为 `[t, 0]`。
- extrusion 11 个 token（`extrude_sequence.py` L211-248，注释 L247）：`e1,e2,ox,oy,oz,theta,phi,gamma` 与 `sketch_size` 为 `[v+11, 0]`；`boolean` 为 `[b+7, 0]`（只占 7..10，L234）；收尾 `END_EXTRUSION=6` 为 `[6, 0]`。**extrusion 全部 token 的第二列恒为 0**。
- `flag_vec`：START/sketch token 为 0；extrusion 段为 `[1] + list(range(1,11))`（L257，共 11 个值）；padding 为 11。`index_vec`：第 i 个 sketch-extrusion 段为 i；padding 为 `max+1`（L270）。

### 2.3 落盘格式（`CadSeqProc/json2vec.py` L254-263）

每个 UID 一个 `.../seq/<chunk>.pth`：
`vec = {cad_vec (272,2) int32, flag_vec (272,), index_vec (272,)}`；
`mask_cad_dict = {attn_mask, key_padding_mask}`，其中 **`key_padding_mask = cad_vec == 0`（逐 (位置, 列) 的布尔 mask，L262）**；`attn_mask = generate_attention_mask(272-1)`（L261），即 271×271 上三角因果 mask，且 **`mask[1:, 0] = -inf`——位置 0 的 START 对所有后续位置不可见**（`utils.py` L1790-1803，float32，0/-inf）。

### 2.4 真实数据抽验（只读，未写缓存）

命令（本机 `D:\miniconda\python.exe`，torch 2.12.0+cpu）：从 ZIP 直接 `zipfile` 读 `test[0] = 0077/00776997` 的 `.pth`（`weights_only=True` 可加载），并对 test 前 30 个 + 随机种子 0 抽 20 个做结构扫描。结果：

- 单样本深查：`cad_vec (272,2) int32`，值域 [0, 266]；`flag (272,)` 值 0-11；`index (272,)` 值 {0,1}；`key_padding_mask (272,2) bool`，**逐元素等于 `cad_vec == 0`**；`attn_mask (271,271) float32 ∈ {-inf, 0}`，`[1:,0] = -inf`。序列：`[1,0],[139,139],[139,266],[5,0],[4,0],[3,0],[2,0],[189,0],…,[17,0],[6,0],[1,0]`，未填充长度 19（1 个 circle + 1 次 extrusion），尾部 253 行 `[0,0]`，pad flag=11、pad index=1（=max+1）。
- 50 样本扫描：50/50 全部满足——mask 与 `cad_vec==0` 逐元素一致、首 token `[1,·]`、末真实 token `[1,0]`、pad flag=11、pad index=max+1、词表 ≤266、无零长填充；未填充长度 min/max/mean = 19/142/33.86。
- 该扫描只覆盖 50/8,046；全图库逐样本核验（含 272 全长无 padding 边界）属后续 runner 阶段的 `data_audit` 输出。

## 3. Teacher forcing、损失与输出头（训练侧证据）

### 3.1 Teacher forcing shift（`Cad_VLM/train.py` L247-287）

```
L257  shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]   # 目标位置的官方 mask（True=PAD）
L259  cad_vec_target           = vec_dict["cad_vec"][:, 1:].clone()         # 目标 = 位置 1..N-1
L264-265  for key,value in vec_dict.items(): vec_dict[key] = value[:, :-1]  # 输入 = 位置 0..N-2（cad/flag/index 同切）
L268-270  mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:, :-1]  # 输入 mask 同切（attn_mask 保持 271×271）
L273-278  cad_vec_pred, _ = model(...)   # 注释形状 (B, N1, 2, C1) = (B, 271, 2, 267)
L281-287  criterion({"pred": ..., "target": cad_vec_target, "key_padding_mask": ~shifted_key_padding_mask})
```

即：输入去尾、目标去头，标准 next-token teacher forcing；传给损失的 mask 是**官方 mask 取反**（True=有效）。`attn_mask` 原样来自数据文件。

### 3.2 两路 x/y 损失（`Cad_VLM/models/loss.py`）

- L13：`self.ce_cad = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)`。
- L28-34 / L35-41：对 x、y 两路分别计算 `sum(CE ⊙ mask) / sum(mask)`——**按 (位置, 头) 有效 mask 的逐头 masked mean**；`target[:, :, 0]`、`target[:, :, 1]` 是同一 `cad_vec` 的两列标签。
- L43：`loss_seq = (loss_seq_x + loss_seq_y) / 2`——**两头再取平均**。
- 结论：训练损失 = 每头「masked 平均的带 0.1 label smoothing 交叉熵」的双头均值。PyTorch label smoothing 使损失 = `(1-ε)·NLL + ε·(− 全部 267 类 log p 的均值)`，**不是 NLL**；masked mean + 双头平均也不是序列求和。**绝不能把该训练 loss 当作目标 NLL 或序列对数概率。**

### 3.3 输出头与嵌入（`Cad_VLM/models/decoder.py`、`text2cad.py`、`layers/embedder.py`）

- 输出头（decoder.py L100-105）：`seq_output_x = Linear(cdim→267)`、`seq_output_y = Linear(cdim→267)`，两个**相互独立的线性头**，共享同一 267 类词表；forward 返回 `cat([Sx, Sy], dim=2)` → `(B, 271, 2, 267)`（L139-142）。**是原始 logits（未过 softmax）。**
- token 嵌入（embedder.py L67-103）：`sf(flag) + si(index) + cx(x) + cy(y)` 四嵌入求和，x/y 各一张 267 表；L95-96 用 `~key_padding_mask` 把 PAD 位置的 x/y 值清零（对官方数据是冗余恒等操作，因 PAD 位置值本为 0）。位置编码为确定性强 sin/cos（embedder.py L19-27，全 271 长度叠加）。
- 自注意力 mask（decoder.py L378-381）：`key_padding_mask = all(官方mask, axis=2)`——**只有两列皆 PAD 的行（纯 `[0,0]`）才在自注意力中被屏蔽**；`[v,0]` 行照常参与注意力。
- decoder 结构（trainer.yaml + decoder.py）：cdim=256、8 层、8 头、dropout 0.1、`ca_level_start=2`（第 0-1 层仅自注意力，第 2-7 层含对文本的交叉注意力）；每层一个 `Linear(1024→256)` downsampler（decoder.py L336，所有层都创建参数）。
- 文本侧（text2cad.py L86-105）：BERT 全序列 `last_hidden_state`（不取 [CLS]）→ `TextAdaptiveLayer`（一层自注意力 + FFN；配置 in=out=1024，**无 downsampler**，adaptive_layer.py L49-51）→ 作为交叉注意力上下文进入 decoder。交叉注意力 mask 由文本 padding mask 展开（text_embed.py L8-18）。
- checkpoint 内容（text2cad.py L123-127）：`get_trainable_state_dict` 仅保存不含 `base_text_embedder` 的键（adaptive_layer.*、cad_decoder.*）；BERT 不存入 checkpoint，须从固定基础权重加载。官方推理脚本加载时剥 `module.` 前缀且 `strict=False`（test_user_input.py L146-157）——**我们的 runner 不允许照搬 `strict=False`**，须显式断言缺失键恰为 BERT 键、无意外多余键（阶段 B 的零前向键审计）。
- 训练侧 BERT 冻结（train.py L86-87）：`base_text_embedder` 全部参数 `requires_grad=False`。

### 3.4 文本预处理（`Cad_VLM/models/layers/text_embed.py` + `dataprep/t2c_dataset.py`）

- **进入 tokenizer 的是原始 CSV 文本**：`t2c_dataset.py::_prepare_data` L86-89 把四个 level 的原串存入 `level_data` 并返回；L93 的 `remove_substrings(...).lower()` 结果赋给局部变量 `prompt` 但**未被返回（官方代码死代码）**——训练/评测实际使用的文本未做该清洗。`test.py` L191-198 与 `test_user_input.py` 同样直接使用数据原文/用户输入。**结论：评分侧不得对查询文本做官方实际路径之外的清洗/小写化**（M0 协议规定的 `strip()` 已在冻结 query 清单时完成，属协议本身）。
- `TextEmbedder.__init__`（L21-30）：`BertTokenizer`/`BertModel.from_pretrained("google-bert/bert-large-uncased")`，`max_position_embeddings=max_seq_len`；`get_embedding`（L32-50）：`tokenizer(texts, max_length=512, truncation=True, padding=True)`（trainer.yaml L14 `max_seq_len: 512`）——**超过 512 BERT token 的文本会被静默截断**。uncased 小写化发生在 BERT tokenizer 内部，是模型固有行为。
- **硬编码 CUDA（已知必须适配点）**：L42 `input_ids = ... .to("cuda")` 无条件上 CUDA；L24 按 `torch.cuda.is_available()` 放模型。CPU-only 机器直接崩溃；在 Ascend 910（torch_npu、无 CUDA）上模型落 CPU 而输入试图上 CUDA，同样崩溃。**阶段 B 适配：改为显式 device（输入 `.to(next(self.model.parameters()).device)` 或构造参数传入），并核对输出语义不变。** 官方 `device="cuda"` 还散见 `test.py`/`validation_one_epoch`（train.py L430）等推理路径。
- 生成路径（decoder.py L166-282，仅供参考，评分不用）：从 `[[1,0]]` 起步自回归，greedy argmax（nucleus=0；t==1 时取 topk 候选做 hybrid 校验），每步 `create_flag_vec`/`create_index_vec` 确定性补 flag/index（utils.py L538-639），并用 `cad_vec[:,:,0] <= 6 或 flag>0` 的行把新 token 的 y 列强制置 0（L263-280）；循环固定跑满 `maxlen=272`，无早停。

## 4. 评分定义（精确定义，已实现于 `text2cad_vector_scorer.py`）

对每一对 `(q, C)`（文本查询 + 图库候选的官方 `cad_vec/mask`），在冻结模型上以与训练完全相同的 teacher forcing 前向一次，得到原始 logits `pred`（单候选形状 `(271, 2, 267)`；`pred[t, h, v]` = 位置 `t+1` 第 `h` 头对词表值 `v` 的 logit）：

```
target[t, h] = cad_vec[t+1, h]                      t = 0..N-2, h ∈ {x=0, y=1}
valid[t, h]  = (target[t, h] != 0)                  ≡ 官方 key_padding_mask 取反后切 [1:]
lp[t, h]     = log_softmax_FP32(pred[t, h, :])[target[t, h]]      # 仅取有效位置
score_x      = Σ_{t: valid[t,0]} lp[t, 0]           # FP32 逐 token 值，float64 累加
score_y      = Σ_{t: valid[t,1]} lp[t, 1]
S(q, C)      = score_x + score_y                    # 主分数（raw sum）
```

每对必须同时落盘：`score_x`、`score_y`、`score_total`、`n_valid_x`、`n_valid_y`、`n_valid_total`、`valid_seq_len`（=未填充长度−1）、`n_real_tokens`、`mean_x/mean_y/mean_total`（**仅诊断**，不作主指标）、`spec_version`、拒绝原因（若被拒）。

**逐 token 类型的有效头对照表**（证据见 §2-§3）：

| cad_vec 行 | x 标签 | y 标签 | x 头计分 | y 头计分 |
| --- | --- | --- | --- | --- |
| 初始 START `[1,0]`（t=0） | 1 | 0 | n/a（永不为 target） | n/a |
| 曲线坐标 `[x+11, y+11]` | 11..266 | 11..266 | ✓ | ✓ |
| END_CURVE `[5,0]` | 5 | 0 | ✓ | ✗ |
| END_LOOP `[4,0]` / END_FACE `[3,0]` / END_SKETCH `[2,0]` | 4/3/2 | 0 | ✓ | ✗ |
| extrusion 参数 `[v+11,0]`（e1,e2,ox,oy,oz,θ,φ,γ,s 共 9 行） | 11..266 | 0 | ✓ | ✗ |
| extrusion 布尔 `[b+7,0]` | 7..10 | 0 | ✓ | ✗ |
| END_EXTRUSION `[6,0]` | 6 | 0 | ✓ | ✗ |
| **终止 START `[1,0]`** | 1 | 0 | ✓ | ✗ |
| PADDING `[0,0]` | 0 | 0 | ✗ | ✗ |

**终止 token 处理**：无独立 EOS；最后一个真实 token 是 `[START,0]`，其 x 因子（标签 1）计入 `score_x`，y 因子被 mask。其后全部 PADDING 行双头不计分。**任何「最后一个有效 token 是 EOS」的假设都不成立，已在实现与测试中按 START 固化。**

**长度规则**：候选必须恰好为官方 padded 格式（`N ≤ 272`，runner 恒传 272）；核心拒绝 `N > 272`（官方数据不可能存在，`add_padding` 对负 padding 抛错）、`N < 2`、全空目标、非法标签（<0 或 >266）、非尾部 padding、`x=0,y≠0` 行、缺初始/终止 START、mask 与 `cad_vec==0` 不一致。文本侧 512 token 截断检查（BERT tokenizer）属阶段 B runner 的前置审计（本核心不加载 tokenizer）。

## 5. 该分数能否称为联合对数概率？——不能

定名：**two-head factorized conditional score（双头因子化条件对数概率求和排序分数）**，`spec_version = text2cad-vector-score-v1`。理由：

1. **架构即因子化**：两个独立线性头共享 267 类 softmax，`p(x_t, y_t | h_<t, q)` 被建为 `p_x(x_t|·)·p_y(y_t|·)` 的条件独立分解；即便接受该分解，逐位置「联合」也只是模型自定义的因子形式。
2. **监督范围之外无标签**：y 头在所有非坐标 token 上（占序列大部分）从未被训练（损失中 mask 掉），其「标签」是序列化的结构常数 0 而非随机变量；PADDING 行同理。把这些因子计入只会加入未训练/常数项，故按官方 mask 排除——但也因此它不是「完整序列的联合对数概率」。
3. **训练目标不是该和的 NLL**：训练用 `label_smoothing=0.1` 的 masked mean 再双头平均（§3.2），logits 并未按本分数校准。

因此报告口径：`S(q,C)` 是冻结模型在官方监督因子上的原始 logits 条件排序分数；可比较同 query 下不同候选的相对名次；**不得**表述为 CAD 几何的（边缘）似然、精确联合对数概率或 MLE 模型的 NLL。长度/复杂度偏置须靠 `n_valid_*`、`mean_*` 诊断与基线暴露。

## 6. 本地实现（阶段 A 交付）

`experiments/src/idea01/text2cad_vector_scorer.py`（纯 CPU，仅依赖 torch；无文件 I/O、无模型加载、不读 qrels/CSV/UID 映射）：

- 常量按官方 `macro.py` 复刻并注明出处与 CC BY-NC-SA 4.0 许可；
- `validate_cad_vec(cad_vec, key_padding_mask=None, flag_vec=None, index_vec=None)` → 结构诊断或带稳定 `reason` 的 `ScorerInputError`（§4 全部拒绝路径；显式 mask 必须等于 `cad_vec==0`，反转/移位 mask 均拒绝）。`flag/index` 仅检查形状、值域与 padding 值，不重建所有真实位置的确定性标记；runner 必须原样传入 pinned 官方向量；
- `two_head_factorized_score(logits, cad_vec, key_padding_mask=None)` → §4 公式；FP32 `log_softmax`（任意浮点输入先转 FP32），逐 token  gather 后 float64 累加；单样本/批量同构；绝不就地修改输入；
- `dedup_texts(texts)`（逐字相等去重，不做任何再归一化）与 `ScoreRequest` dataclass（输入模式恰为 `query_text/cad_vec/key_padding_mask/flag_vec/index_vec`，多字段即 TypeError）——供后续 pair 构建器复用，本阶段不接 runner。

## 7. 测试与运行记录

环境：本机 Windows，`D:\miniconda\python.exe` = Python 3.13.5、torch 2.12.0+cpu、numpy 2.3.4、pytest 9.1.1（与官方 environment.yml 不同；本核心只用 torch 张量运算，版本敏感性低，阶段 B 在服务器锁定环境复跑）。

命令：`PYTHONDONTWRITEBYTECODE=1 /d/miniconda/python.exe -m pytest experiments/tests/test_text2cad_vector_scorer.py -q`
结果：**45 passed, 0 failed**（1.80s）。覆盖：shift 与手算参考一致；均匀 logits 闭式 `-n·ln267`；主导 logits 分数≈0；终止 START「x 计分/y 屏蔽」双向扰动验证；mask 位置 logits 任意污染下分数逐位不变（含 PADDING 行）；显式 mask ≡ 派生 mask；272 带/不带 padding 边界；`N=273`、`N=1`、全 PAD、仅 START 拒绝；标签越界/负值/非尾 padding/`x=0,y≠0`/缺初始或终止 START/mask 逐位不符/**反转 mask 陷阱**/移位 mask 形状/NaN/Inf/词表 266/头维 3/logits 整数/步数不符/bool 或 float cad_vec/flag/index 越界与 padding 不符——全部按预期 reason 拒绝；batch 与单样本逐位一致；fp64/bf16 logits 统一按 FP32 核计分；随机批量对纯 Python float64 暴力参考（rtol 1e-4）；输入非变异；输出键集合精确且无 UID/qrels 字段；`ScoreRequest` 多余字段 TypeError；`dedup_texts` 逐字保序与坏文本拒绝。

真实数据集成核验（非测试套件，因依赖 `.cache`）：对 `0077/00776997` 真实 `vec/mask` 以合成 logits 评分——`n_valid_x=18`、`n_valid_y=2`（恰为该 circle 的两个坐标 token）、`valid_seq_len=18`；均匀 logits 下 `score_total` 与 `-20·ln267` 闭式一致。

回归：`pytest experiments/tests -q` → 251 passed；`test_cadmium_scorer.py::TestRealTokenizerCPU` 4 项失败与 `test_m0_manifest.py` 9 项 error 为**既有环境性失败**（本机缺 `transformers`、`pyarrow`，与本次新增文件无关；本次改动为纯新增两文件，未触碰任何既有文件）。

## 8. 假设（阶段 B 必须复核）

- A1：runner 前向须复现训练侧全部 mask/输入构造——`attn_mask` 直接用数据文件（含 START 列屏蔽）、输入三向量与 mask 同步 `[:, :-1]`、文本原串进 tokenizer、右 padding。任何「等价重写」（如自行重建 causal mask 而漏掉 `mask[1:,0]=-inf`）都会改变条件分布。
- A2：评分与 batch size 无关的前提是右 padding + 官方 mask 在真实前向下同样成立（合成层面已证，真实 logits 层面属 G1 的 batch=1 vs B 对照）。
- A3：`flag_vec/index_vec` 不参与分数本身，但参与模型输入；runner 须原样使用数据文件值（生成路径的 `create_*` 函数仅用于自回归，不用于 teacher forcing）。
- A4：checkpoint 键预期为 `adaptive_layer.* + cad_decoder.*`（可能带 `module.` 前缀），BERT 键全部缺失；阶段 B 零前向审计须逐键核对并记录统计，禁止 `strict=False` 放行。
- A5：`.cache/bert_large_uncased` 的 `model.safetensors` 源自 `BertForMaskedLM`，以 `BertModel.from_pretrained(..., local_files_only)` 加载时 `cls.*` 键将按未使用键报告，须记录而非静默。

## 9. 未解问题与后续阶段事项（均不阻塞本阶段结论）

1. 计划文件 BERT SHA256 少 1 位（§1），请 Codex 修订。
2. 阶段 B：checkpoint 零前向键/统计审计；`TextEmbedder` 显式 device 适配（L42 硬编码 `.to("cuda")`）；pinned tokenizer 的 512 token 截断审计（逐查询记录 token 数，超长拒绝而非截断）；`layers/functional.py` 注意力 mask 语义复核；全 8,046 候选逐样本结构核验（含无 padding 全长边界）。
3. 50/8,046 抽验不能替代全量 data_audit；未发现但理论上存在的异常（如非尾 padding）会在 runner 的逐样本 `validate_cad_vec` 下被拒绝并记账。
4. 官方 `environment.yml` 的 transformers 未钉版；服务器克隆环境为 transformers 4.53.1，tokenizer 行为差异属阶段 B 指纹核验项。

## 10. 停止条件自查（对照 `ACTIVE_TASK.md`）

- checkpoint 非 BERT 键无法严格对应 → 本阶段未加载权重，阶段 B 审计项 A4/A5 已预登记；未触发。
- 官方数据 CAD 向量/有效 mask 不自洽 → 50/50 抽验自洽 + 核心对任何不一致硬拒绝；未触发。
- 文本或 CAD 被静默截断 → CAD 侧：>272 不可能存在且核心拒绝；文本侧：已定位官方 512 截断点并列为 runner 前置审计；未触发。
- qrels 泄漏 → 核心接口无 qrels/UID/文件 I/O，测试锁定输出模式与字段隔离；未触发。
- 910 移植改变数学 → 本阶段仅 CPU 数学核心；device 适配点已定位（text_embed.py L42），数学定义与设备无关；未触发。
- 资产 SHA/许可不明 → §1 全部复核一致（BERT 计划值勘误已记录），许可 CC BY-NC-SA 4.0 已在改编文件头部声明；未触发。
- 远端 Git 不干净 → 本轮无 SSH；未触发。

官方评分定义无关键歧义（终止 token、双头 mask、shift 均有源码与数据双重证据），故**未创建 BLOCKERS.md**。

`review_required`
