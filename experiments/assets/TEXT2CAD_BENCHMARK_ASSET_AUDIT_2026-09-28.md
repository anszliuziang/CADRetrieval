# Text2CAD v1.0 检索评测：资产与划分审计

本文件记录 Codex 于 2026-09-28 在本地作的只读盘点；Kimi/GLM 实现时须独立复算，不能把下列数字当作评分结果或自动准入。所有数据/权重本体在 Git 忽略的 `.cache/` 与共享 OBS，Git 只存元信息。Hugging Face 登录 token 和 SSH 私钥均不入仓库、不回填。

## 官方来源、版本和本地核验

| 来源/固定版本 | 文件 | 大小（字节） | 本地 SHA256 | 官方 LFS SHA256 |
|---|---|---:|---|---|
| `SadilKhan/Text2CAD@422d4c93198c3cd98d5c2092c0ab214b3a76a68d` | `text2cad_v1.0/train_test_val.json` | 2,992,377 | `4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b` | Git 小文件，无 LFS SHA |
| 同上 | `text2cad_v1.0/text2cad_v1.0.csv` | 1,078,124,103 | `32427ca7169f8b93411b64a025c387cdb0b55d9b9ce0e9cf9755ac8a3528f4a3` | 相同 |
| 同上 | `cad_seq.zip` | 606,626,050 | `9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa` | 相同 |
| 同上 | `text2cad_v1.0/misc/minimal_json/minimal_json_0000_0099.zip` | 114,762,278 | `7e9a29ead357fe284c29d1b1c7a411e1b8c681bdd48e2aba0542a376955fb879` | 相同 |
| 同上 | `text2cad_v1.0/Text2CAD_1.0.pth` | 91,601,866 | `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598` | 相同 |
| `chandar-lab/CADmium-ds@ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4` | `data/train-00000-of-00001.parquet` | 90,747,623 | `22429763b823fb67bdf240a12b2932664647b96099b35a641a6379c085fc4d30` | 相同 |
| 同上 | `data/validation-00000-of-00001.parquet` | 5,936,270 | `1608b54362402b48d8a4f0dbf687c42de2363510ec3efe3e59d7ad066b3fd110` | 相同 |
| 同上 | `data/test-00000-of-00001.parquet` | 5,420,575 | `c0f09e44d4d2540c1add36690a4c09c5929671299d0248aa5d3e1b822e4dddac` | 相同 |

模型待固定：`chandar-lab/CADmium-3B@6c6a79d160595207804171be56a037883b369abd` 的 adapter，base 为 `Qwen/Qwen2.5-Coder-3B-Instruct@488639f1ff808d1d3d0ba301aef8c11461451ec5`；原 Text2CAD 使用 `google-bert/bert-large-uncased@6da4b6a26a1877e173fca3225479512db81a5e5b`。CADmium adapter 文件 `adapter_model.safetensors` 479,002,040 字节，官方/本地/共享 OBS SHA256 均为 `531090f702d6b4b0680821b524c95f61cf5d208ee035b23caeabc2e8ec20634b`；其 `tokenizer.json` 官方/本地/共享 OBS SHA256 为 `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa`。BERT 的 `model.safetensors` 1,344,951,957 字节，官方/本地/共享 OBS SHA256 均为 `28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4`。Qwen base 两块权重仍在选择性下载，官方 LFS SHA256 依次为 `5b684409377e037e5ed75d680a070126e841dab17e8cb9a818868deadf01ab39` 与 `8bc5227aae6174eaf0b589fd4fbeb0de046a1fc951a67f2d2538acad811e1068`；未完成本地及远端校验。官方源码本地只读缓存：Text2CAD Git HEAD `669a81472ede2269837c22a4d00070eaac360c82`，CADmium Git HEAD `2b2a5e7346e17a239ffe6f4a932610bfdb9766c6`。

## 固定划分与协议风险

- 官方 v1.0 split：159,049 train / 8,925 validation / 8,046 test UID，三组交集为零。`cad_seq.zip` 有全部 8,046 test UID 的向量文件；v1.0 minimal JSON ZIP 亦有全部 8,046 test UID 的 JSON 文件。
- v1.0 CSV 共 174,000 行；test UID 仅命中 7,946 行/UID，缺 100 个 test UID，另有 12 个命中但 L0 `abstract` 为空。主协议暂为 8,046 CAD gallery + 7,934 条非空 L0 查询，配对 UID 为登记目标。
- v1.0 非空 L0 的 7,934 条查询在仅用 `str.strip()` 去掉首尾空白后，有 3,644 种逐字不同文本；639 个重复文本组覆盖 4,929 条 query，最大组 371。固定单正 UID 评测的理想 R@1/2/5/10/20 上限依次为 **45.93% / 53.98% / 64.31% / 72.13% / 80.40%**。若完全不去首尾空白，则有 4,010 种原始字符串、693 个重复组、最大组 320、R@1 上限 50.54%；这一口径把纯空白差异当作文本差异，不用于主协议。这仅是相同输入导致的数学上限，不是模型表现。
- 本地已在此前由 HF 官方 SHA 校验的 v1.1 CSV (`cded1b8048f432c192127a5d177da59dc926149ac02fb9ae31e1015ee6219136`) 覆盖 8,046/8,046 test UID，仅 1 条 L0 为空。[IMMA](https://arxiv.org/html/2605.05572)未明确说明使用 v1.0 还是 v1.1，不能从 8,023 个过滤后的 CAD 样本反推文本版本。
- CADmium-ds 的 118,299 train / 8,925 validation / 8,046 test UID，与官方 v1.0 对应 split 逐项无交叉，三份 parquet 自身也无交叉。CADmium 可进入 v1.0 test held-out 候选，但仍须核对模型训练脚本和权重版本。parquet 中 `uid,annotation,json_desc` 可直接给每个官方 test CAD 提供 CADmium 的**原生目标 JSON**；评测 query 使用 Text2CAD L0，不使用 CADmium 配套 annotation。
- 本地 P0 的 Text-to-CadQuery UID 清单与官方 v1.0 test 比对：其 train 99,236 UID 有 4,581 项交叉；val 5,513 与官方 test 交叉 259，test 5,514 与官方 test 交叉 247。故该 Qwen 权重不可在整个官方 test 上声称 held-out 检索。完整 CadQuery 代码覆盖和设计家族近重复仍待独立核验。

## 协议及许可原始出处

- [Text2CAD 官方数据卡](https://huggingface.co/datasets/SadilKhan/Text2CAD)明确 v1.0/v1.1 版本、文件及 **CC BY-NC-SA 4.0**，数据访问需用户接受条款；用户已在本机授权，2026-09-28 实测下载 v1.0 split 成功。该非商用/相同方式共享条件随派生数据使用，不把这些文件推入 Git。
- [Text2CAD 官方代码](https://github.com/SadilKhan/Text2CAD)的 `Cad_VLM/models/loss.py` 用 `label_smoothing=0.1` 和双坐标头的 masked CE；原训练 loss 不是原始命令的无平滑 NLL。其原始 logits 能否构造有意义的候选分数须在 M1 验证。
- [CADmium 官方论文附录 G](https://arxiv.org/html/2507.09792)声称新数据 MIT、微调模型 Apache-2.0；[官方仓库](https://github.com/chandar-lab/CADmium)列出数据与模型。下游使用 Text2CAD v1.0 的文本仍遵守原数据许可。CADmium `tokenize_dataset.py` 用 system/user/assistant chat 模板并将 assistant 前缀设为 `-100`；目标完整性、padding、停止 token 与实际权重训练配置须由 M1 核实。
- [IMMA 原文](https://arxiv.org/html/2605.05572)报告按点云可转换性过滤后的 8,023 test、只用 L0，R@1/5/10 = 9.71/25.54/36.05%。未取得作者 23 个过滤 UID 的最终清单、图库与文本版本，不能把我们的 v1.0 主协议成绩直接与其数字并榜。

## OBS 状态

共享 OBS 目录按 `datasets/Text2CAD-v1.0/<revision>/`、`datasets/CADmium-ds/<revision>/`、`models/Text2CAD-v1.0/<revision>/`、`models/CADmium-3B/<revision>/`、`models/Qwen2.5-Coder-3B-Instruct/<revision>/`、`models/bert-large-uncased/<revision>/` 分类。2026-09-28 `cvteam14:/data/obs_data_liuza/CADRetrieval/` 上，Text2CAD v1.0 的 split/CSV/cad_seq/minimal_json 四个文件、CADmium-ds 三份 parquet、原 Text2CAD 权重、CADmium adapter 及 BERT 权重均已逐文件远端 SHA256 校验通过。Qwen base 尚未就绪。云脑的 `/data/data_dev` 与共享 OBS 不是同一挂载，本次只读路径检查卡住；此处资产不计就绪，不能从共享 OBS 就绪推断云脑可读。**只有逐文件远端 SHA256 与上述官方值一致后才写“OBS 就绪”；目录存在或 scp 返回 0 不单独算核验。**
