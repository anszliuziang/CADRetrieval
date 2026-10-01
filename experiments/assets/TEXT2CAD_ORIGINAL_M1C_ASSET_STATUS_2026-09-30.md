# 原版 Text2CAD M1c 资产与零前向结构核验

2026-09-30；用户选择原版 Text2CAD 后，Codex 从本地官方 pinned 资产复制至 `yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/`，未修改服务器 Git checkout。共享 OBS 上的同源大文件此前已核验，见 `TEXT2CAD_BENCHMARK_ASSET_AUDIT_2026-09-28.md`；云脑 `/data/data_dev` 挂载未用。本地与云脑按整文件 SHA256 重算的七项一致，固定清单见 `experiments/configs/text2cad_original_asset_sha256_20260930.json`。

| 资产 | 大小（字节） | SHA256 | 云脑 checkout 外路径 |
| --- | ---: | --- | --- |
| `Text2CAD_1.0.pth` | 91,601,866 | `235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598` | `models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/` |
| `cad_seq.zip` | 606,626,050 | `9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa` | `datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/` |
| BERT `model.safetensors` | 1,344,951,957 | `28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4` | `models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b/` |
| BERT `config.json` | 571 | `f9a38c9ffa369f288c3206d0bf352dfed11d7bdee93aa5b2de6d7fdc7bff7871` | 同上 |
| BERT `tokenizer_config.json` | 48 | `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10` | 同上 |
| BERT `tokenizer.json` | 466,062 | `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` | 同上 |
| BERT `vocab.txt` | 231,508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` | 同上 |

在远端干净 detached Git checkout `b227ece55c6aca74d9908deb12ed4f08c0c20c00` 执行只读脚本 `experiments/scripts/inspect_text2cad_checkpoint.py`，`exit 0`；**未构造模型、未做前向、未占用 NPU**。原始 JSON/stderr 留在本地忽略目录 `experiments/runs/m1c_structure_20260930/` 与云脑 runtime 同名 run。`torch.load(weights_only=True)` 看到 `model_state_dict` 共 172 个张量、170 个非零；键带 `module.` 前缀（官方保存时的 DataParallel），无 BERT 权重，故模型加载时需从 pinned BERT 基座补齐并严格审核其它键。官方 test CAD `0000/00000093` 的 `vec.cad_vec` 是 `[272,2]`，`key_padding_mask` 也是 `[272,2]`；x/y 有效位置分别 140/74。当前这些是结构事实，不证明加载/评分正确，真实前向预算仍为 0。

Codex 后续在本地按同一 pinned checkpoint 列出了两个全零张量：`module.cad_decoder.cad_decoder_layers.0.norm_seq.norm_3.bias`、`.1.norm_seq.norm_3.bias`；它们位于无文本交叉注意力的前两层，需保留为预期例外，其余任何全零权重/偏置应单独审查。此观察不证明模型已经正确加载。

官方源码本地只读镜像 Git SHA `669a81472ede2269837c22a4d00070eaac360c82`；其 `LICENSE` 为 CC BY-NC-SA 4.0。若改编必要实现进入项目 Git，保留出处、许可与差异说明。下一道门是逐头 mask/shift、checkpoint 严格加载、910 适配的代码审查。

## 8,046 项图库零前向结构覆盖（Codex，2026-09-30）

Codex 在本地用 `D:\miniconda\python.exe experiments/scripts/audit_text2cad_gallery_vectors.py --split-json .cache/text2cad_official/text2cad_v1.0/train_test_val.json --cad-zip .cache/text2cad_official/cad_seq.zip` 扫描官方 test 的每个 UID；脚本只读 ZIP、以 `torch.load(weights_only=True)` 读取候选向量，核对固定 `[272,2]`、逐元素 `mask == (vec == 0)`，再调用阶段 A 的 `validate_cad_vec` 审首尾 token、尾部 padding、flag/index 和词表边界。退出码 0，`n_test=8046`、`n_valid=8046`、`reasons={}`。真实未填充序列长度分位：最短 19、P10 23、P25 24、中位 34、P75 56、P90 87、P95 106、P99 142、最长 212；x/y 有效项中位分别 33/8。此项只证明当前结构合同覆盖整库，不替代真实权重加载和前向验证。速度外推必须用这些长度分布，不能把统一填充长度 272 当作每项真实长度。

## pinned BERT 零前向加载与 query 截断检查（Codex，2026-09-30）

在云脑干净 detached checkout `47de292ec0a4d42172327fa533de6950f6d1cd18`、现有 `cadretrieval-idea01-torch271` 环境运行 `experiments/scripts/inspect_text2cad_bert_load.py`，原始 JSON/stderr 留在 checkout 外 `CADRetrieval_runtime/runs/m1c_bert_audit_20260930/`；未调用模型 forward，未占用 NPU，远端 Git 仍干净。传入的本地 M0 `query.jsonl` 与远端复制件 SHA256 均为 `a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd`。

结果：`BertModel.from_pretrained(local_files_only=True)` 加载基础 BERT **391/391 张量**，`missing_keys=[]`、`mismatched_keys=[]`、`error_msgs=[]`，词嵌入与 pinned safetensors 精确相等；仅 `cls.*` 的 7 个语言模型头张量属于预期未使用键。7,934 查询恰有 3,644 种逐字不同文本；在官方 `BertTokenizer` 下 token 数最大 98、P99 91、P90 66，`>512` 为 0。Stage B runner 仍须保留逐输入长度门，防止将来 query 更换后发生静默截断。这是基础 BERT 单独加载的事实，尚未证明 Text2CAD 完整 checkpoint 键匹配或真实前向。

## B1 审核提交上的完整模型零前向加载（Codex，2026-09-30）

在远端干净 detached checkout `76dd70aad5fe1531d3d39df0feb75afb669c659d`，Codex 用本地已提交的 `experiments/scripts/audit_text2cad_model_load.py`、固定 checkpoint/BERT 与 `experiments/configs/text2cad_checkpoint_expected_keys_20260930.json`，通过 `/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python` 做完整**零前向**加载审计，`exit 0`。原始文件在云脑 checkout 外的 `/home/ma-user/work/CADRetrieval_runtime/runs/m1c_model_load_20260930_b1/`：`stdout.json` SHA256 `c0afa6349f41315f55552c71ad332af795afdec4f841d434ed2973c1c52bee2a`，`stderr.log` SHA256 `1ee8090f4ed15e1399485ef9f16234342912cb3bdd13869c4f7039a2067a0f7c`。stderr 仅有 torch_npu 提示 Ascend toolkit 安装目录所有者与当前用户不同的警告，非加载错误。执行后 `git status --porcelain` 为空。

本次一体化检查结果：BERT 源与模型各 391 张量、`missing/mismatched/error=[]`、仅 7 个 `cls.*` 未使用、词嵌入精确相等；Text2CAD 非 BERT checkpoint 172 张量全匹配、逐张量数值相等，预期全零键仅两个无交叉注意力层的 `norm_3.bias`；模型为 `eval()`，状态为 `structure_only_no_forward`。**没有任何真实模型 forward、吞吐或检索精度结果；NPU 预算仍为 0。**

## M0 冻结 query/gallery 清单的云脑运行副本（Codex，2026-09-30）

将本地 `.cache/m0_real_run_v3/` 的 `query.jsonl`、`gallery.jsonl` 两个轻量数据清单复制到云脑 checkout 外 `/home/ma-user/work/CADRetrieval_runtime/pair_inputs/m0_v3/`，没有复制 qrels 到评分运行目录，也没有修改远端源码树。云脑重算 SHA256 分别为 `a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd`、`09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30`，与本地和固定指纹一致。复制后远端 Git 仍干净，detached HEAD 为 `76dd70aad5fe1531d3d39df0feb75afb669c659d`。这只准备输入资产，未执行模型前向。
