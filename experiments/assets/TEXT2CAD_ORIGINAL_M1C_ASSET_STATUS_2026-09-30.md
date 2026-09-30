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

官方源码本地只读镜像 Git SHA `669a81472ede2269837c22a4d00070eaac360c82`；其 `LICENSE` 为 CC BY-NC-SA 4.0。若改编必要实现进入项目 Git，保留出处、许可与差异说明。下一道门是逐头 mask/shift、checkpoint 严格加载、910 适配的代码审查。
