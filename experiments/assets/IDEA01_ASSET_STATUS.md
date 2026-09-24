# Idea 01 资产及环境准备记录

更新时间：2026-09-24。此文件是**逐项实测状态**，不是 Kimi 的实验开跑授权；状态变化时在本地更新并经 Git 同步。模型/数据本体不入 Git。

## 固定来源

| 资产 | 官方源 | 固定 revision | 许可元数据 | 本轮用途 |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B-SFT | `https://huggingface.co/ricemonster/qwen2.5-3B-SFT` | `aaa2cd16993bc67067b4c198a910d99cf2b7affe` | 模型 API `apache-2.0` | 冻结生成评分；只下载推理文件 |
| IEEE-T-ASE | `https://huggingface.co/ricemonster/IEEE-T-ASE`，仓库类型 **model** | `4091ce30ad85b774a9231484f9b40372dc0451d3` | 仓库 API license 字段为空；不可宣称数据 Apache | train/val/test 的泄漏核验和 query/program pair |

## 已核对的文件

以下 expected SHA256 来自官方 Hugging Face pinned resolve 的 `X-Linked-ETag`，本地下载后用 `Get-FileHash -Algorithm SHA256` 比对；两处 OBS 复制后再用 `sha256sum` 比对。文件名、大小与 hash 三者全同才记“就绪”。

| 文件 | 大小（字节） | 官方 SHA256 | 当前状态 |
| --- | ---: | --- | --- |
| `IEEE-T-ASE/data/data_train.jsonl` | 153864710 | `ece08d953d718fd238c67a374cbcf5e0494c5c3346dbd5f1238089e018b64b5f` | 本地、cvteam14 OBS、yunnan OBS 三处核对通过 |
| `IEEE-T-ASE/data/data_val.jsonl` | 8521454 | `272891bb3693c450b6f62daf8d52c16b8cffbb6248ea44735e61cc69f48b7d5b` | 三处核对通过 |
| `IEEE-T-ASE/data/data_test.jsonl` | 8527216 | `de31c643a6731271855be07473e22a8c7334b3cfa3621ce31a058adb6657590c` | 三处核对通过 |
| `IEEE-T-ASE/text2cad_v1.1.csv` | 1298988495 | `cded1b8048f432c192127a5d177da59dc926149ac02fb9ae31e1015ee6219136` | 本地、yunnan OBS、cvteam14 OBS 核对通过 |
| `IEEE-T-ASE/CadQuery.zip` | 102915929 | `1748082daef341f5f81d273cd4290e8e1c8d0d943928ddf22c6ca90e5a2e67c0` | 三处核对通过 |
| `qwen2.5-3B-SFT/model-00001-of-00002.safetensors` | 4957560304 | `377dad432a50cdb51ca3995d4f8237722d7d3e4e5b97540a8fd1f6acb4631997` | 本地下载完成、yunnan OBS、cvteam14 OBS 核对通过 |
| `qwen2.5-3B-SFT/model-00002-of-00002.safetensors` | 1214366696 | `3a4f707655fe8b114826cb55d8f9f31f2d74ccb30aea6ee4f4e90ed72879dfcb` | 三处核对通过 |

模型的 `config.json`、`model.safetensors.index.json`、`generation_config.json`、`added_tokens.json`、`special_tokens_map.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`、README 已从相同 pinned revision 取得，yunnan OBS 的完整 `SHA256SUMS` 校验通过，并完成本地目录加载。没有下载 `optimizer.pt`、scheduler、trainer state。数据仅有 `input`、`output` 两字段；逐行统计得到 train 99,236 / val 5,513 / test 5,514。源 CSV 可恢复 UID，但**没有家族字段**，不能宣称 family-disjoint 或多正例语义真值。

## 来源映射的只读初测

对 pinned CSV 的 176,017 行，用 `intermediate` 与三个 JSONL 的 `input` 做**逐字**连接：110,263/110,263 均唯一匹配；三个 split 的 UID 交集均为 0。再按 `CQ/{uid}.py` 检查 ZIP：validation 5,513/5,513、test 5,514/5,514 程序逐字相同；train 99,234/99,236 逐字相同。两条训练例外为 row 16122 / UID `0000/00000102` 和 row 28217 / UID `0000/00000144`，JSONL 版本比 ZIP 增加注释等内容。此初测是 Codex 的只读盘点，Kimi 须在正式准入报告中独立复算。UID 一致性解决样本溯源，不解决近重复设计、几何等价或语义多正例。

官方 Qwen 训练源码 `Text-to-CadQuery/Text-to-CadQuery@4f7f50176f3c642f6d897c4a8670f7a83acb31bd` 以 `### Instruction:\n{input}\n\n### Response:\n{output}{EOS}` 构造串，并只保留总 token 数 ≤1024 的样本；因果 LM collator 没有把 instruction 从训练标签中剔除。用该格式和 checkpoint tokenizer 对 validation **前 512 条**做不截断只读测量，485/512 条 ≤1024，最短 233、中位 414.5、最长 1987 token。此为可行性抽样，不是全 split 覆盖率；后续 Kimi 须完整统计。

## 存储与复制路径

- 本地临时源（Git 忽略）：`D:\projects\IdeaSpark\CADRetrievalIdea1\.cache\idea01_assets\`。
- 权威 OBS：`cvteam14:/data/obs_data_liuza/CADRetrieval/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`；模型在同一 `CADRetrieval/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`。
- 910C OBS 副本：`yunnan-share2:/data/data_dev/CADRetrieval/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`；模型在同一 `CADRetrieval/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`。
- 910C 本机运行副本：`/home/ma-user/work/CADRetrieval_runtime/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`，从已校验 OBS 复制，11 文件 `sha256sum -c` 再通过；仅作模型加载缓存，不作为权威资产，不在 Git checkout 内。
- 数据本机运行副本：`/home/ma-user/work/CADRetrieval_runtime/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`，同样从已校验 OBS 复制，6 文件 `sha256sum -c` 全部通过；供 CSV/ZIP 随机读取审计使用。
- 两个 OBS 挂载不在同一路径；当前核验使用本地 `scp` 到各位置、远端重算 SHA，不依据挂载名称推断共享。

## 环境与连接状态

- `ssh yunnan-share2` 免密通过。原 SSH config 别名拼写为 `yunnao-share2`，已新增 `yunnan-share2` 指向同一经核实端口；远端 `whoami=ma-user`、checkout `/home/ma-user/work/CADRetrieval`。
- 远端 `GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD` 无交互成功；本地曾 push 计划提交 `40fd5ee2f7862e4ea35e862b1970abd5e75d2ebe`，远端无交互 fetch 并 detach 到同 SHA，checkout 干净。后续文档更新需再次走同一流程。
- NPU 2 张，`torch.npu.is_available()=True`。从预装 PyTorch-2.7.1 环境克隆到 `/home/ma-user/work/envs/cadretrieval-idea01-torch271`，真实 NPU seeded matmul witness 为 `NPU_WITNESS (4, 4) 3920.0`。
- Qwen2.5-3B-SFT 完整权重在 yunnan OBS 经官方 SHA 校验，克隆环境从本地目录加载成功。11-token 短文本、FP16、`npu:0` 前向输出形状 `[1,11,151936]`、全部有限值；tokenizer 0.77 s、模型加载 25.497 s、前向 0.686 s、峰值分配 6,350,782,976 字节。这只证明模型可前向，不证明 CAD 条件评分正确或吞吐能扩到检索矩阵。
- 另以官方前缀和 validation row 6 的真实 `input + output + EOS` 共 455 token 做不计分前向：logits `[1,455,151936]`、全部有限值，FP16 `npu:0`，模型加载 74.014 s、前向 0.811 s、峰值分配 6,552,092,672 字节。两次加载耗时差异提示 OBS 冷/热读取不稳定；这不是评分吞吐或长期任务预算的测量。
- 同一 455-token 样例用 checkpoint 原生 **BF16** 从 OBS 前向成功：输出有限、加载 192.273 s、前向 0.684 s、峰值分配同为 6,552,092,672 字节。改从本机运行副本加载，BF16 权重加载 19.308 s、前向 0.743 s。P0 以 BF16 为主精度并优先用本机副本；模型加载差异不能当稳态评分吞吐。
- 环境 spec 的最新 canonical hash 前缀 `10c18c17`，独立 agent-follows-doc 在相同 prefix/环境变量下重跑 seed kernel（退出码 0）及模型目录 `sha256sum -c` 的 11 个文件（退出码 0）；实际 3B 前向的数值记录由 Codex 运行，未作为独立 agent 的原样重跑项。

## 未闭合项

1. 两处 OBS 的数据 6 文件均由各自 `sha256sum -c` 全部通过；模型 11 文件在 yunnan OBS 全部通过，在 cvteam14 OBS 第一片直接 `sha256sum` 与官方值一致、其余 10 文件由清单剩余部分 `sha256sum -c` 全部通过。两处资产准备闭合。
2. 已恢复 UID，但设计家族/近重复、数据许可链、CadQuery 程序可执行率和独立语义 qrels 尚未闭合；P0 可做仪器核验，不可宣称语义检索有效。
3. `experiments/ACTIVE_TASK.md` 已激活 `idea01_p0_scoring_audit_20260924` 的实现与限定烟雾测试；正式 gallery 检索和 P1/P2 仍需代码与结果审核后另行授权。
