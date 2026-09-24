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
| `qwen2.5-3B-SFT/model-00001-of-00002.safetensors` | 4957560304 | `377dad432a50cdb51ca3995d4f8237722d7d3e4e5b97540a8fd1f6acb4631997` | 下载中，**未就绪** |
| `qwen2.5-3B-SFT/model-00002-of-00002.safetensors` | 1214366696 | `3a4f707655fe8b114826cb55d8f9f31f2d74ccb30aea6ee4f4e90ed72879dfcb` | 下载中，**未就绪** |

模型的 `config.json`、`model.safetensors.index.json`、`generation_config.json`、`added_tokens.json`、`special_tokens_map.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`、README 已从相同 pinned revision 在本地取得；仍须与完整权重一起上传，验证 Transformers 能从**仅本地目录**加载。没有下载 `optimizer.pt`、scheduler、trainer state。数据仅有 `input`、`output` 两字段；逐行统计得到 train 99,236 / val 5,513 / test 5,514。**无显式 UID/家族字段**，不能未经额外映射就声称 family-disjoint 或多正例语义真值。

## 存储与复制路径

- 本地临时源（Git 忽略）：`D:\projects\IdeaSpark\CADRetrievalIdea1\.cache\idea01_assets\`。
- 权威 OBS：`cvteam14:/data/obs_data_liuza/CADRetrieval/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`；模型待放 `.../models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`。
- 910C OBS 副本：`yunnan-share2:/data/data_dev/CADRetrieval/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`；模型待放 `.../models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`。
- 两个 OBS 挂载不在同一路径；当前核验使用本地 `scp` 到各位置、远端重算 SHA，不依据挂载名称推断共享。

## 环境与连接状态

- `ssh yunnan-share2` 免密通过。原 SSH config 别名拼写为 `yunnao-share2`，已新增 `yunnan-share2` 指向同一经核实端口；远端 `whoami=ma-user`、checkout `/home/ma-user/work/CADRetrieval`。
- 远端 `GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD` 无交互成功且 checkout 干净；尚须在本地文档 commit 后实际 push、远端 fetch 指定 SHA。
- NPU 2 张，`torch.npu.is_available()=True`。从预装 PyTorch-2.7.1 环境克隆到 `/home/ma-user/work/envs/cadretrieval-idea01-torch271`，真实 NPU seeded matmul witness 为 `NPU_WITNESS (4, 4) 3920.0`。
- **Qwen 权重尚未完整下载，因此模型前向未测试；不得说“NPU 可跑 Qwen”。**

## 未闭合项

1. 两个模型分片完整下载、官方 SHA256、两处 OBS 副本核验。
2. 从本地模型目录加载 tokenizer/模型并在克隆环境中完成一条有条件 Qwen NPU 前向；记录峰值 HBM、tokens/s 和失败栈。
3. 数据源 UID/家族与许可链；CadQuery 程序真实可执行率；独立语义 qrels 的构造与覆盖。
4. Kimi 实现任务尚未发布；`experiments/ACTIVE_TASK.md` 仍为 `idle`。
