# Idea 01 / P0：评分器实现与数据准入审计

任务 ID：`idea01_p0_scoring_audit_20260924`。本任务只在 `experiments/ACTIVE_TASK.md` 指向本文件且状态为 `implementation` 后生效。上位计划：[Idea 01 预备与 pilot](IDEA01_PREFLIGHT_AND_PILOT_2026-09-24.md)。执行者：本地 Claude Code / Kimi K3；代码与结果审查者：Codex。

## 目标与边界

交付一个可审核的 Qwen2.5-3B-SFT teacher-forced **条件程序评分器**，以及可重复的数据来源/泄漏/覆盖报告。允许在 `yunnan-share2` 单张 Ascend 910C 上做最多 **60 对**短样例评分的 P0 烟雾测试，总 NPU 占用上限 **1 卡小时**；不做 gallery 检索、R@K、P1 规模实验或 CadQuery 几何结论。P1/P2 必须等 Codex 审核代码及 P0 原始结果后另行更新 `ACTIVE_TASK.md`。

## 固定输入

- 仓库代码只能在 Windows 本地修改；远端 `/home/ma-user/work/CADRetrieval` 只通过 `git fetch` 与 `git checkout --detach <完整 SHA>` 运行。模型与数据源只读，取自 `experiments/assets/IDEA01_ASSET_STATUS.md` 所列 **910C OBS** 完整 revision 目录；本机运行副本已在 checkout 外的 `/home/ma-user/work/CADRetrieval_runtime/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/` 和 `/home/ma-user/work/CADRetrieval_runtime/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`，运行缓存和输出也放 `CADRetrieval_runtime/`。运行前 OBS 与本机副本分别 `sha256sum -c SHA256SUMS`；任何失败即停止。优先从本机副本加载模型并读取 CSV/ZIP：相同 BF16 455-token 前向的模型加载从 OBS 约 192 s，改为本机副本约 19 s。
- 模型 `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`，数据 `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`，环境 `/home/ma-user/work/envs/cadretrieval-idea01-torch271`。用单张 NPU，先记录空闲 HBM；**主精度 BF16**（checkpoint 配置和官方训练均为 BF16，910C 上 455-token 前向已通过），log-softmax 在 FP32 计算。不自动换模型、换 split 或下载别的权重。
- 官方训练/推理源码固定核验版 `Text-to-CadQuery/Text-to-CadQuery@4f7f50176f3c642f6d897c4a8670f7a83acb31bd`：Qwen 前缀为 `### Instruction:\n{input}\n\n### Response:\n`，输出原文后接 tokenizer EOS。训练脚本用因果 LM 对整串标签训练，完整 token 串先筛到 ≤1024；本项目条件分数只汇总 **response + EOS** 的 log-prob。不要用生成文本或训练 loss 代替此分数。

## 必须实现的核验

1. **来源映射**：解析 JSONL 与 CSV，按 `input == text2cad_v1.1.csv:intermediate` 恢复 `uid`；核对 `CadQuery.zip` 中 `CQ/{uid}.py` 与 `output`。逐 split 报告总数、映射成功/歧义/缺失、程序逐字一致/不一致、UID 交集和输入/程序的 exact duplicate。已有一次只读普查观察到 110,263 条文本均可唯一映射，val/test 程序全一致、train 有 2 条程序版本不同；请独立复算，不把这份初测当真值。UID 不等于设计家族，不能声称 family-disjoint。
2. **目标覆盖**：用 checkpoint 自带 tokenizer 的 `use_fast=False` 与官方前缀，测量每条 `prefix + output + EOS` token 长度、≤1024 覆盖率、prompt/response 长度、EOS 与边界可分性、导出样板/注释比例。不要默默截断或按评分值挑样本；长样本标记 `unsupported_full_sequence`。对 token 边界若独立编码的前缀不是完整串 token 的真前缀，记录并提出可复现方案，停在受影响样本。
3. **评分正确性**：对完整串前向，shift 后只 gather response token 的 `log_softmax(logits.float())`，含 EOS、排除 prompt/pad。保存每 token 的位置、目标 token ID、log-prob、sum、有效长度、mean 以及模型输入 token ID。用手构造 logits 样例检查 shift/mask/EOS；至少人工复核两个真实样例的 token 对齐。sum 是原始条件序列 log-prob，mean 只作长度诊断。不得剥离注释或 `cq.exporters` 后仍称“完整程序 NLL”。
4. **P0 选择与运行**：只在 validation 中选择最多 20 个完整串 ≤1024 的 UID；固定选择规则和种子，不依赖模型分数。每个候选最多算正确 query、另一个 UID 的错配 query、空 query 各一次，共 ≤60 对；长样本及失败样本全留计数。先对 1 对运行并自查有限值/显存/逐 token 表，再完成剩余 P0。输出条件边际分布，但不报告检索 R@K，也不把这 20 条的符号当研究结论。
5. **隔离与可复现**：评分器只能接收传入 query 文本与候选 `output`；UID/qrels/CSV 的其他 caption 只能由数据审计或评测层使用。P0 原始评分记录必须有 query UID、candidate UID、条件类型、完整路径 revision、Git SHA、环境 spec hash、token 长度、失败原因、设备/精度/耗时。缓存、日志、原始输出写 checkout 外；结束后远端 `git status --porcelain` 为空。

## 交接与停点

- 本地源码放 `experiments/src/`、脚本放 `experiments/scripts/`、配置放 `experiments/configs/`、必要的边界测试放 `experiments/tests/`；不要修改 `CLAUDE.md`、`ACTIVE_TASK.md` 或此计划。先在本地提交、推送，再远端检出同一个 SHA；不能把 `scp` 当代码同步。
- 本地 `experiments/submissions/idea01_p0_scoring_audit_20260924/IMPLEMENTATION.md` 写方法、代码 SHA、逐项测试证据、远端运行命令和与官方代码的对应；`RESULTS.md` 分开写事实、自己的分析、建议；`BLOCKERS.md` 记录不确定或失败。原始输出在 OBS 保留，轻量结果和清单回填本地。
- 达到 60 对或 1 卡小时、出现非有限值/边界歧义/模型哈希失败/超过 HBM 或远端源码树变脏时停止；不自行扩大预算。完成后把实施与结果交给 Codex 做**代码与原始分数复核**，等待下一阶段裁决。
