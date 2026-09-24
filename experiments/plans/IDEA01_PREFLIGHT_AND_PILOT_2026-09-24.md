# Idea 01：冻结文生 CAD 模型做文本→CAD 检索

状态：**准备阶段；尚未授权 Kimi 写实验代码或运行正式检索。** 日期：2026-09-24。计划负责人：Codex；后续实现与运行：本地 Claude Code / Kimi K3，经 Codex 代码审核后执行。

## 0. 问题、边界和结论标准

问题：给定文本查询 `q` 和只含 CAD 原生程序的候选库 `G`，已有文生 CAD 生成器冻结后，以 teacher-forced 条件分数 `S(q,x)` 排序，是否得到**独立语义真值认可**的检索结果？本轮限 U1，不研究 U2/U3；不增训对齐头。当前首选仪器为 `ricemonster/qwen2.5-3B-SFT` 的 CadQuery 输出。跨 CADmium/HoLa 的数值不能当同表示或架构因果比较。

主张只设两条：

| 编号 | 待证主张 | 最低证据 | 主要反解释 |
| --- | --- | --- | --- |
| C1 | 原生条件分数含有可用于文本→CAD 检索的语义信息 | 在独立正负例和固定 gallery 上，条件分数的成对排序超过查询无关长度、模板、词面基线；换成错误/空 query 时优势减弱；报告不确定区间和失败覆盖 | 只记住训练模板、程序长度或 UID 链接 |
| C2 | 已有候选先验/长度校正是否足以解释或改善 C1 | 同分数矩阵比较原始 sum、mean、`b̂`、BLiM CPN `α` 网格，全部在 validation 固定后评 test；报告收益和额外计算成本 | `S_corr` 等新校正只是已发表 CPN 的昂贵重述 |

本项目不声称 Bayes 分解、CPN、均匀类别先验假设、teacher forcing、无训练使用或 O(N) 打分本身有新颖性。扩散模型、跨 schema 表和大型等价表示审计不在首轮实现范围。02 是必要对照，03 是测量准入检查，04 暂缓。依据：[Codex 独立审查](../../ideas/novelty_check_2026-09-24/aris_output/CODEX_REVIEW_2026-09-24.md)（本地研究档案，不进入 Git）；实施时以本计划的数值口径为准。

## 1. 数据、资产和准入 E0

固定来源：模型 `ricemonster/qwen2.5-3B-SFT@aaa2cd16993bc67067b4c198a910d99cf2b7affe`；配套仓库 `ricemonster/IEEE-T-ASE@4091ce30ad85b774a9231484f9b40372dc0451d3`。需下载模型推理文件（两片 safetensors、index、config、tokenizer 和 README）以及 `data_train.jsonl`、`data_val.jsonl`、`data_test.jsonl` 与 README；不下载 optimizer/scheduler、1.3 GB CSV 或 103 MB zip，除非后续审查证实必要。仓库为 Hugging Face **model** 类型，数据仓库 `license` 元数据为空；内部研究可做资产核验，公开再分发与许可链另行核查。权重页面标 `apache-2.0`，不能据此推断数据也同许可。

OBS 权威副本计划放在 `cvteam14:/data/obs_data_liuza/CADRetrieval/{models,datasets}/...`，版本路径含完整 commit SHA，保存 `SHA256SUMS` 与 `PROVENANCE.md`。910C 机器的 OBS 挂载是 `/data/data_dev`，已观察到的根目录与前者不同；不得假设它们共享内容。若不共通，将核验过的相同资产复制到 `yunnan-share2:/data/data_dev/CADRetrieval/...`，再次逐文件比对 SHA256。模型缓存、临时文件、conda 环境和运行输出放在 `/home/ma-user/work/CADRetrieval_runtime/` 或 checkout 外的其他确定目录。任何仓库源码和计划只在 Windows 本地改，Git push 后远端检出指定 SHA；资产不进 Git。

E0 检查顺序：

1. **资产**：公开仓库 revision/文件哈希/许可字段与已下载大小一致；每个文件非空。检查 tokenizers、safetensors 索引和模型 config 能组成完整加载目录。下载异常保留错误，不把空文件算成功。
2. **环境**：`ssh yunnan-share2`、远端 `git ls-remote origin HEAD` 无交互成功；远端 checkout 干净。克隆已有 PyTorch-2.7.1 环境到 checkout 外，逐项记录 Python/torch/torch_npu/Transformers/CANN 版本；至少通过 seeded NPU matmul，继而通过 Qwen 小文本前向和数值非空检查。不以 `import` 成功宣布环境可用。
3. **数据**：从原始 JSONL 记录每个 split 的行数、输入/输出字段、来源 UID 或可追溯映射、父设计家族、精确/近似重复与 train/val/test 交叉。若无 UID，就明确只能做按样本/模板隔离，不能宣称 family-disjoint。统计 1024-token 截断比例、可执行 CadQuery 覆盖、`cq.exporters` 等样板覆盖。固定保留/剔除清单，不能因模型分数筛样本。
4. **qrels**：官方 input/output 对可用于**配对 UID 检索 sanity**，不能直接声称自然语义多正例金标。主张 C1 至少还需从独立几何或可执行谓词建立一批正负例；若覆盖不足，首轮只报 sanity，不报“理解 CAD”的结论。
5. **打分定义**：只对输出 token teacher force，明确 chat/instruction 模板、shift、EOS、pad、长度、截断。原始全程序 `Σ log pθ(t_i|q,t_<i)` 作为序列概率主臂；设计 token subset、均值和去导出样板均明确标为诊断分数，不能称完整 NLL。保存逐 token 对齐的人工复核样例。

**E0 阻断条件**：资产/许可来源不清、模型无法在 NPU 正常前向、分数掩码有误、query 与 gallery 文本泄漏、有效 qrels/程序覆盖太低，均不扩到正式检索。可以回本地更正计划；不得静默改规则或更换模型。

## 2. 实验块与运行顺序

| 阶段 | 样本和分数上限 | 比较与产物 | 进入下一阶段的判据 |
| --- | --- | --- | --- |
| P0 单样例核验 | ≤20 条单候选，仅已下载资产；不算正式检索 | 固定提示模板、target token 掩码、EOS/截断、逐 token log-prob、NPU/CPU 或手算局部一致性 | 评分实现被 Codex 审核；输出有限且可复算 |
| P1 条件敏感性 | 32 个 query × 8–16 个候选，最多 512 对；优先同家族难负例 | 正确/错配/空文本、原始 sum/mean、长度/模板/词面基线；程序执行和逐对证据 | 真实 query 对比错配 query 有稳定正向成对边际；结果不只由长度/模板解释；否则止损或修正测量 |
| P2 小库检索与校正 | 最多 256×256，需新的 `pilot_authorized` 和已审核代码 SHA | U1 R@1/5/10、MRR、每 query 多正例口径与 UID 口径分开；`b̂`、CPN `α∈{0,.25,.5,.75,.8,.9,1}`、`S_corr` 仅在确有收益假设且成本可控时加 | 独立语义 qrels 有足够覆盖，至少一个条件分数臂超过最强无条件/词面基线且区间支持有用效应 |
| P3 主矩阵 | gallery 1,003 → 全 test，另行预算和审核 | 相同 gallery 上的强判别式或 caption-then-embed 对照；按家族聚类 CI、长度/复杂度/模板分层、wall-clock 与 NPU 时长 | 只在 P2 明确通过后启动；失败也原样报告 |

P0/P1 的**授权对象是后续 Kimi 任务**，此文当前只用于 Codex 资产/环境准备；`ACTIVE_TASK.md` 保持 `idle` 直至 Codex 完成准备并发布实施任务。P2/P3 不因本计划存在而自动授权。任何模型、数据、依赖或代码版本变化均重新核验打分正确性与 SHA。

主表最低三类基线：查询无关（随机、长度、模板/复杂度）、同分数家族（sum/mean/`b̂`/CPN）、同 gallery 的现成检索器或 caption-then-embed；后一类即使需要生成 gallery caption，也明确计入推理成本与输入条件。正式指标以 query 为重抽样单位，并按 CAD 家族聚类；调参只看 validation，不按 test 结果重新挑 `α`/模板/清洗规则。零效应须报告置信区间，不能把“无显著性”写成“等效”。

## 3. 计算、停止与交付合同

- P0/P1 单卡优先，910C 上的吞吐、显存、编译缓存和算子兼容性尚未验证，GPU 小时不预填虚假数字。先记录 10 对的真实 tokens/s、峰值 HBM，再以该数字估 P1/后续成本，设运行时间上限。P2/P3 的预算另行批准。
- 对每次运行记录模型/数据完整 revision、SHA256、代码 Git SHA、环境 spec hash、NPU ID、精度、提示模板、score mask、样本清单和所有失败原因；保存逐对原始分数与执行日志，不只存平均 R@K。
- Kimi 的代码只能写 `experiments/src/`、`scripts/`、`configs/`、`tests/`，本地修改并经 Git 同步；实现报告写 `experiments/submissions/<task_id>/IMPLEMENTATION.md`。Codex 在扩大前审查代码、输入隔离与 P0 输出；Kimi 的 P1 结果与分析回填 `RESULTS.md`，Codex 再审查原始分数和复算指标。
- 止损优先级：技术有效性 E0 → query 条件敏感性 P1 → 强基线与独立语义真值 P2 → 大库可行性 P3。任何阶段失败不得把下一阶段的“无数据”描述成支持论点。

## 4. 当前准备记录（2026-09-24，待持续补全）

| 检查 | 当前证据 | 状态 |
| --- | --- | --- |
| `ssh yunnan-share2` | 本地原只有 `yunnao-share2`；已添加同端口别名，登录到 `ma-user` 的两 NPU 机器 | 通过 |
| Git 免交互 | 远端 checkout `/home/ma-user/work/CADRetrieval` 干净；`GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD` 返回 `dc220c5...`，与当前远端 HEAD 一致 | 通过（只读） |
| 910C 基础环境 | `/home/ma-user/anaconda3/envs/PyTorch-2.7.1`，Python 3.11.10，torch 2.7.1+cpu，torch_npu 2.7.1.post1；`torch.npu.is_available()` 真、2 设备 | 仅导入通过，内核/模型未测 |
| OBS | 老机器 `/data/obs_data_liuza`；新机 `/data/data_dev`，不能假定同桶 | 配套 JSONL 已分别复制，两端 SHA 与官方一致；模型仍待复制 |
| 数据 | 官方 API revision 已复核；train/val/test 分别 99,236/5,513/5,514 行，仅有 `input`、`output` 字段 | 三份 JSONL 已就绪；UID/家族映射未闭合 |
| 模型 | 小型推理元数据已从 pinned revision 获取；大权重分片仍在传输 | 不就绪；Qwen NPU 前向未测 |

## 5. 准备完成后的启动条件

只有模型/数据完整落盘、SHA 清单核对、环境 kernel + Qwen 前向通过、数据 qrels 可用性至少获得初步证据，并且 Codex 把本计划落实为一个**限定 P0/P1 的** `ACTIVE_TASK.md` 后，才向 Kimi 下发实现工作。下发时注明每阶段资源上限、允许的远端路径、代码审核点、结果回填格式；Kimi 不得把本准备计划当 P2/P3 的开跑许可。
