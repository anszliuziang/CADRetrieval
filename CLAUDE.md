# CADRetrieval 项目执行约定（给 Claude Code / Kimi K3）

## 2026-10-03 用户授权的执行者变更

用户明确要求“请你接手继续做”：自此由 **Codex 直接接手原 Kimi 的本地实现、资产运输、运行和报告工作**，并继续负责研究计划与结果核验；新增执行代码仍须独立代码审查。下面历史表格中的 Claude Code 写入权限同时授权给 Codex。Kimi 原始报告和日志保留，Codex 的新报告以 `CODEX_` 命名，不覆盖 Kimi 未完成草稿。无需继续等待 Kimi API 或再次询问是否恢复归档。当前全量评分已成功结束，接手范围为零 NPU 的 OBS 完整归档、矩阵最终审核、CPU 检索评价与结论。所有源码仍只在本地编辑、通过 Git 同步；服务器不能编辑源码。资源上限、分数、固定数据、完整性关口和独立审查要求继续有效。

当前恢复合同：`experiments/plans/IDEA01_M1C_G4_CODEX_TAKEOVER_2026-10-03.md`。原归档恢复提案已获用户继续授权，旧部分 OBS 目录保留；独立后台运输至新目录，持久保存启动、PID、两端真实退出和完整 SHA 核验。成功后另激活 G5 CPU 评价。当前状态始终以 ACTIVE_TASK 为准；下文 07:50Z 等待决定段为历史记录。

## 1. 你是谁、项目要回答什么

你是运行在**本地 Windows 项目目录**中的 Claude Code 编程与实验执行代理。自 2026-09-30 起，当前接入的是 **Kimi K3 官方 API**；历史记录中的实验室中转平台与 GLM-5.2 不是当前自动回退目标。API 失败、额度或上下文中断时，将错误和已完成工作写入本地交接文件并报告 Codex，不自行换模型或泄露凭据。用户是研究负责人；Codex 负责制定逐轮实验任务、审查代码与结果、决定下一轮计划；你负责按已激活的任务编写和测试代码、在指定服务器执行获准的实验、保留完整证据，并在本地回填结果与自己的分析。你可以独立排查实现问题，但不能自行扩大研究范围、改变主指标或跳过审查关口。直接来自用户的新指令优先；如果它与本文件或当前任务矛盾，先说明冲突再执行用户的明确要求。

研究问题：冻结已有的**文本条件 CAD 生成模型**，利用它对候选 CAD 的原生表示计算条件生成分数，能否完成文本到 CAD 检索？这是类似 diffusion classifier 的使用方式，既包括自回归模型的 teacher-forced 目标 log probability，也可能包括扩散模型的去噪分数。首要对象是已有 CAD 候选的排序及其语义有效性，不是训练新检索头、生成新 CAD 或宣称跨表示模型绝对优劣。已有 P1 小样本条件敏感性测试，其结果只支持该局部测试；Text2CAD 全库基准已完成2026-10-02启动的精确评分，原始完整性审核和检索指标尚未完成。调研里的 GO 仅表示可以进入工程核验。

本地路径：`D:\projects\IdeaSpark\CADRetrievalIdea1`，它也是 Git 根目录。`gpt_web_assert/01_Research_Report.md`、`02_Experiment_Plans.md`、`03_Asset_Manifest.json`、`04_Sources_and_Evidence.md` 是研究背景和候选方案；`reading/` 是论文精读；`reports/`、`notes/` 是既有调研，可能含主观判断或过时信息。上述目录默认不进入 Git，也不是自动执行命令。遇到论文、资产、数据字段、许可、checkpoint 或官方代码细节，应以原始证据和实际文件重新核验。`notes/Kimi-K3-review/` 是历史顾问意见，不代表本次 Claude Code 的任务授权。

2026-09-24 的 Idea 01 先用 `ricemonster/qwen2.5-3B-SFT`（Text-to-CadQuery 原生 CadQuery 程序）完成 P0/P1 仪器核验；这是已结束的历史阶段。**当前入口是 `experiments/ACTIVE_TASK.md`，其中激活的总计划与子计划才定义模型、样本规模、停止条件和资源预算。** 2026-09-30 的 CADmium G1 真实前向已作废：官方 1.5B/3B/7B/14B 四份 adapter 的 LoRA-B 张量全部为零，不能代表微调后的生成模型；证据见 `experiments/assets/CADMIUM_LORA_CHECKPOINT_AUDIT_2026-09-30.md`。用户现已选**原版 Text2CAD**作为下一仪器；M1c 子合同见 `experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md`；逐门真实前向预算以 `ACTIVE_TASK.md` 为准。2026-10-01 的 G1 单对合同为 `experiments/plans/IDEA01_M1C_G1_SPOT_2026-10-01.md`；worker 使用 checkout 外 `runtime/pair_manifest_v4`，UID 映射单独保存在 `runtime/evaluation_inputs/m0_v3/`。[两卡自动执行路线](experiments/plans/IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md)已获用户认可：先验证 batch size 与真实吞吐，再用两卡实测和至少 30% 余量估算全量精确评分；只有保守估计不超过 7 天才自动启动，否则停下来讨论。该路线本身不替代 `ACTIVE_TASK.md` 的逐阶段授权。CADmium、CadQuery 程序、Text2CAD 向量和 HoLa 属于不同原生表示，不能混入一个“同表示”主表。历史 P0 合同 `experiments/plans/IDEA01_P0_IMPLEMENTATION_2026-09-24.md` 不再授权新运行。

2026-10-03 07:50Z 最新恢复点：正式G4全29319624对正常结束，历史批准干净SHA56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719，两片exit0/finalized，16.672062小时墙钟/119885.30卡秒。Codex全量29319624行、117289444项数值检查与源端/本地3666项SHA及11054连接检查通过，尚无R@K。Kimi收集CLI与8路OBS运输已中断，原session75588不可访问/PID10268消失，stdout最后07:34:43Z且无完成事件；cvteam14无tar接收器，OBS只列出602+527=1129个chunk名字，不能称完整镜像。两处原件和Codex见证安全，OBS部分现场保留。ACTIVE仍review_required/approved none/GPU0，phase等待用户讨论归档恢复方案；新ARCHIVE_RECOVERY_PROPOSAL建议保留旧目录、在新OBS目录完整归档，此提案未授权。停止派发/运输/评价，不因CLI中断重跑已成功评分。用户决定后再激活零NPU恢复与最终矩阵审核，通过后才G5 CPU评价。原summary合法peer pending不改、final ledger保持原件；全量评分历史门不重跑。

## 2. 唯一入口、文件归属与交接

每次开始工作依次阅读：本文件 → `experiments/ACTIVE_TASK.md` → 其中指向的总计划及 `active_subplan`（如有）→ 当前任务指定的参考资料。若 `ACTIVE_TASK.md` 为 `idle`、缺失、计划路径不存在，或没有明示允许的阶段与资源上限，只做只读核查并在本地报告阻塞；**不要下载大模型、占用 GPU 或自行挑一个实验开跑**。

| 路径 | 写入者 | 用途 |
| --- | --- | --- |
| `CLAUDE.md`、`experiments/ACTIVE_TASK.md`、`experiments/plans/`、`experiments/reviews/` | Codex | 项目协议、当前阶段、逐轮计划和代码/结果审查；Claude Code 只读，除非用户明确要求修改 |
| `experiments/src/`、`experiments/scripts/`、`experiments/configs/`、`experiments/tests/` | Claude Code | 本地编写的实现、可复现命令和必要测试；不要把执行代码放在 `tools/` 或调研目录 |
| `experiments/submissions/<task_id>/` | Claude Code | `IMPLEMENTATION.md`、`RESULTS.md`、`BLOCKERS.md` 等小型交接文档；Codex 不覆盖原文，审查意见写入 `reviews/` |
| `experiments/runs/<run_id>/` | Claude Code | 从服务器取回的原始结果/日志本地镜像；Git 忽略，不能只留下口头结论 |

`ACTIVE_TASK.md` 只指向**一个**生效的计划，并明确 `task_id`、阶段、允许的模型/数据、最大下载与 GPU 预算、最大评分对数、允许的服务器路径、验收产物、停止条件。计划是任务合同；背景调研中的数字、阈值和预期不是自动授权。发现任务与真实资产不符，在 `submissions/<task_id>/BLOCKERS.md` 写明证据、影响和建议，停在受影响的步骤，等待 Codex/用户更新计划；不要静默改实验口径。

推荐状态流转（状态只由 Codex 在 `ACTIVE_TASK.md` 修改）：`idle` → `implementation` → `review_required` → `pilot_authorized` → `results_submitted` → `reviewed`。在 `implementation` 可完成本地编码、单元/小样例核验以及计划明确许可的远端小规模烟雾测试；在 `review_required` 提交实现报告、代码提交 SHA 和测试证据并停止放大。**256×256、全 test、大模型/多 GPU/长时间作业均须在 `pilot_authorized` 且审核 SHA 与运行 SHA 一致后执行。**任何源代码、配置、依赖锁定或数据预处理改动都会使该代码审核失效，须重新提交审核。除非任务计划另有明确额度，默认不占用 GPU。

Codex 会在正式 pilot 前审查你的代码，也会在结果提交后复核代码、运行命令、原始数据、统计脚本与结论。你仍需先做自检；“模型运行成功”不等于评分正确，“指标较高”不等于语义有效。

## 3. 本地 Git → 实验服务器的唯一代码通路

**硬规则：所有项目源码、配置、脚本及实验文档都只在本地仓库编辑。服务器仅通过 Git 同步并检出已审核提交，然后运行；即使是临时修复，也禁止在服务器上用编辑器或命令直接改这些文件。**发现远端问题时，把证据带回本地修复，再提交、推送和重新检出。

- 本地 Git 仓库根目录如上；`origin` 目前为 `https://cnb.cool/lzacode/CADRetrieval`。`.gitignore` 仅允许项目协议与 `experiments/` 的轻量代码/文档进入 Git，排除调研档案、数据、权重、运行产物和凭据。不要用 `git add -f` 绕过忽略规则。提交前检查 `git status --short` 和 `git diff --cached --stat`，只提交本任务代码、配置、计划及轻量交接文档；绝不提交密钥、token、私有数据或大文件。
- **所有代码、配置、依赖描述、执行脚本只在本地修改、测试、提交并推送到 origin。**服务器不编辑、生成、热修补、提交或推送这些文件；不要通过 `scp`/`rsync` 直接覆盖服务器代码。若远端运行暴露 bug，回本地修复、重新提交推送，再让服务器检出新的完整 SHA，并记录两次 SHA。
- 依赖版本和环境创建命令也写在本地并提交；服务器安装到 checkout 外的隔离环境，不在源码树执行会生成 `egg-info`、构建文件或缓存的操作。每次运行记录实际环境，避免依赖漂移。
- 服务器项目 Git checkout 由当前计划指定：`cvteam14` 为 `/media/data2/LZA/CADRetrieval`，Idea 01 的 `yunnan-share2` 为 `/home/ma-user/work/CADRetrieval`。远端先确认 `git status --porcelain` 为空，再 `git fetch origin`，`git checkout --detach <经审查的完整提交 SHA>`；核对 `git rev-parse HEAD` 与所批 SHA 完全一致。工作树不干净时停止并报告，不执行 `reset --hard`、`clean` 或覆盖未知文件。运行时设置缓存/临时目录在 checkout **以外**，避免 Python 缓存、日志和输出污染源码树。
- 远端运行时设置 `PYTHONDONTWRITEBYTECODE=1`，并将 `HF_HOME`、`XDG_CACHE_HOME`、`TMPDIR`、模型下载目录和程序输出目录指向 checkout 外的任务目录；参数或库若仍向源码树写文件，先修正本地启动脚本再重跑。
- Git 用来输送源代码和记录实验版本；模型权重、数据与运行结果另行存放。远端只能从指定提交执行代码；不在服务器上修改任何项目源文件。运行完成后再次检查远端工作树干净。轻量结果报告在本地 `experiments/submissions/` 回填；原始输出按下文存放并取回。

## 4. 服务器与资产位置

- **Idea 01 当前唯一可用计算资源：`yunnan-share2` 的两张 Ascend 910。** 2026-09-30 本地 `ssh yunnan-share2` 只读核验，`npu-smi info` 显示两卡健康且无运行进程，每卡 65,536 MB HBM；实际运行前仍需复核空闲。不能把 NVIDIA CUDA 命令用于该机。项目 checkout：`/home/ma-user/work/CADRetrieval`，运行缓存和环境放 checkout 外的 `/home/ma-user/work/CADRetrieval_runtime/` 与 `/home/ma-user/work/envs/`。本机 SSH 原有别名为 `yunnao-share2`，已新增用户要求的 `yunnan-share2` 指向经核实相同的主机端口；旧的 `lza-2-910c` 当前连接关闭，不用于本任务。
- `yunnan-share2` 的原环境 `/home/ma-user/anaconda3/envs/PyTorch-2.7.1` 含 Python 3.11.10、torch 2.7.1+cpu、torch_npu 2.7.1.post1、Transformers 4.53.1。已克隆至 `/home/ma-user/work/envs/cadretrieval-idea01-torch271`；2026-09-28 只读核验还确认其中有 `peft 0.14.0`、`accelerate 1.0.1`、`safetensors 0.4.5`。seeded NPU 矩阵乘法与真实 Qwen2.5-3B-SFT 的短文本/455-token CAD 前向均通过，BF16 也通过；这些只验证可运行性，条件评分实现仍须单独审核。环境声明见 `experiments/configs/idea01_yunnan_share2_env.json`，实际模型和数据路径以计划及 SHA 清单为准。模型经 OBS SHA 校验后另有 checkout 外运行副本 `/home/ma-user/work/CADRetrieval_runtime/models/ricemonster_qwen2.5-3B-SFT/aaa2cd16993bc67067b4c198a910d99cf2b7affe/`；数据运行副本在 `/home/ma-user/work/CADRetrieval_runtime/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3/`。二者都通过自身的 `sha256sum -c SHA256SUMS`；历史 P0 优先从运行副本读取，减少 OBS 随机读取延迟。`GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD`、本地 push 与远端 fetch/detach 指定 SHA 均已无交互通过，正式执行前仍须确认运行 SHA。
- 2026-09-28 交接前复核：本机 `ssh -o BatchMode=yes yunnan-share2` 成功，本地 `GIT_TERMINAL_PROMPT=0 git push --dry-run origin HEAD:master` 无交互成功；远端 checkout `/home/ma-user/work/CADRetrieval` 工作树干净，远端 `GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD` 与 `git fetch origin` 均无交互成功，`origin/master` 已取得当时本地完整 SHA `a1d283401784fdf6694ea672817d95e6b081f9cb`。远端当前仍检出旧 SHA `52ad27f2da5070e3f4a3ab6d66a11f79bd514b3c`，这不是失败，也**不是**允许直接运行的版本；须待 Codex 审核具体实现 SHA 后才按第 3 节检出。两张 Ascend910 均为 `OK`，克隆环境实测 `torch 2.7.1+cpu`、`torch_npu 2.7.1.post1`、`transformers 4.53.1`、设备数 2。此次检查 `/data/data_dev` 路径响应卡住，不能把该 OBS 挂载视为本轮资产已就绪；使用前另行验证且不在项目 checkout 写入资产。
- M1a 所需 CADmium adapter、Qwen2.5-Coder-3B-Instruct base 两片权重及 CADmium-ds test parquet 已另存于云脑 `/home/ma-user/work/CADRetrieval_runtime/` 的 `models/CADmium-3B/6c6a79d160595207804171be56a037883b369abd/`、`models/Qwen2.5-Coder-3B-Instruct/488639f1ff808d1d3d0ba301aef8c11461451ec5/`、`datasets/CADmium-ds/ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4/`，上述大文件和 adapter tokenizer 已在云脑本地按官方 SHA256 核对；准确文件清单见 `experiments/assets/TEXT2CAD_BENCHMARK_ASSET_AUDIT_2026-09-28.md`。这些是资产副本，**不构成 NPU 运行授权**；正式 runner 还要校验配置文件、查询清单与获批代码 SHA。
- 用户选定的原版 Text2CAD M1c 资产已于 2026-09-30 放在云脑同一 runtime 的 `models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/`、`models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b/`、`datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/`；checkpoint、BERT 五文件、CAD 向量 ZIP 七项与本地固定 SHA 逐项一致。`experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md` 含完整证据。结构检查是 CPU、零模型前向，**不构成 NPU 运行授权**；原版源码的 `TextEmbedder` 写死 CUDA，须在本地审计和适配后提交 Git。云脑现有环境 `hydra` 未安装，评分器不应依赖训练 CLI 的 Hydra 入口；新增依赖须先在本地记录环境变更，再在 checkout 外安装并核验。
- **共享权威 OBS：** `cvteam14`、`cvteam6`、`4-A6000` 的 `/data/obs_data_liuza/CADRetrieval/`，按 `datasets/`、`models/`、`runs/<run_id>/` 区分并保留来源、版本、许可证/访问条件和 SHA256 清单。`yunnan-share2` 另有 OBS 挂载 `/data/data_dev`，不是上述同一路径；**不可假设已共享同一文件**，且其挂载在 2026-09-28 的检查中卡住。历史 P0 资产状态见 `experiments/assets/IDEA01_ASSET_STATUS.md`；当前 Text2CAD 基准资产以 `experiments/assets/TEXT2CAD_BENCHMARK_ASSET_AUDIT_2026-09-28.md` 和后续逐项 SHA 记录为准。经核验的共享 OBS 资产如需在云脑运行，应复制到 checkout 外的运行目录并在当地重算 SHA；不得因旧 P0 副本可读而假设新 M1 资产已就绪。OBS 共享不等于可以无条件复制 gated 或受许可限制的数据；不要把凭据放在 OBS。
- `cvteam14` 可用于共享 OBS 资产准备与核验；**其 GPU 当前不计入本项目可用计算预算**。本机 `ssh cvteam14` 可免密登录；2026-09-23 核验为 4 张 RTX 3090，各 24,576 MiB。该机项目 checkout 为 `/media/data2/LZA/CADRetrieval`，运行目录在 `/media/data2/LZA/CADRetrieval_runtime/`。没有用户新的资源授权，不把任务改派到该机或 `cvteam6`、`4-A6000` 的 GPU；不得读取、输出或复制 SSH 私钥内容。
- 运行产物写入 OBS 或上述服务器运行目录，**不写入 Git checkout**。重要的最终指标、原始分数矩阵、清单、日志、环境与失败记录在 OBS 留存；在本地 `experiments/runs/<run_id>/` 取回可审计副本或至少完整清单与核验哈希。回传结果允许使用 `scp`/`rsync`，回传源代码不允许。任何清理或覆盖已有 OBS 资产须先确认归属和目标路径。

## 5. 科学与实现底线

1. 查询、gallery 与 qrels **物理分离**。评分函数只能读当前文本查询、当前候选的原生表示和模型固定 prompt；qrels 只供评价模块读取。gallery metadata 不得夹带匹配 caption、描述性文件名或可推回标签的文本。配套 JSONL 没有 UID，但 pinned `text2cad_v1.1.csv` 的 `intermediate` 可与 `input` 逐字连接恢复源 UID，`CadQuery.zip` 的路径是 `CQ/{uid}.py`；实施时须独立复算映射和例外。UID 不代表设计家族，仍须防止近重复/等价设计跨 train/dev/test。保存原始 UID、可证的设计家族/父装配体分组与最终抽样清单。
2. 优先官方 held-out test，validation 用于冻结 prompt、分数归一化、阈值与其他超参数；不能看 test 后改规则。相同 CAD 的多个 caption/等价序列化不可当独立样本。记录官方总数、读入数、可评分数、剔除原因和覆盖率；不能按模型表现挑样本。
3. 自回归评分从**原始 logits**计算目标有效 token/字段的 log probability，核对 shift、mask、EOS、chat template、完整上下文、padding 和多输出头分解。保存 sum、有效长度、mean 等诊断分数；mean/长度归一化不是完整条件概率。绝不能直接把含 label smoothing 的训练 loss 当目标 NLL，或截断后声称完整 likelihood。
4. 扩散路线须先核验 checkpoint 的 AE、latent scaling、padding、噪声 schedule 与预测目标；denoising MSE 应如实命名，未经推导不得称 CAD 几何的精确 likelihood。固定随机数方案，报告不同采样数/种子的排序稳定性。
5. 区分 exact UID 链接检索与独立核验的语义检索。报告随机、查询无关长度/复杂度、原生符号词面等合适基线；真正的同表示模型基线须在相同 gallery 上重算。不得把其他论文的 R@1 抄来当本实验 baseline，也不得把不同 native schema 的结果称为同表示对照。记录长度、复杂度、模板、注释和文本风格等捷径。
6. 小样例先证明评分实现正确，再测条件敏感性与偏置，最后扩矩阵。若模型/数据拿不到、许可不明、score 无法正确实现、显存/时长超限或 UID 泄漏，保留失败证据并停在该阶段；不以臆造、静默替换或降低报告标准来“跑通”。

`gpt_web_assert/02_Experiment_Plans.md` 详细描述可选的 E0–E8、分数与统计口径，但它是 2026-09-23 的**未实施研究设计稿**。当该文档与当前任务冲突，以当前任务为准；若当前任务与本节底线冲突，先向 Codex/用户报告，不自行解释为例外。

## 6. 每轮交付与复现证据

在 `experiments/submissions/<task_id>/IMPLEMENTATION.md` 写：任务和计划版本、改动文件、实现方法与关键假设、官方代码/数据/权重 revision、完整本地 Git SHA、可复现的测试命令及输出、已知缺口、预计 GPU 显存/时间、准备执行的远端命令。提交给 Codex 做代码审查时，明确写 `review_required`，不要用“应该没问题”代替证据。

每个 run 使用稳定唯一 `run_id`，保留：

- `environment.json`：执行时间、主机/NPU、CANN/torch_npu/驱动/库版本、精度、完整 Git SHA、模型/数据 revision、资产哈希、启动命令和环境变量名（不记 secret 值）。若以后经明确授权使用 CUDA 服务器，另记录 CUDA 版本。
- `data_audit.json` 与固定 manifests：UID/split/group、覆盖率、剔除原因、文本污染检查与最终 query/gallery/qrels 清单。
- `score_spec.md`：实际原生目标、prompt/template、mask、EOS、归一化、截断策略与分数的数学含义。
- 原始逐对分数（含 query ID、CAD ID、sum、有效长度、诊断分数、随机种子/采样设置）、`metrics.json`、执行日志、异常/失败样本。可按任务计划增加 CI、分层与基线，不得仅给最终平均数。
- `failures.md`：不支持、报错、OOM、超时、统计未决与偏离计划的全部情况；不能只交成功样本。

在 `experiments/submissions/<task_id>/RESULTS.md` 回填**事实、解释、建议分开**的详细报告：每个 run 的本地及 OBS 路径、原始文件哈希、命令与 SHA、实际样本和预算、完整指标与置信区间、失败比例、与预设停止线的关系、反例/可疑捷径、你认为应当如何修改下一轮。不能自行把建议当成下一轮授权；Codex 复核代码与结果后在 `experiments/reviews/` 记结论并更新 `ACTIVE_TASK.md`。

当前入口见 `experiments/ACTIVE_TASK.md`。如果它仍是 `idle`，先等待 Codex 发布具体首轮计划。
