# Text2CAD G4：完成后仅 CPU 收集与原始交付

2026-10-03。上位合同 `IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md` 的科学口径不变。本阶段由本地 Claude Code / 官方 Kimi K3 执行，Codex 独立审核。**零模型/BERT/NPU 前向，零新评分，不恢复、不重试，不计算检索指标。** 历史授权与原件保持原样。

## 已核事实与未通过的门

- run `idea01_text2cad_g4_full_20261002`；yunnan-share2 checkout 干净 detached `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`，不 fetch / detach / 修改源码或安装包。
- controller 3682036、scorer 3682124 / 3682194 已退出；实际 card0/1.exit_code.txt 均为0；controller_exit.status=ok、自有组empty、无后代/写入错误/截止违约，NPU两卡健康无进程。
- invocation `20261002T131158Z-c3812433` / `20261002T131159Z-6ce644de` 均 finalized、exit0，各14659812对/916466批/1822chunk。总29319624对，summary拒绝0，扣账119885.30卡秒（60002.09+59883.21）。
- controller共同起点epoch1790946711.70277，结束1791006731.1260767，墙钟60019.42330670357秒=16.67206202963988小时，低于共同七天上限。
- Codex已核3644个发布header的全范围/资产/分数绑定及17项完成元数据。**尚未逐行审核29319624条score，尚未核全量本地/OBS SHA，尚无R@K。** header/summary通过不是全矩阵数值通过。
- chunk原件共14858708934 bytes；只运输既有运行结果，不新增下载模型或数据。

## Kimi 固定任务

1. 阅读CLAUDE、ACTIVE、本合同、上位G4合同及 `IDEA01_M1C_G4_CODEX_COMPLETION_PREFLIGHT_2026-10-03.md`。只读核两summary/实际退出/controller_events/最终账本/NPU/Git；不能用当前零预算授权改写历史。full无G3 timing sidecar、cold_start/peak字段，不补造测量。
2. 从RT/budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json取稳定副本为本run的final_ledger.json。既存者核SHA相同；不同停止不覆盖。两个summary中真实即时peer-pending快照保持原样，不能用final ledger回写。
3. 收集恰3644完整chunk、两summary、final ledger、card0/1.exit_code.txt、controller_launch/events/exit、两scorer全stdout/stderr、PID/launch_time/实际argv、外层nohup log/PID。外层文件来自RT/runs同run前缀，作为原件副本加入run，记录来源/SHA。**禁止父目录递归归档混入CPU fixtures/其他run**；按实际清单，不猜 `.exit.json` 或 `summary.json` 文件名。
4. 补只读结束证据和固定输入身份：UTC、远端Git HEAD/status、NPU、磁盘、Python/库版本/实际平台、checkpoint/BERT五文件/cadzip/worker四文件实际SHA与既有固定清单比较；保存命令/真实退出。可以标准库metadata、sha256sum、git、npu-smi；不导入模型/torch_npu做新烟雾，不装包，不写服务器源码。既有环境spec10c18c17 warm reuse。
5. 生成LF SHA256SUMS，覆盖交付的所有原始/补充证据（清单不自引用），记录清单SHA/文件数/总bytes。云脑完整逐SHA真实exit0。取回本地 `experiments/runs/idea01_text2cad_g4_full_20261002/`，归档cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002/；双方完整逐SHA真实exit0。不能仅核tar/目录存在、截取成功行或用管道尾掩盖失败。未知旧产物/哈希不同保留报告，不删除覆盖或重跑评分。
6. 仅本地新写 `experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_G4_FULL_RESULTS_2026-10-03.md`，事实/解释/建议分开：完整运行SHA/授权ref/原件清单SHA、准确对数/批数/拒绝、实际wall/card时间、退出/组清理、资产与归档核验、限制/偏离。明确原始数值由Codex审核、当前无UID检索精度；双头评分不是纯MLE/几何NLL。
7. exact stage自己的新submission、commit完整SHA后回填报告SHA再提交；不push、不改CLAUDE/ACTIVE/plans/reviews/src/config/tests，不启动G5。提交后停止交接，不争用index。

## 运输和后续

- RT=/home/ma-user/work/CADRetrieval_runtime；运行环境PY=/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python。源码始终本地Git。复制原始数据/账本/日志不是源码同步；若需新执行脚本，先本地实现/提交审核，不现场热补。
- SSH多行stdin用tr去除CR后bash -s，保留transport/真实退出。长传输允许后台并读状态，不超过60秒阻塞等候、不重复启动。cvteam14仅OBS，避/data/data_dev。
- API/连接/数值/指纹/资源真实阻塞保存证据停下交流，不换API/模型、放宽门或重新评分。
- Codex独立全矩阵稠密索引、有限六分数、有效长度/算术、资产/预算/退出、本地OBS SHA审核后，另激活G5 CPU评价，恢复7934原查询UID指标、固定基线/重复上限/文本组CI。100-query或短名单不能替代全量。
