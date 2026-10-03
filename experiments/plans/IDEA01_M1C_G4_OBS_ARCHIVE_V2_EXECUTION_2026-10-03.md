# G4 完整 OBS v2 归档执行

状态：用户工程持续授权，真实双条件诊断及独立代码审核通过，零NPU。

- 执行代码：8bc4eeabea57f416982a286add9b65ba35f57ed1；本地run_g4_archive_recovery.py --mode full。
- ACTIVE必须pilot_authorized、完整approved_sha、archive_mode=full、gpu_budget=0/max_scored_pairs=0；代码字节等Git blob。必须读取同批准代码的prepared真实诊断通过状态。
- source：本地experiments/runs/idea01_text2cad_g4_full_20261002；清单SHA97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733，3666payload/14858752230bytes/3644chunks。完整评分已成功及全行数值审过，不重复NPU/模型/BERT/旧全行审核。
- target：cvteam6:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002_recovery_v2_20261003。旧两个部分目标保留；新独占mkdir，不复用未知目录。cvteam6只共享OBS存储，不用GPU；yunnan-share2不fetch/detach/编辑/安装。
- control：checkout外CADRetrievalIdea1_runtime/g4_archive_recovery_v2_20261003。新Windows后台controller仅一次，Start-Process Hidden，实际PID/argv/退出持久化；健康任务仅读进度。
- helper完整源SHA后预建清单全部父目录并检查存在，最多8 tar流，双方实际exit与stderr/PID/命令保留；全部成功才远端全SHA，非零不能视为成功。
- 运输后Root用既有Py3.8只读witness SHA43db1b90be46f346cdf6a9bddf83717a7b5d777814631e9d70172f1b654a1323独立完整3666SHA/bytes/3644densechunk覆盖，不混入父目录CPU fixtures。原source/local审核和两片完整行hash绑定不再无故重复。
- 完整matrix最终review通过后才激活G5既有固定CPU指标。此次diag/root/code/运输证据均需OBS归档，不篡改原manifest或伪造缺失计时；所有代码文档本地Git。

工程故障保留真实证据，依据根因修复、审查后继续，不重复许可。协议/数据/数值/预算变动仍讨论；禁止删除覆盖未知资产或无依据重复全量复制。
