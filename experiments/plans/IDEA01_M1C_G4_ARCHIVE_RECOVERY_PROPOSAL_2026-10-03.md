# G4 OBS 归档恢复方案（待用户决定）

2026-10-03。`status: pending_user_decision`。当前只允许只读诊断和记录，不授权启动运输、实现或评价。原 G4 评分、score 定义、图库、查询、历史 SHA 和预算完全沿用。

## 问题与推荐动作

Kimi 收集 CLI 和8路运输已中断，OBS 只有1129个 chunk 文件名，不能核验完整镜像；源端与本地的3666项原件、29319624条评分全行审核已通过。推荐**保留中断现场，在新 OBS 目录完成完整归档**，不处理已结束的评分作业。

## 固定源与目标

- 来源：本地 `D:/projects/IdeaSpark/CADRetrievalIdea1/experiments/runs/idea01_text2cad_g4_full_20261002/`，或云脑 `RT/runs/idea01_text2cad_g4_full_20261002/` 的同字节原件；不用父目录、CPU fixtures 或其他 run。
- 固定 payload：3666文件、14858752230 bytes，3644 chunk；LF SHA256SUMS SHA `97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733`。
- 保留的中断目标：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002/`。不删除、覆盖、清理或冒称完整。
- 推荐的新目标：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002_recovery_20261003/`。实施前确认不存在，若存在则只读报告，不能覆盖未知资产。
- 核验、失败、命令和运输退出日志放在独立的 `runtime/g4_archive_recovery_20261003/`，不加入原始评分 manifest；新归档成功后保留明确的 canonical archive 路径和旧目录的 incomplete 状态。

## 用户同意后的顺序

1. Codex 激活零NPU恢复合同，交给官方 Kimi k3[1m]；先核原 CLI/运输已结束，不重复启动健康作业。不变更 API、模型、环境或评分规则。
2. 仅从固定 manifest 运输评分原件，最多8路并发；用可持久保存的启动、PID、stdout/stderr 和实际退出记录，不将 Claude Code 的短期工具会话当作唯一监督与证据。若需要新增运输程序，先在本地 Git 实现，Codex审码并批准完整 SHA 后才能执行；服务器不写项目源码。
3. 每一路保留真实生产端与接收端退出，任何失败停止受影响步骤并保留现场，不把原中断补写为正常退出。运输总量约14.86GB既有资产，零新模型/数据下载、零模型/BERT/NPU前向，不重跑 G1–G4。
4. Kimi 在新目录逐项核3666 SHA；Codex再独立核相同 manifest、全部文件 SHA、覆盖、原始索引关联和运输退出。复用已经完成的两片全行数值审核及源端/本地审核，不重算117289444项数值检查。
5. Kimi 定稿原 `TEXT2CAD_G4_FULL_RESULTS_2026-10-03.md`，详细记录中断、旧部分目录、实际恢复方式、最终 canonical archive、各真实退出和新 SHA 证据；只提交自己的报告并交接。
6. Codex 最终审核完整矩阵后，另激活 G5 CPU 评价：恢复7934原查询UID指标、sum主表/mean诊断、固定tie、随机/长度基线、重复上限/分层和文本组CI。当前没有指标，也不允许看test调方法。

## 决策

推荐继续本方案。也可暂时停止研究推进，保留云脑、本地和部分 OBS 原件。两种选择均不需要再占用910C或重新评分。
