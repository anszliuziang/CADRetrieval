# Codex 接手 G4 归档：运行交接

2026-10-03，北京时间。**本报告为归档运行中交接，不是完整 OBS 或检索指标通过报告。** 用户已明确要求 Codex 接手继续；原 Kimi `TEXT2CAD_G4_FULL_RESULTS_2026-10-03.md` 未完成草稿保留、不覆写。执行者和审核责任更新于 CLAUDE/ACTIVE；不再依赖 Kimi CLI/API。

## 已完成

- 全量评分原件已成功：3644×8046=29319624对；历史评分SHA56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719，两scorer exit0/finalized、controller ok，16.672062小时墙钟/119885.30合计卡秒。
- Root全部29319624行/117289444数值检查、源端/本地3666 SHA、11054连接检查通过，已有稳定证据在runtime/g4_results_codex_20261003及OBS runs/m1c_g4_codex_rows_source_20261003。不因收集中断重跑评分或全行审核。
- 新归档脚本3ffd222独立只读审查；唯一Important（receiver启动失败遗失sender PID/argv）经真实反例修复f9c149fe1a8d616680d778b585f16ede2dc7a5d7，7项针对CPU全部通过；reviewer实际八路Windows tar toy全字节通过。Root另核gzip tar流的LF字节保存。
- 正式零NPU归档授权HEAD ade1ba0558c25b23de8c0ff00779ca4dbd316f98，已Git push；实际script SHA256 a608eb7acbfc1298751776f5fc94b6d36e537ec2293824e498ea15841fffa728，授权ACTIVE SHA25643178fed3925129df95b7035eb36b39521d0de672734f8568c0204a87cce9f35。
- G5规则已预注册a2ca7c7，尚未读qrels/计算精度：sum主表/mean诊断、固定tie、7934查询UID、长度/随机基线、重复上限/分层、5000文本组bootstrap。同图库口径和限制见G5计划。

## 正在运行

Windows `Start-Process -WindowStyle Hidden` 独立后台控制器于2026-10-03T08:29:46.5772665Z（16:29北京时间）启动一次，PID26168，已在启动工具会话结束后的独立检查中确认存活。launcher JSON/外层stdout/stderr保存于checkout外runtime。

运行目录：`D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_archive_recovery_20261003/`，state.json记录阶段，streamN_launch/exit及双侧stderr持久保留。最初 source_verify；观察记录g4_archive_recovery_observation_20261003.json确认265秒后进程健康、读取约11.72GB，说明原件校验持续前进，此I/O计数仅是进度诊断不是精确payload校验证书。

固定source `experiments/runs/idea01_text2cad_g4_full_20261002/`；3666payload/14858752230bytes/3644chunks，manifest97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733。目标 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002_recovery_20261003/`：独占mkdir，禁止复用/覆盖；原部分OBS现场保留。仅8路既有评分数据运输，零前向/模型下载/依赖安装/NPU。

## 尚未通过及下一步

1. 只读后台状态和真实两端退出，不重启健康作业。target存在不能重复启动helper。
2. 全部transport实际exit0和远端3666 sha256sum真实exit0后，Root再独立核OBS每字节/覆盖/unknown extras并合并原全行hash证据；尚未批准full_matrix。
3. 最终矩阵review通过后，Codex亲自实现G5 CPU评价；新代码独立审查、固定SHA后运行一次，Root从raw独立复算所有正例rank。完整报告及本地/OBS证据后结束监控。

实际连接/资源/SHA/退出/完整性失败保留现场并联系用户；不自动重试，不把低精度当工程故障。当前没有R@K，不把native双头含label smoothing训练称纯MLE/几何NLL。每30分钟监控已更新为Codex接手流程，正常无须用户行动时安静。
