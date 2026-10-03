# Codex：Text2CAD G4 完成状态预核

2026-10-03，Codex独立只读。结论：**全量评分进程正常完成，准入原始收集；全矩阵数值审核和检索评价尚未完成。**

## 证据与复算

- `D:\projects\IdeaSpark\CADRetrievalIdea1_runtime\g4_monitor_20261003T061241Z\`：06:12:43Z远端原始snapshot/transport，3644发布header的全范围/8046score/0rejected/资产与批准SHA一致；29319624实际扣账。健康两卡无进程，Git干净detached56ac0f8。运行中监控的PID缺失/chunk陈旧/controller日志非空本次由真实退出确认为正常完成，不重启。
- `D:\projects\IdeaSpark\CADRetrievalIdea1_runtime\g4_completion_preflight_20261003\metadata.json`：原始两summary、card0/1.exit_code.txt、controller_exit及其他小型元数据，SSH exit0。
- 同目录independent_check.json：17项完成绑定/真实退出/组清理/授权/范围/结算算术全部通过。

| 项目 | 独立核验值 |
| --- | --- |
| 完整运行SHA | 56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719 |
| 启动授权ref | 67e89a20eb6fc3b35d27c3ba2363da4311d9113c |
| 两片 | 各1822chunk/14659812对/916466批，summary status=ok/failure=null/拒绝0 |
| invocation | 20261002T131158Z-c3812433 /20261002T131159Z-6ce644de；两finalized/exit0 |
| 总扣账 | 29319624对，119885.30卡秒=33.3014722222合计卡小时 |
| 单卡时间 | 60002.09 /59883.21卡秒，与ended-started四舍五入一致 |
| controller | status=ok，两exit0/组empty/无后代/事件或记录错误/截止违约 |
| 共同墙钟 | 1791006731.1260767-1790946711.70277=60019.42330670357秒=16.67206202963988小时 |
| chunk bytes | 14858708934 |

card1完成时summary中card0仍pending，是真实时点快照，必须保留。最终审核用稳定final ledger，不要求两个summary快照均finalized，不改写历史。

## 未通过项与下一步

本次只检header和完成元数据，尚未逐行审核29319624条score，尚未独立核全量本地OBS哈希，不能声称检索精度或完整数值成功。撤销新NPU额度，激活零NPU原始收集合同 `IDEA01_M1C_G4_RESULTS_COLLECTION_2026-10-03.md`；原批准历史SHA/账本不变。完整原件及Codex全行核验后再激活独立CPU评价。
