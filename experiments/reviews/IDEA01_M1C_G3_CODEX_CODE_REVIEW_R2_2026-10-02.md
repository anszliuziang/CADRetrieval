# Text2CAD G3：R1修复后的第二次代码审查

2026-10-02。R1实现 `7f650d22a6259bb7b25bdc45abb8de989579ff79`；报告及回填 `da476b235c6310140fd2c9afa6622bd5be02f146`。Root独立核验及原只读reviewer按固定SHA复审，未做真实模型/BERT/NPU前向。

## 裁决

**仍不批准上卡；只补两个证据一致性检查。** 原四处修复已关闭，模型数学、全局索引和ETA公式未变。新合同 `IDEA01_M1C_G3_R2_BINDING_FIX_2026-10-02.md` 只处理授权卡时绑定与同卡串行窗口，不改变科学问题或要求新的用户决策。

## 通过的核验

- Root实际运行新增probe及受影响runner/pair tests：**142 passed、22 subtests，104.02秒**。未重跑无关全套、无skip替代。本地CPU假模型测试不构成检索实验结果。
- Root另复算23项：合法fresh证据、合法peer pending+最终账本通过；零planned/forwarded/charged、超卡时、外来invocation、错误epoch、缺最终账本、最终扣账为零、两卡一致的错误manifest/BERT/full272、真实reader空chunk及孤立timing均拒绝。七个评分/加载/计费核心AST与base一致，固定选择SHA仍为 `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`。
- 独立reviewer另运行22项针对测试全通过，9.678秒。源码核实生产CLI读取全部完整chunks/真实有效项数；producer测试调用真实scorer、atomic publish、sidecar、SharedBudgetLedger与summary函数；第一卡pending-peer来自真实账本。资产/pinned实际load门与fresh/resume双向一一对应已接入。

## 剩余Important

1. **卡时授权与账本cap可矛盾。** `text2cad_g3_probe.py:863/946/1082` 分别只检查各自上限，未要求相等。独立CPU见证将两卡authorization改为0.1小时（360卡秒）、snapshot cap仍43200且实际花费3000，ETA仍通过；final cap3001与snapshot43200也通过。应绑定 `authorization.card_hours*3600 == 两snapshot.max_card_seconds == final_ledger.max_card_seconds`。本轮实际批准固定12卡小时/43200卡秒，保持SharedBudgetLedger计费算法不改。
2. **同卡矛盾窗口仍可证明双卡重叠。** `text2cad_g3_probe.py:1303`只核每条窗口落在invocation内。将同卡50条query start全部改1000、保留各条E2E，仍通过。每query时长加权公式本身不变，但80%并发门可能接受不可能的单进程时间轴。须按生产query顺序检查下一条start不早于上一条end，拒绝逆序/重叠，不改变跨卡80%门或1.30系数。

Minor：Kimi R1报告写 `$RT/budget/`，实际 `PRODUCTION_BUDGET_ROOT` 是 `$RT/budgets/`。保留旧报告，在新的R2报告显式更正准确路径与命令。

## 证据

Root外置目录 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g3_probe_cpu_r1_20261002/` 包含独立脚本/23项见证/142项测试输出。reviewer外置目录 `g3_r1_readonly_review_7f650d22_etk6sa6a/` 包含新三组拒绝缺口见证及22项测试输出。

四份CPU原始证据已镜像到本地 `experiments/runs/m1c_g3_cpu_review_r2_20261002/` 与共享OBS `/data/obs_data_liuza/CADRetrieval/runs/m1c_g3_cpu_review_r2_20261002/`，OBS逐文件校验exit0；SHA256SUMS SHA `fd704056f2d57e9721dcae25ab1be7187d1286e6ba250e07645953c768bbdbbc`。初次manifest因Windows换行被Linux拒绝，改LF后重新核验成功，未掩盖退出码。

G2真实估计38.28394小时保留；G3没有真实运行。R2只需要新拒绝路径和合法时间轴的CPU验证，再复审这两处即可，不追加真实模型测试或新选择规则。
