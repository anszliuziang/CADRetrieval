# Text2CAD G4 R1复审：仍需最小R2修复

2026-10-02。R1代码226c4108f57e332ebfe967df24980ccc35c0595b；报告6fd8407e1db1170eaab4a094588bcad7b9a908cc，回填HEAD 0f66086b1a5ad17b9bfded85e17fcff471fc5e19。GPU/模型/BERT前向0，远端仍G3干净487d02b9，没有正式G4产物。

## 已关闭及审核范围

原post-spawn pid/time/launch记录异常已改为记错并继续监管，活孩子无假NONE或重复exit；event/末次Git/退出记录故障的基本收尾亦有实际CPU孩子证据。Root独立针对18项测试exit0、16.49秒；唯一复用的fresh reviewer针对23项exit0、13.28秒。旧七评分核心AST、full固定argv、npu-smi混合marker/busy解析通过。没有重跑G1/G2/G3或无关全套。

Windows actual只能证明本地收尾及降级；POSIX组信号仍是mock，未在Linux实测。当前**不批准正式G4**：以下两个Important已独立复现；不得把R1报告的“硬界不变”当证据。

## Important 1：真实后代终止可越过七天截止

text2cad_g4_launch.py约1165行把deadline+kill_wait_s作为_drain_card_group真实settle/TERM/KILL的hard_end。原来的额外reap记账等待不授权后代在七天后继续运行。父在deadline附近自然退出后，新的10秒settle可把真实组TERM推到deadline之后。

Root见证CODEX_G4_R1_REMAINING_GATES_20261002.json：明确mock direct drain，原deadline1600，父1599已退出，调用方hard_end1630，真实函数到1609才请求TERM（晚9秒）。这是逻辑反例，不是NPU/真实Linux实测。Fresh reviewer读源代码和反例确认。修复应将后代执行和组TERM/KILL纳入原共同deadline及其预算内宽限；记录/reap延时不能变成继续运行授权。需要接近deadline的组残留、单片已退出/peer仍运行等边界证据，不扩墙钟/卡时。

## Important 2：最后事件写失败仍报ok且持久证据隐去错误

同文件约1251行先计算ok并写controller_exit.json，再events(controller_exit)。若只有最后事件append失败，event_errors才被追加；返回status仍ok，持久JSON的event_errors仍空。

Root实际两个CPU短孩子exit0见证：注入仅controller_exit事件OSError，返回ok/error非空，持久JSON ok/error空。需在完成状态持久化前确定所有必需事件I/O的结果，或者明确改为失败并保留一致证据；保持fail-if-exists/真实退出，不伪称成功。不要为追求完美日志进行无限retry或改旧评分器。

## 见证有效性及Minor

- ProcessGroupTest两真实树测试先在finally用测试自身杀后代，再断言组/marker；这会污染“controller没有留下活后代”的证据。实际组/descendant PID存活及controller退出时状态必须在测试清理之前采集、断言；finally只清理测试自己创建的进程，不能代替被测controller。
- group_alive把任何OSError当empty；非ESRCH错误不能声称已空。Fresh将其列Minor鲁棒性，可同局部修复如实记录probe错误，不引入通用资源平台或扩大研究门。
- R1报告“组内不可能碰其他作业”过强：它自己的pgid复用限制已承认；正式报告保留该限制，不扩大成普遍保证。

## 原始证据

Root4文件镜像本地experiments/runs/m1c_g4_cpu_review_r2_20261002/与cvteam14 OBS同名run；LF manifest SHA d0ab8b8d1c09d3f85446ac0ea51bfa8ed9b58dffc3fcae52dfe7f89d27fdc78b，本地/OBS4项逐SHA及OBS exit0通过。Kimi5文件原R1 witness本地/OBS run m1c_g4_r1_cpu_witness_20261002，manifest c88d83fa9b8986c0ae656845b339b4890399d6668ebd02c193a177ab55fc9fd5，双方5项逐SHA/OBS exit0通过。全是零NPU CPU证据，存储归档不构成远端源码部署。

## 下一步

激活IDEA01_M1C_G4_R2_COMPLETION_FIX_2026-10-02.md，限于新增controller/test的上述边界。修复审码后，另授权Git部署的零NPU Linux必要CPU组见证（先观察后清理）；不运行全部62套、不安装包。通过真实POSIX收尾后才发布新完整SHA和正式七天预算授权。评分资产/FP32/full272/b16/无trim-cache、29319624对与38.28394小时估计均不变。
