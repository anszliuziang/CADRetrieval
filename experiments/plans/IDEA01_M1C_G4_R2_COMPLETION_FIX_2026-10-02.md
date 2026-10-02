# Text2CAD G4 R2：完成状态与共同deadline收口

2026-10-02。上位G4完整合同和用户七天授权继续；当前仅本地CPU、GPU0/approved none。基线R1代码226c4108f57e332ebfe967df24980ccc35c0595b，报告回填0f66086b1a5ad17b9bfded85e17fcff471fc5e19。按R2 Codex审查修两个已复现Important，不重新设计任务平台。

## 最小范围与验收

仅修改本轮新增text2cad_g4_launch.py、run_text2cad_g4_launch.py（必要时）、test_text2cad_g4_launch.py，写自己的新TEXT2CAD_G4_R2_IMPLEMENTATION_2026-10-02.md。旧scorer/loader/math/ledger/worker/config/env、Codex文档、旧报告不改。模型/BERT/NPU/下载/安装/远端checkout修改额度0。

1. 自有进程组及其后代必须按原共同deadline终止，TERM/KILL宽限在预算内。deadline+kill_wait_s只可用于退出记账/已发送终止后的确认，不能把真实后代settle/信号安排到截止后。修复须覆盖父近deadline退出、父先退出而peer仍活、最后一片退出后残留；不增加共同604800秒或336合计卡小时。默认普通双卡成功、一卡失败peer继续监管不变；不自动retry。
2. 只有最后controller_exit事件写失败也必须诚实失败，返回状态与持久JSON的错误/完成状态一致；旧fail-if-exists/真实exit保留。必要CPU只注入最后事件OSError，两个实际短CPU孩子退出，检查内存/持久状态都不伪称ok，磁盘完全不可写时仍非零且不留下活孩子。
3. Linux树见证先采集并断言controller返回时组/后代存活状态，后在finally清理测试自己创建的PID。不能由测试清理后再宣称controller完成收尾。Windows只报告明确PID-only降级，组mock不称真实Linux。非ESRCH probe错误不能误称empty，必要时同局部修复记错并failed（Minor，不扩框架）。

Root反例与驱动在checkout外D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_r1_codex_review_20261002/CODEX_G4_R1_REMAINING_GATES_20261002.py/.json，或experiments/runs/m1c_g4_cpu_review_r2_20261002/。原R1 post-spawn收尾保持；只跑新边界及必要原正常/拒绝/七核心AST/full argv回归，不重复G1/G2/G3、全部无关套件。

真实Linux必要CPU见证仍待Codex审码和单独授权：提供Git内标准库targeted测试/待审命令，不SCP源码、不自行fetch/detach或占卡；优先只PostSpawnFaultTest/ProcessGroupTest/CommonDeadlineTest及必要正常双卡/peer失败路径，避免依赖pytest和安装包。POSIX执行时输出真实PID/进程组、controller退出前后状态、共同deadline与原始日志；缓存/字节码/临时目录均在checkout外。保持所有源码本地编辑Git同步。

全full固定3644×8046=29319624对、每片1822chunk/14659812对/916466批、FP32/full272/b16尾14/无trim-cache/无probe-token-audit-resume均不变。七个旧评分核心AST与487d02b9一致。报告完整代码SHA/实际CPU命令与exit/原始路径、Windows actual/POSIX mock/未跑Linux边界。exact stage自己的文件、commit及报告SHA回填后停review_required，不push、不启动G4。真实API或必须改旧评分环境等阻塞保存证据停，不换模型/扩预算。
