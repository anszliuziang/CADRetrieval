# Text2CAD G4 R1：启动生命周期最小修复

2026-10-02。上位完整G4合同与既有用户七天授权继续；本轮仅本地CPU，无NPU授权。首次审核IDEA01_M1C_G4_CODEX_CODE_REVIEW_R1_2026-10-02.md中两Important须关闭；正式全量/score/资产/预算均不改变。

## 精确范围

只可修改本轮新增的text2cad_g4_launch.py、run_text2cad_g4_launch.py、test_text2cad_g4_launch.py，并写自己的新TEXT2CAD_G4_R1_IMPLEMENTATION_2026-10-02.md。禁止动其他既有scorer/loader/math/ledger/worker/config/env、Codex文档、旧报告。不安装依赖、不真实BERT/模型/NPU，不fetch/detach远端，不创建远端G4产物/账本。

1. spawn成功后pid/time/event/launch_record/exit_record等I/O或最后Git读取异常均不能让已启动子进程失去监管。与spawn本身失败区分，不给活进程预写NONE，不二次覆盖真实退出。异常路径可靠收尾本控制器启动的进程、尽力保留错误/PID/真实退出/stderr；全磁盘不可写时允许写失败但仍须收尾并非零退出。普通单卡scorer失败仍保持peer正常监管、不自动retry。
2. Linux对自己start_new_session创建的进程组落实TERM/KILL和父退出后的后代清理；共同deadline/预算内宽限不变，不能kill其他作业。Windows本地CPU路径可以明确降级/替身，但不据此声称POSIX真实通过。
3. 必要新CPU证据：Root两种post-spawn写失败；日志/记录/末次Git失败时已启动孩子不遗留；真实父+后代在正常父结束和deadline两路径均收尾；普通card失败/peer成功与两card成功保持；真实退出和共同deadline不因修复放宽。不重跑G1/G2/G3或无关全套。
4. 仅小幅处理npu-smi带竖线进程列误检及其测试（Minor，可同patch修正，不独立阻塞固定本机）；新报告的controller启动命令注明PYTHONDONTWRITEBYTECODE=1及checkout外cache/tmp。

实现保持现有full固定argv/29319624对/b16/FP32/full272/无trim-cache/无probe-resume，七个旧评分核心AST与487d02b9一致。只提交自己的改动，完整SHA+真实测试exit+原始证据路径；报告清楚区分Windows实际测试、POSIX mock和未运行的Linux见证。若需真实POSIX CPU见证，先提供Git内最小测试和待审命令，Codex审核SHA后另授权远端零NPU执行，禁止擅自换checkout或从SCP发源码。

CPU预算不限于已有小测试的必要复现，不允许模型前向或其他占卡操作。任何API/连接/必须改旧评分环境的真实阻塞保存证据停下，不换模型或扩预算。提交新报告及完整SHA回填后停review_required，不push/不启动G4。
