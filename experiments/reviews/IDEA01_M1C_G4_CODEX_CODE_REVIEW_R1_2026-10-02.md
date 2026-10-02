# Text2CAD G4启动控制：首次Codex审查（需修复）

2026-10-02。Base c9af3b92419306b217674343780fda7f9a1dab41；代码0bba6a7acb573aefbe1797a043b6dafe974e61e8，报告5acb984a60d58aff26ec5a47322f74cad7a029cf/回填877fa37bda2abf0558380dfc4524cb5f257a0dde。当前没有NPU授权。

## 已通过与裁决

固定full argv、两片数学、SHA/只读Git/ACTIVE授权及fresh门结构已检查；Root实际52测试+6 subtests通过（33.49秒）。七个旧评分函数类的AST与487d02b9完全一致，既有runner文件未改。新控制器未在真实服务器执行，远端仍历史G3干净SHA。

**不批准正式G4运行。需修复以下两个生命周期问题；不是评分模型、资产或ETA门失败。** requesting-code-review要求的唯一fresh reviewer已实际读控制器/测试与合同，无二次委派。Root另用真实CPU短子进程复现post-spawn故障；零NPU/模型前向，结束后只清理自己创建的CPU孩子。

## Important 1：spawn后写证据异常，进程失去deadline监管

文件experiments/src/idea01/text2cad_g4_launch.py：launch()的spawn try/except（约804–840行）、launch record写入（约874行）及主wait loop。

pid/time/event写都在Popen之后，任一失败被当成spawn_failed，立即写card.exit_code=NONE，但proc仍在pending。随后实际reap再次写同一exit文件，fail-if-exists触发RunnerError，整个controller退出；同伴可能仍活着。若controller_launch.json直接写失败，异常也越过wait loop而退出，两卡仍运行。

Root见证：注入card0.pid.txt写OSError→异常为已有card0.exit_code拒绝覆盖，card0记录NONE、peer仍运行、无controller_exit；注入controller_launch.json写OSError→两个真实CPU子进程都仍运行、无controller_exit。证据在checkout外CADRetrievalIdea1_runtime/g4_launch_codex_20261002/root_g4_fault_probe_20261002.py与CODEX_G4_POSTSPAWN_FAULT_PROBE_20261002.json。

修复应把Popen成功与后续记录失败分开：活进程不能预写假exit；成功spawn后的一切异常都须有生命周期收尾，保留PID/真实退出和错误。不能因日志、记录、最后Git读取失败而留下无监督的已启动作业。可在异常时结束本controller启动的作业，尽力保留失败证据和stderr；不要求磁盘完全故障时保证写文件成功，不允许因此伪称正常完成。普通单卡评分失败仍保留对卡的已授权正常监管，不自动retry。

## Important 2：deadline只杀父PID，后代可继续运行

同文件约889/893行使用Popen.terminate()/kill()；start_new_session=True创建了独立session，但未终止其进程组。wait loop仅以父PID.poll()移除pending，父退出后后代不再监管。

Fresh reviewer真实CPU父子进程两路径见证：共同budget缩短至0.9秒，controller约0.617秒返回failed、父deadline_enforced，但后代返回后仍写标记；正常父exit0时controller约0.162秒返回ok，后代仍在返回后写标记。见证平台Windows；生产Linux源码同样只对父PID发送信号，尚无真实POSIX组收尾证据。此处应对自己创建的进程组完成Linux TERM/KILL与父退出后的后代清理，不能杀其他人的进程；维持共同deadline及预算内宽限，真实退出不可被spawn成功替代。

## Minor与操作限定

- npu-smi busy正则期待空格分隔的三数字，真实表格列带竖线，因此现test_process_row_refused靠缺No-running标记间接通过。当前固定单NPU2双chip忙时仍会因缺标记拒绝，**不单独阻塞本机G4**；可同patch小幅修正列解析与混合marker/busy测试，不能借此泛化资源平台。
- 报告nohup命令未给controller自己设置PYTHONDONTWRITEBYTECODE=1（只child env有）。正式执行合同必须在启动controller前设置，cache/tmp同样留checkout外；可改自己的新R1报告示例，旧报告原文保留。
- 1037行控制器与1122行测试已超出预期最小体量；本轮不因此重构，修复限于上述生命周期，不再扩功能。

## 下一步

IDEA01_M1C_G4_R1_LIFECYCLE_FIX_2026-10-02.md定义最小本地修复，GPU0/旧scorer不变。只复审两生命周期和必要新CPU拒绝/真实退出/组收尾证据；任何POSIX必要见证先提交Git内代码/待审命令，Codex另授权远端零NPU CPU测试，不绕过本地Git协议。唯一fresh reviewer最终6测试+2 subtests已回报通过正常契约，但明确上述两Important阻塞；未判断真实Linux生命周期/NPU吞吐/科研结果，仅对七核心AST作未改验证。不新增第二席位。


## 原始证据归档

5个原始CPU证据文件（Root baseline/post-spawn probe+结果、fresh进程树probe+结果）镜像于本地experiments/runs/m1c_g4_cpu_review_r1_20261002/与cvteam14:/data/obs_data_liuza/CADRetrieval/runs/m1c_g4_cpu_review_r1_20261002/。LF SHA256SUMS的SHA为debfa27a81b306ab3569e2e17688a5989237268f6a350d7c778544cb72b1f3f4，本地5项逐SHA通过，OBS sha256sum -c五项通过且exit0、manifest一致。探针源码为CPU见证存档，不是通过SCP部署实验源码；服务器checkout未变。
