# OBS 双条件接收诊断代码审查

2026-10-03，Codex接手并依据用户“之后这类问题全部授权，无需询问”继续工程恢复。批准代码：`dd14d4ca7abbda4eb47e27a47e19cd1bd5661ca9`，仅diagnostic模式，零NPU，固定两个条件各一次，合计不超过16MiB。此批准不代表OBS通过或完整矩阵已归档。

## 审查与证据

独立只读reviewer `/root/g4_archive_review` 审查cd843820实现，再针对dd14d4c最小修复复审；最终Critical/Important/Minor均无未解决项。Root11项必要CPU测试真实exit0，日志在checkout外 `CADRetrievalIdea1_runtime/g4_obs_diag_code_review_20261003/targeted_tests_r1.log` 与exit JSON。

原Important：implicit条件8路退出0但远端SHA失败时异常提前结束，不能执行prepared。Root回归RED后修复：保存真实sha256_exit与ok=false，继续预定义独占prepared条件。reviewer独立实际子进程SHA9/7反例证实两条件均落盘、prepared_passed=false/controller exit1，full在读源/远端操作前拒绝。

准入：新目标独占mkdir；prepared在spawn之前从清单预建所有父目录并test -d；两端实际PID/argv/stderr/exit保留；完整SHA非零不通过；ACTIVE archive_mode绑定diagnostic/full；full必须同代码SHA的真实prepared通过。源代码仅本地，远端只系统工具，旧部分目录保留。

实际挂载只读findmnt exit0：`/data/obs_data_liuza`为`fuse.obsfs`；日志同证据目录。旧5路首payload ENOENT支持父目录并发作为待测假设，尚未确认根因。此次两条件：implicit自动建父目录，prepared先显式建全部父目录；16×65536字节/条件，两个条件2MiB加清单，低于16MiB。

## 执行后门

真实diagnostic退出、16个pipeline的双方状态、各条件字节SHA/覆盖均须核对。prepared通过后才另写full授权，目标新`_recovery_v2_20261003`；完整3666payload/14858752230字节独立OBS审核与已有全量数值hash连接通过后才最终matrix批准/G5。真实挂载行为及外层后台监管留到执行审核，不以本地toy替代。

## 存储入口恢复复审

第一次controller21036在09:13:12Z启动，20秒后SSH banner timeout/255；诊断root命令未认证执行、payload没有发送。`g4_obs_receiver_diag_20261003`真实state=failed/controller_exit1与stderr保留，不复用；这不是OBS写入失败，不能推断父目录假设。

Root随后只读cvteam14重连同样banner timeout。cvteam6 findmnt同路径fuse.obsfs成功，既有26证据包SHA256SUMS实测2532bfd68cee4978faa30edec31178f9c208d2f2d60e91292a20f54675c04227，与共享OBS既有pin一致，新diag v2/full v2目标都不存在；原始证据在同code_review目录cvteam6_shared_obs_preflight.log/exit0。

唯一新增存储路由实现`8bc4eeabea57f416982a286add9b65ba35f57ed1`由同一独立reviewer只读复审通过，Critical/Important/Minor无新增；Root11针对CPU exit0，代码字节等Git blob。SSH和destination记录统一cvteam6，本地/远端diagnostic用新v2独占目录，full读取同新目录且同批准SHA。仅零NPU存储，无计算/评分/协议变化。此节批准8bc4eea的diagnostic，旧dd14d4c批准不再用于重启旧诊断。
