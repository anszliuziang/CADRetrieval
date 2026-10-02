# Text2CAD G3：审核后双卡运行交接

2026-10-02。上位 `IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md` 与R1/R2准入修复继续有效。审查 `IDEA01_M1C_G3_CODEX_CODE_REVIEW_R3_APPROVAL_2026-10-02.md` 已通过；具体授权仅以最新版origin/master的ACTIVE_TASK为准。

## 准入和唯一范围

运行SHA **487d02b9ac92f5e2338718e45b1f023c308dd332**；run_id `idea01_text2cad_g3_probe_20261002`。每片50固定文本×完整8046图库，402300对/卡、总804600对；只有两次fresh scorer调用（npu:0/shard0，npu:1/shard1），batch16/尾14、FP32、完整272行、无trim/cache，不增加任何前向或预热。12卡小时**双卡合计上限**，单次timeout21600，总6小时墙钟上限；不扩大预算或自动重试。

## Git与环境

Kimi先本地读CLAUDE、ACTIVE和本合同。服务器仅Git fetch、干净detach批准SHA运行；origin/master含新授权，批准SHA的旧ACTIVE正文不是授权来源，runner已从fresh origin/master读取并比对完整SHA。检查工作树和运行SHA，不reset/clean或编辑远端源码。所有新源码/脚本只能本地经Git，本次无需写新源或重构启动器；远端shell只执行已审CLI、写命令/进程/结果证据到runtime。

计算唯一 `ssh yunnan-share2`，checkout `/home/ma-user/work/CADRetrieval`；RT `/home/ma-user/work/CADRetrieval_runtime`；PY `/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python`。复用env spec10c18c17，不安装/升级或额外NPU烟雾；两卡健康空闲、run目录无任何旧产物、`$RT/budgets/text2cad_m1c_<approved_sha>.json`不存在才启动。未知进程/脏Git/旧目录/ledger等停止保留证据，不删覆盖。

## 真实命令与记录

参考原G3实现报告第6节的score命令，两卡只改device/shard/out-dir；R2报告更正为**budgets**最终账本路径。必须使用Git内固定runner、checkpoint/BERT/CADzip原revision完整路径、worker四文件及token审计；`--probe-selection`固定Git配置；`--batch-size 16 --texts-per-chunk 1 --cards-used 1 --n-shards 2`。offline/cache和OMP/MKL/OPENBLAS=4同G2，不设置遮蔽设备编号的可见性变量。

保留完整实际CLI、env名称/版本（无secret）、HEAD/干净状态、PID/起始UTC、每卡stdout/stderr/真实exit code。进程可错开3秒避Git fetch锁；不以重复调用解决启动失败。用后台作业可保证SSH断开后继续，timeout真实退出码必须落盘；命令记录是证据，不在远端生成或编辑源码。短轮询确认两进程都真实计费、chunk/sidecar开始产生、npu-smi进程对应两卡后，写本地 `TEXT2CAD_G3_LAUNCH_2026-10-02.md`（事实/路径/PID/完整SHA/预算/未完成项），Git提交自己的新submission并退出交接给heartbeat，不等待数十分钟的同步sleep。

## 完成后的结果交接

两卡exit0、finalized后按R2报告取回 `$RT/budgets/text2cad_m1c_487d02b9ac92f5e2338718e45b1f023c308dd332.json` → run `final_ledger.json`。CPU运行 `python -m idea01.text2cad_g3_probe --mode eta-g3`，需四文件worker/token审计/repo-root/恰两card summaries/必填final-ledger，输出新ETA JSON，不覆盖旧产物；CLI失败或超七天保留现状报告，不放宽规则。不得读取probe分数后调score或选择规则，不把probe指标当正式全量。

保存两卡50+50完整chunks/timings/summaries、共享final账本、启动/环境/退出/前后Git与NPU证据、CPU ETA及失败日志。SHA256SUMS用LF，云脑/本地/OBS逐项核验且不以pipeline尾命令掩盖失败。取回本地 `experiments/runs/<run_id>/`，重要原始产物归档cvteam14共享OBS `/data/obs_data_liuza/CADRetrieval/runs/<run_id>/`，只用存储，不用其GPU；避/data/data_dev。

后续本地 `TEXT2CAD_G3_PROBE_RESULTS_2026-10-02.md`：事实/解释/建议分开，真实804600覆盖、成功/失败、卡时/HBM/每层最慢query/慢卡ETA/归档SHA及限制。Codex独立重算完整索引/数值/长度/计时/账本与ETA后才决定正式全量；Kimi不自行启动G4。
