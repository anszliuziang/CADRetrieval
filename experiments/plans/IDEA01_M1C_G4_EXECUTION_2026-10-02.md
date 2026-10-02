# Text2CAD G4：正式全量执行与启动交接

2026-10-02。上位IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md不变。R3代码及真实Linux22项CPU已通过；IDEA01_M1C_G4_CODEX_FULL_APPROVAL_2026-10-02.md批准完整56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719。只有ACTIVE明确pilot_authorized及同一SHA/预算后才执行。执行代理本地Claude Code/官方Kimi K3；Codex独立监控和审核。

## 固定范围

- run idea01_text2cad_g4_full_20261002；完整3644唯一文本×8046 CAD=29319624对。每卡1822个query chunk/14659812对/916466批；后续恢复7934原查询UID指标。
- 仅yunnan-share2两卡npu:0/1。FP32/full272/batch16尾14/无trim-cache/无probe-selection-token-audit-resume-cache；两次fresh scorer，不预热/重跑/空查询。模型、BERT、worker、cadzip和原评分资产/pins全部与G3相同。
- 硬共同604800秒墙钟/336合计卡小时=1209600卡秒。保守估计max(G2,G3)=38.28394小时；估计与截止不同。TERM/KILL预算内，晚启动第二卡不增加墙钟。
- checkout /home/ma-user/work/CADRetrieval；RT /home/ma-user/work/CADRetrieval_runtime；PY /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python。环境spec10c18c17 warm reuse/offline/线程4，不安装包或改变device可见性。cvteam14只共享OBS，避/data/data_dev。

## 启动步骤（不写服务器源码）

1. 本地读CLAUDE/ACTIVE/本合同/Full Approval；只写自己的新TEXT2CAD_G4_FULL_LAUNCH_2026-10-02.md。无需重新实现或重跑旧门。
2. 远端已有CPU验证的干净56ac0f8；先核HEAD/status，再git fetch origin，核最新origin/master ACTIVE的pilot_authorized/同一SHA/max_scored_pairs29319624/gpu_budget336/run一致。若HEAD非批准值，仅按Git干净detached批准SHA；未知脏树停止，不reset/clean/删除旧产物。
3. 启动前核NPU两卡健康空闲、runtime/local/共享OBS各>=100GB；正式run根与budgets/text2cad_m1c_56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719.json及锁不存在；旧G1/G2/G3/CPU目录不复用/改写。实际asset绑定仍由每fresh scorer已审门校验，保存启动前指纹/环境与授权来源证据，不做新模型验证前向。
4. 只从Git批准的experiments/scripts/run_text2cad_g4_launch.py启动。controller自身import前PYTHONDONTWRITEBYTECODE=1，HF_HOME/XDG_CACHE_HOME/TMPDIR指checkout外。用nohup/stdin DEVNULL保证SSH断开后继续，日志/初始controller PID落RT/runs之外层专属记录，不生成新shell/源码。

```bash
PYTHONDONTWRITEBYTECODE=1 \
HF_HOME=/home/ma-user/work/CADRetrieval_runtime/hf \
XDG_CACHE_HOME=/home/ma-user/work/CADRetrieval_runtime/xdg \
TMPDIR=/home/ma-user/work/CADRetrieval_runtime/tmp \
nohup /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  /home/ma-user/work/CADRetrieval/experiments/scripts/run_text2cad_g4_launch.py \
  --approved-sha 56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719 \
  > /home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g4_full_20261002.controller.nohup.log 2>&1 < /dev/null &
```

不要第二次运行该命令；controller创建正式run的原子fresh guard、两条full固定argv、实际PID与共同deadline。原子记录/退出由已审代码完成，不在服务器热补脚本。controller外层日志/PID也须fresh检查，不覆盖已有文件。

5. 只读确认controller及两scorer PID/命令、同一SHA和完整argv无probe；新账本恰两invocation、cap29319624/1209600，均cards_used1。至少两卡各首个完整chunk与非有限/拒绝0、实际mtime、NPU进程/设备对应、共同deadline、干净Git。正式full路径不生成G3的timing_sXX.jsonl；以mtime观察产出节奏，以final ledger/controller退出核总耗时，不声称逐query计时或冷启动。不是spawn成功就宣称全量成功。遇真实失败记录并停下交流，不重启/重复评分；单卡失败peer由既有controller持续监管，不擅自杀正常作业。
6. 本地自己的新LAUNCH报告列实际UTC/PID/invocation/授权ref/argv/ledger/首chunk/日志/源资产/限制、完整代码与报告SHA。exact stage自己的报告、commit及SHA回填后停止交接，不push、不改Codex文档/源/配置。Codex随后只读审核启动并推送报告。

## 运行后

每30分钟只读监控，状态不变无用户行动保持安静。运行中的远端checkout固定56ac0f8，不fetch/detach；原始产物和ledger进度真实落盘。完成两片各1822chunk/两exit0/finalized/全29319624对及零拒绝后收集完整矩阵、final ledger、环境/资产/控制器退出/计时与失败日志，本地+OBS逐SHA核验，再按上位合同激活独立CPU评价。CPU评价实现可后续另立本地合同，不与启动交接争用Git index。
