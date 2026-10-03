# G4 Codex 接手：完整归档与后续评价

2026-10-03。用户明确授权 Codex 接手继续，原 ARCHIVE_RECOVERY_PROPOSAL 从待定转为获准执行。实现者 Codex；新增代码由独立只读 reviewer 审核。零模型/BERT/NPU前向，零新增模型/数据下载。

## 归档实现合同

源为本地 `experiments/runs/idea01_text2cad_g4_full_20261002/`；3666 payload /14858752230 bytes /3644 chunks；SHA256SUMS 固定 `97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733`。旧 OBS 原 run 目录保留。新目标唯一为 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002_recovery_20261003/`，使用独占 mkdir；已存在则失败，不能覆盖或重试。

本地标准库后台控制器独立于 Codex/Claude 短期工具会话运行，Windows Start-Process Hidden 启动，日志在 checkout 外 `runtime/g4_archive_recovery_20261003/`。源码放 `experiments/scripts/run_g4_archive_recovery.py`；本地 Git 提交完整 SHA 与干净 tracked tree 绑定，独立审码后才能运输。不修改云脑 checkout；远端只执行系统 tar/mkdir/sha256sum 接收和只读核验，不写项目源码。

只读解析并校验固定 LF manifest、全部安全相对路径/禁止 symlink、自引用和重复、固定数量与字节、源端逐 SHA；分成最多 8 路 gzip tar，清单只含固定 manifest payload，SHA256SUMS 单独随第一路运输。记录每路 argv、sender/receiver PID、两端 stderr、真实退出。控制器终态不得用接收端成功掩盖发送端失败；任何一路失败保留新目录和现场，不自动重试。通过后执行远端 sha256sum -c，记录真实退出；Codex 再用已有独立 Py3.8 witness 核全部字节、manifest 覆盖和未知文件。运输日志不进入 raw manifest。

必要 CPU 测试：路径/重复/manifest拒绝、旧输出目录拒绝、真实 producer/receiver 成败组合、八片无遗漏/无重复；不重复旧 scorer/G1–G4 测试。记录 RED→GREEN 和独立 review。长传输健康时仅查询持久状态和 PID，不重复启动。

## 既有证据与验收

历史评分 SHA56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719，29319624 对 /119885.30 卡秒 /16.672062 小时墙钟，两 scorer exit0/finalized、controller ok。两片全行审核117289444项、源端/本地3666 SHA 和11054连接审核通过。复用这些证据，不重评分、不重核全行。

归档成功必须同时满足两端全部运输退出0、远端全3666 sha256sum exit0、独立 OBS 字节与覆盖通过、已完成全行哈希绑定。Codex 新 submission 单列接手/中断事实与 canonical OBS 路径，保留 Kimi 草稿。写最终矩阵 review 后才激活 G5：7934 原 query exact UID 指标，sum 主表、mean 诊断、cad_index 升序 tie、随机/长度基线、重复上限/分层和文本组 CI；先冻结评价合同，再读 qrels/指标，不看 test 改方法。

## 执行记录

- 初始 BASE ff12ea325659df261c20a14121583cd29c41af50；没有活跃 Kimi/旧运输。用户新授权取代历史等待决定和 Kimi 唯一实现者约束。
- 尚未启动新运输；先实现、测试、审查、Git 提交，随后写完整 approved_sha。
