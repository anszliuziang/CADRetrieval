# G4 完整精确矩阵最终审核

结论：**通过，full_matrix_approved=true。仅允许进入既有预注册 G5 CPU 评价；不授权新评分。**

## 审核对象与事实

- 原 run `idea01_text2cad_g4_full_20261002`，历史批准评分 SHA `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`。
- 3644 唯一 strip 文本 × 8046 CAD = **29319624 对**；每片 1822 完整 chunk、14659812 对、916466 批，两实际 exit0/finalized、零拒绝。
- 两片原始逐行审核共 **117289444 项检查，0 failure**；覆盖全部 29319624 行、稠密索引、六项有限分数、有效长度及四项算术等式。已完成的数值审核未重复运行。
- 本地原件与数值行哈希连接审核 11054 项，通过；18 直接完成快照、11 固定输入、干净实际评分 Git、原始命令与外层启动 provenance 均通过。
- 稳定账本扣账 29319624 对、119885.30 合计卡秒；实际墙钟 60019.4233067 秒（16.672062 小时）。共同七天/336 合计卡小时上限内；controller ok、两自有组 empty、无后代/事件写入错误/截止违约。

## 三处完整原件与运输

冻结 LF manifest SHA：`97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733`。

共 **3666 payload、14858752230 bytes、3644 精确 dense chunk**。云脑源端及本地已通过；本轮 OBS v2 最终独立 witness 也逐字节通过：actual SSH/wrapper exit0、all_passed=true、verified_files=3666、无缺失、无额外文件。

- 本地：`experiments/runs/idea01_text2cad_g4_full_20261002`。
- 源端：`yunnan-share2:/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g4_full_20261002`。
- 权威 OBS：`cvteam6:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g4_full_20261002_recovery_v2_20261003`。
- v2 运输执行代码 `8bc4eeabea57f416982a286add9b65ba35f57ed1`，历史授权 ref `4501fa01b0b3d9cf439daa2dbf409bd0bc20e62c`。显式预建全部父目录；8 路 sender/receiver 均真实 exit0，controller0/远端 sha256sum0。
- 独立 byte witness SHA `43db1b90be46f346cdf6a9bddf83717a7b5d777814631e9d70172f1b654a1323`，11:29:27Z 启动一次，9598.20 秒后结束；没有因慢 OBS 重启或重复审计。

## 最终证书与原始证据

`IDEA01_M1C_G4_MATRIX_CERTIFICATE_2026-10-03.json` SHA **afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6**。最终连接 **7359 项检查**，固定 manifest 将缓存的全行数值哈希与三处字节审核连接；证书记录实际证明文件 SHA/size、原 run/代码/预算/退出。

Root 原始 witness 位于 checkout 外 `CADRetrievalIdea1_runtime/g4_results_codex_20261003`、`g4_archive_recovery_v2_20261003`、`g4_obs_independent_audit_v2_20261003`。已有26份原始数值/源端证据 OBS `runs/m1c_g4_codex_rows_source_20261003` 已独立逐 SHA 通过。新的归档恢复/诊断/运输/独立见证另存独占小型证据包并逐 SHA 验证。

两次最终连接工具中的路径/输出格式拒绝是证据整理错误：manifest 原始路径带 `./`，以标准路径规范化连接；原运输 wrapper 将本地化 SHA 状态词解码成替换字符，原件保留，按 exact 路径覆盖、实际退出和独立完整 byte witness 证明成功，不猜造 `OK` 文本。首次反例、修正程序均保留；没有重新评分或重跑全行审核。

## 失败与研究范围

原 Kimi 收集 CLI 未形成完成事件，原因未定；其草稿与旧部分 OBS 现场保持。Codex 第一归档恢复的并发隐式父目录 ENOENT 经相同 payload 的两条件真实实验定位，显式预建通过；SSH banner timeout 独立保留。没有删除/覆盖旧资产。用户已授权 Codex 接手，故完整交接以新的 CODEX 记录与证书为准，不要求 Kimi 补交。

本审核尚无 R@K；成功矩阵不证明检索质量。随后仅按已冻结 G5 协议恢复7934原查询，sum 主表、mean 独立诊断、固定 tie/基线/重复上限/文本组加权 CI，不看 test 调方法。native 双头训练含 label smoothing，不能称纯 MLE 或 CAD 几何 NLL；full 没有 G3 timing/cold/peak 字段，不补造。
