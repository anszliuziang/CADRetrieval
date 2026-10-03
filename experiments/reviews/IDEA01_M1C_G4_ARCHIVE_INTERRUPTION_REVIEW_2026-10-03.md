# G4 原始审核进度与归档中断（2026-10-03）

Codex，2026-10-03 07:50Z。**评分本身正常完成；完整矩阵最终准入尚未批准。当前阻塞在 OBS 运输和 Kimi 收集交接，尚无检索指标。**

## 已核实的原件

- 历史运行 SHA `56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719`；run `idea01_text2cad_g4_full_20261002`。两片各1822 chunk / 14659812对 / 916466批、exit0/finalized；共同墙钟60019.42330670357秒，119885.30合计卡秒。完成退出和预算的17项直接预核沿用原始证据，不修改历史账本。
- Codex 两片标准库全行见证均真实 exit0，各14659812行/58644722项检查，合计29319624行/117289444项检查、0失败。覆盖全部稠密索引、六项有限分数、固定候选有效长度、四项分数算术、原 source/config 绑定。不是抽样审核。
- 见证 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g4_results_codex_20261003/codex_g4_raw_shard_audit.py` SHA256 `43bdb05e22ef0619ee5fcffa1069d302fcd667bc4b73238e65e841c8ca142f87`；两份 `CODEX_G4_SHARD0/1_RAW_AUDIT_20261003.json` 含全部3644个原始文件 SHA、byte、行数。原 session52601/8904 已完成，不重跑。
- `CODEX_G4_LOCAL_ROW_HASH_BINDING_20261003.json`：再核11054项通过；全部数值审核的文件 SHA 连接源清单、18份直接源端快照及11项固定资产身份。3666个 payload / 14858752230 bytes，manifest SHA `97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733`。本地额外 `local_verify_20261003.log` 明确为清单生成后审核输出，不当作评分原件；没有未知额外文件或缺失。
- Codex 云脑独立完整3666逐 SHA 核验真实 exit0，`source_hash_stdout.json` / `source_hash_exit.json`，session96800已完成。源机器与本地的完整评分原件均在。
- Codex 26份稳定见证已归档到本地 `runtime/g4_raw_local_source_bundle_20261003` 及共享 OBS `runs/m1c_g4_codex_rows_source_20261003`，manifest `2532bfd68cee4978faa30edec31178f9c208d2f2d60e91292a20f54675c04227`；运输、本地和 OBS 26项逐 SHA 均真实 exit0。OBS Python3.8 不支持 Path.is_relative_to 的首次诊断失败已保留，换等价 Path.parents 判定后验证通过，未改变任何数值门。

## 现在确认的中断

1. 原统一 exec session75588 不再可访问，原记录 PID10268不存在。原 CLI 日志最后写入2026-10-03T07:34:43.299781Z，未记录 `result` 完成事件；没有可核实的正常 CLI 退出码。系统原始 CLI session_id 为 `4926b2d4-ecd2-4950-b2d4-6d98eac53ed3`。只读取工具调用/结果和会话元数据，未输出内部思考或凭据。
2. 本地进程核查未找到本次 k3[1m] print CLI 或 tar/bash 运输进程。cvteam14 07:47–07:50Z 的 CPU `ps` 核查无 tar 接收器。没有假定它们仍在正常运行，也没有重复派发。
3. OBS 仅可列出 card0 的602、card1 的527个 chunk 文件名，共1129/3644；这是文件名计数，**不代表这些文件 byte 完整或 SHA 通过**。最后 Kimi 运输进度记录约4.43GB/29.8%，不是当前完整性结论。8路运输的最终双侧 PIPESTATUS 没有交接，不能补造 exit0。
4. cvteam14 的只读 SSH `date` 与 CPU `ps` 均 exit0，说明主机 SSH 当前可达。一次枚举并逐文件 stat 的诊断40秒超时已记录；改为只列名字的诊断约10秒 exit0。没有据此声称 SSH 服务故障或 OBS 全面不可用。中断原因尚未确定，不把账号切换/应用重启当作已证实原因。
5. Kimi `TEXT2CAD_G4_FULL_RESULTS_2026-10-03.md` 仍是未提交草稿，其 OBS 项尚未定稿；原 stdout 无 API 失败，stderr只有73B旧SDK模型名称告警。不能将草稿当作完整交接或三处归档通过。

诊断原件在 `runtime/g4_results_codex_20261003/` 的 `obs_process_snapshot_0748.json`、`obs_process_snapshot_exit_0748.json`、`obs_names_only_0753.json` 和本地 `.cache/m1c_kimi_g4_results_stdout_20261003.jsonl`。40秒超时的原工具输出亦保留；不存在正常退出证据就如实写缺失。

## 裁决与下一步

用户要求真实退出/完整性阻塞时保存证据、停下来交流；原收集合同禁止覆盖未知或哈希不同的既存资产。因此当前停止推进，不启动 G5、不重跑评分、不清理或覆盖部分 OBS 目录、不重新派发 Kimi。源端、本地全量原件与 Codex 全行审核均安全。

待讨论的具体恢复方案见 `experiments/plans/IDEA01_M1C_G4_ARCHIVE_RECOVERY_PROPOSAL_2026-10-03.md`：保留部分 OBS 现场，在新的目录完整归档并核验。**没有任何新的模型/NPU 计算需求。** 只有归档、运输退出和 Kimi 报告全部通过，才另写最终矩阵准入和 CPU 评价合同。
