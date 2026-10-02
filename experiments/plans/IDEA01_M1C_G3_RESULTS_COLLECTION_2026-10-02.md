# Text2CAD G3：完成后仅CPU收集与分析

2026-10-02。G3运行 `idea01_text2cad_g3_probe_20261002` 在干净detached `487d02b9ac92f5e2338718e45b1f023c308dd332` 完成。Root只读确认两卡50+50 chunk/timing、exit0、finalized各402300对/25150批，无失败，最终总804600对、3315.90卡秒。后续仅收集原始证据/CPU ETA/报告，不做任何真实模型/BERT/NPU前向或重复评分。

## 事实与范围

- card0 invocation20261002T075359Z-4037ac89，card1 20261002T075359Z-09217532；真实card seconds1652.94/1662.96，cold33.16292490391061/32.88705727108754，峰值HBM1893844992/1894202880 bytes。
- 上位G3/R1/R2合同和R3代码审查继续适用于结果；上述运行summary不是完整数值/索引审核，Codex完成独立原始结果核验后才裁决全量。
- 当前ACTIVE禁用新前向，approved_sha=none、GPU0；历史summary/账本仍绑定原487d02b9和12卡小时授权，不能改写它们使之与当前状态相等。

## Kimi任务

1. 只读复核两summary/真实exit/最终账本/无NPU进程/Git干净且原SHA。不要更换checkout或新fetch/detach以运行模型；只用既有Git内已审纯CPU入口。
2. 将完成后稳定账本 `$RT/budgets/text2cad_m1c_487d02b9ac92f5e2338718e45b1f023c308dd332.json` 取为run `final_ledger.json`。若文件已存在，核SHA相同后复用，不覆盖不同内容。CPU `eta-g3` 必填该文件、两summary、token审计、四文件worker/repo-root；输出run `eta_g3_20261002.json`，保留真实CLI/stdout/stderr/exit。ETA失败或>604800保存原始证据停下，不改代码/系数/时序/上限或重跑评分。
3. 补实际结束时Git/NPU/版本/环境的只读证据（不记录token/key/env secret）。无需安装依赖、重新tokenization或前向。
4. 全部100完整chunks、两timing/two summaries、最终ledger、启动命令与包装执行记录/所有stdout/stderr/真实exit/环境/前后Git-NPU、ETA原始输入输出统一LF SHA256SUMS；取回本地 `experiments/runs/idea01_text2cad_g3_probe_20261002/`，归档cvteam14共享OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g3_probe_20261002/`。云脑/本地/OBS逐项SHA核验，不用pipeline尾命令掩盖失败；OBS仅存储，避/data/data_dev。不同已有产物保留并报告，不删/覆盖未知内容。
5. 仅写新的 `TEXT2CAD_G3_PROBE_RESULTS_2026-10-02.md`（事实/解释/建议分开），完整代码/资产/选择SHA、实际数量/覆盖/对数/批数/冷启动/卡时/HBM/真实并发/每层最慢query/全量ETA与七天门、归档SHA及限制。只有工程验证，不能报probe R@K为正式结果或按分数调整方法。准确3.6e3×8046完整A保留。
6. exact stage自己的新submission，commit后停止results_submitted；不push、不改Codex文档/src/config/tests、不启动G4。Codex审原始结果和独立ETA后才另批准全量。

运行环境warm reuse不改；本地编辑/Git同步源码规则继续。任何新的启动控制脚本均应先本地写/Git审，而非在远端改代码。本轮只收集已经留下的runtime包装命令和真实退出状态，不改变历史运行。
