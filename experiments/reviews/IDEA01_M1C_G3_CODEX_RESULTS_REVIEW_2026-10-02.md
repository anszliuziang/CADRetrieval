# Text2CAD G3：Codex完整原始结果审核

2026-10-02。run idea01_text2cad_g3_probe_20261002；实际执行完整SHA 487d02b9ac92f5e2338718e45b1f023c308dd332。两卡invocations为20261002T075359Z-4037ac89、20261002T075359Z-09217532；代码准入见R3_APPROVAL审查。Kimi的新TEXT2CAD_G3_PROBE_RESULTS_2026-10-02.md首次提交de02bb4aa248dc37c3f4077b70193d5681d2e807、SHA回填2b006501d42607fd62d59c1ed4204306f2be5625，已逐项对照其关键数字与原始产物；本裁决依据完整原始结果。

## 裁决

**G3完整原始结果通过。G2和G3保守ETA均低于七天，可按用户既有授权进入G4本地启动实现/CPU审查；当前不批准新NPU前向。** 独立审核3219147项全部通过，完整804600行覆盖全部固定100文本×8046图库；零拒绝，六项原始分数均有限、有效字段计数和sum/mean算术一致。评分数学仍依赖已审G1原始logits与生产源码；本次无G3原始logits，不声称重新前向或复算了每个log_softmax。

| 真实证据 | card0 / npu:0 | card1 / npu:1 |
| --- | --- | --- |
| 完整文本chunk / timing | 50 / 50 | 50 / 50 |
| 实际评分对数 / 批数 | 402300 / 25150 | 402300 / 25150 |
| 实际退出 / 账本 | 0 / finalized | 0 / finalized |
| 卡秒 | 1652.94 | 1662.96 |
| 冷启动秒 | 33.16292490391061 | 32.88705727108754 |
| 峰值设备内存bytes | 1893844992 | 1894202880 |
| 全量计算秒（各层最慢query×完整层数） | 79864.10362434387 | 76347.37244844437 |

最终共享账本恰两次fresh调用，804600对/3315.90合计卡秒（0.9210833卡小时），与两summary各自真实invocation绑定一致；授权12合计卡小时、快照及final cap=43200秒相等。每query串行时间轴与invocation边界一致，实际两卡评分窗口重叠比例1.0，非“文件不同”或理想乘二推断。

## 独立选择、数值与ETA

脚本只用Python标准库，未导入生产评分器/ETA模块，零模型、BERT、NPU前向。从原worker哈希、固定token审计和盲选配置重算：每片1822文本按(token长度,index)排序，round(1821*j/49)取50个，global tier=floor(3*rank/3644)，配置SHA f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985。每卡完整tier数605/613/604及610/602/610，probe各17/16/17。worker四文件与token审计实际SHA均逐项核验。

逐行核查全100个chunk：实际8046稠密CAD索引、全局pair_index、query范围、header/代码/资产/精度/score与probe绑定；六项有限非正分数、独立CAD长度表、score_total=x+y及各mean关系；每query有效字段合计439091，所有timing统计/真实批数503一致。strict load实际172 tensor/零missing-unexpected/数值相等、eval+冻结、原checkpoint和BERT五文件测量SHA、8046/8046候选预载均通过。

ETA逐卡逐tier取最慢完整query的E2E（包括评分、序列化、原子写盘），乘该卡完整tier查询数，加本卡实测冷启动；两卡较慢者×1.30。独立结果103866.44651402211秒=28.851790698339475小时，与已审eta-g3实际输出逐值一致，真实CLI exit0，必填final-ledger。G2独立137822.16767930926秒=38.283935466474794小时也通过；G4计划采用两者较大估计38.28394小时。不是正式全量完成时间保证。

## 本地与OBS完整性

原始本地experiments/runs/idea01_text2cad_g3_probe_20261002/，云脑runtime同名run，OBS cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g3_probe_20261002/。完整130项SHA256SUMS为LF，manifest SHA 59991e8a7e21a4ad664a7ed5d605fd30d89a8b89c628f5c2e337495e43e3702f。Codex独立本地130项全部通过；OBS执行无管道掩盖的LC_ALL=C sha256sum -c --quiet SHA256SUMS，真实exit0、同manifest SHA。原始总大小407854259 bytes（含manifest）。七份Codex在收集前直接取得的summary/timing/exit/final ledger与Kimi完整镜像逐SHA一致。

Codex独立脚本、原始直取快照及JSON共12文件另归档本地experiments/runs/m1c_g3_codex_results_check_20261002/和OBS同名run，不改变原始130项manifest。独立manifest SHA 4aa1c83ac150a5f7907ad61444014bbe48884778696f33f1f5c8675c41918fd0，OBS严格逐SHA exit0；full审计JSON为CODEX_G3_FULL_RAW_AUDIT_20261002.json，哈希及7快照核验为CODEX_G3_HASH_AUDIT_20261002.json。

操作记录：独立hash审核最初按字符串比较“./path”和“path”产生覆盖误报；规范化相对路径后130项及实际文件集一致，未改原始文件。Kimi归档曾纠正SHA清单临时文件放置，最终正式清单已独立核验；不将所有后处理描述为无操作错误。服务器完成后干净detached原SHA、NPU健康无进程，post-run版本与spec10c18c17环境一致；已留资产13项后验哈希及命令/wrapper/exit证据。

## 结论的限度与下一步

执行者报告的限定：503是每query批数，正确对数公式为8046=16×502+14；约248对/秒为本次观察值，不构成未来吞吐下限，按抽样最慢值计算的ETA也不是数学上界。G3的固定盲选清单仅属于probe，不能把建议中的“保持盲选清单”带到G4；正式全量不传probe-selection，覆盖全部3644文本。真实sum与mean、计时和归档数字均与独立复算一致，这些文字限定不改变准入。

本轮只有工程持续运行与完整性结论，没有检索质量/R@K或语义正确性结论。唯一分数/FP32/full272/b16/无trim-cache不变，batch32失败保留。接下来由IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md细化全29319624对：本地最小控制器和必要CPU测试、Codex审码、完整SHA逐阶段授权、Git部署两卡，完成后恢复7934原查询UID指标。原版label smoothing双头分数不改称纯MLE或CAD几何NLL。
