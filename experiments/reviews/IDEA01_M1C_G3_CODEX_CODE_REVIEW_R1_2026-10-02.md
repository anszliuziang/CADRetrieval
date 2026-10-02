# Text2CAD G3：Codex第一次代码审查

2026-10-02。Base `7d117b38b1790f1881c825ff160e8f09f71f529c` →实现 `759691b9c6781b5a21401d8e63b427521d259bc2`；Kimi报告doc `5422ed8ec4229abafc546dfbb2890b3bcc11994b`。按requesting-code-review技能派发全新只读审查者，使用指定SHA的checkout外快照；未SSH、未真实模型/NPU前向、未改Git/源码。

## 裁决

**暂不批准G3上卡。** 盲选与原生score数学正确，但ETA证据准入不完整，须按 `IDEA01_M1C_G3_R1_FIX_2026-10-02.md` 修复并重审。此为可在本地修复的工程问题，不改变研究假设、指标、图库或用户资源政策，无需用户做新的科研决策。

## 已通过的独立核验

- Root在Kimi产生选择配置之前，从已冻结token审计独立预计算100文本。与新Git配置逐索引/分片/实际token/全局层核对完全一致；50+50，每片probe层计数17/16/17，原全层计数shard0=605/613/604、shard1=610/602/610。
- 四worker源指纹/token审计/G2配置六项SHA均匹配，配置LF-only；配置SHA `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`。总804600对、402300/卡，每query503批（502×16+14），全局pair索引及原8046图库不改。
- Root独立执行新probe测试 **36 passed/5 subtests，9.72秒**。Kimi225/33子测试输出亦读取核对，包含此前修正的测试失败记录，不仅采信报告。
- Root比较7个核心函数/类的AST：assemble_batch、forward_and_score_batch、score_pair_token_diagnostic、load_gated_model_and_store、SharedBudgetLedger、authorize_text2cad_run、ZipCandidateStore与Base一致；无需重新消耗G1真实前向。新循环计时在chunk实际原子发布之后结束，包含评分/序列化/写盘；公式为每卡每层最慢query×全层数+冷启动、两卡max×1.30。

## 必须修复的三项

1. **实际chunk内容未核验。** `text2cad_g3_probe.py:1157` 的CPU CLI仅查exists；测试fixture自己创建空chunk。独立reviewer复现100个空文件、总0字节仍exit0/gate=true。ETA必须解析每个完整chunk并核原密集pair/全局索引、8046有效scored行、有限分数和全绑定，再与timing的实际计数/有效项总数联核。
2. **账本/统计/时序仅信成功字段。** `text2cad_g3_probe.py:718` 与`:918` 未核实际charged、planned/forwarded/batch统计、预算上限和卡时，也未把timing.invocation/window绑定到对应成功账本。Root及reviewer均复现：统计/计划/charged全0、卡时99999而cap1、timing ID不在账本，仍passed；原合法fixture的ledger epoch1–3与timing epoch1000之后也被接受。须核每卡402300对/25150批和总804600、准确预算、真实成功贡献invocation及时间范围。第一完成卡的summary可有正常peer pending，宜结合最终账本，不误拒绝真实并发。
3. **彼此一致代替固定资产验证。** `text2cad_g3_probe.py:895`/`:976` 取summary自称manifest并只比较两卡一致。独立CLI复现两summary/timing manifest全0、BERT全0、最大CAD长度128仍passed。必须绑定已经审定且运行时核验的worker/checkpoint/BERT五文件/score/full272及实际load/preload证据。

这些是后处理准入缺陷；未发现本次更改破坏模型原生评分数学。不把合成拒绝路径见证解释成模型不理解文本。

## 同轮补齐的断点问题

reviewer将其列为Minor：严格空目录首跑不会触发，但已承诺的resume不完整。`text2cad_frozen_runner.py:1629`仅在resume读sidecar，`:994`仅校验“已有chunk必须有timing”；孤立timing没有chunk时该query仍进todo，可能重扣固定预算。fresh-run遇已有sidecar应拒绝；resume须核有效本片probe timing与已验证完整chunks一一对应、无孤立/外来/重行/撕裂。不用虚构时间补齐，也不重新评分这类残留。

## 证据和下一步

Root CPU目录：checkout外 `CADRetrievalIdea1_runtime/g3_probe_cpu_20261002/`：独立盲选JSON、新测试输出、`CODEX_G3_MATH_PATH_20261002.json`、`CODEX_G3_ETA_REFUSAL_WITNESS_20261002.json`。独立审查者 `g3_readonly_review_759691b9_5efpksgo/READONLY_REVIEW_WITNESS.json` 含empty_chunks/wrong_asset_binding/contradictory_ledger/orphan_timing四组见证。未批准NPU，未生成真实G3 score，不需回滚服务器。

R1只修验证与resume、补必要CPU拒绝/producer→reader测试。原Kimi报告保留，新建R1报告；CPU/源码通过重审后才更新完整approved SHA。G2真实38.28394小时估计不变；G3/全量尚未运行。此轮未做新的文献/语义/SOTA裁决。
