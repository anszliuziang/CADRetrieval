# Idea 01 两卡执行追踪表

本表是 [自动执行路线](IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md)的轻量索引；实际授权只看 `experiments/ACTIVE_TASK.md`。状态：计划待讨论，尚未启动任何新 NPU 运行。

| ID | 阶段 | 对象 | 最小交付 | 当前状态 | 下一门 |
|---|---|---|---|---|---|
| T00 | M0 | Text2CAD v1.0 | query/gallery/qrels、UID/重复/资产审计 | 已审，见 `IDEA01_TEXT2CAD_M0_CODEX_REVIEW_2026-09-28.md` | G0 |
| T01 | M1a | CADmium CPU scorer | 模板、shift、EOS、4096 边界、测试 | 已审 CPU；未授权模型前向 | G0 |
| T02 | G0 | CADmium 真实 runner | Kimi `IMPLEMENTATION.md`、本地 SHA、长度/资产清单、Codex 审码 | 待实现 | G1 |
| T03 | G1 | CADmium 1+10 对 | 逐 token 核验、HBM/吞吐、SHA 与预算台账 | 待授权 | G2 |
| T04 | G2 | CADmium 固定小池 | 100×100、长度分位吞吐、空查询和基线 | 待授权 | G3 |
| T05 | G3 | CADmium 真全图库子集 | 100 query×8,046 CAD、覆盖和 cluster CI | 待估算/授权 | G4 |
| T06 | G4 | CADmium A 正式全量 | 去重穷举 3,644×8,046，并恢复全部 7,934 条 query；短名单仅可另报 | 全量必做；G2/G3 保守 ETA≤7 天才启动，否则讨论 | G5 |
| T07 | G1–G4 | CadQuery 干净子集 C | 训练交叉、覆盖、同图库分数及诊断 | 待单独激活 | G5 |
| T08 | G1–G4 | 原版 Text2CAD 向量 | 多头目标和 Ascend 正确性，A 诊断结果 | 待单独激活 | G5 |
| T09 | 可行性 | HoLa/STEP | checkpoint、表示覆盖、预处理与成本门，可能只交阻塞报告 | 待单独激活 | G5 |
| T10 | G5 | 跨臂结果审计 | 原始分数、失败、基线、分层与 Codex 独立复算 | 待前面各门 | 结论 |

每一行的执行 SHA、run ID、实际耗时、产物 SHA、审查结论在该阶段 `submissions/` 与 `reviews/` 的独立文档中记录；本表只跟踪进度，不承载授权参数。
