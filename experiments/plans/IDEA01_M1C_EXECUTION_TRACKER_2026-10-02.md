# 原版Text2CAD两卡执行追踪（2026-10-02）

当前科学总协议仍为IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md；实际运行授权只看ACTIVE_TASK。本表替代旧路线中已失效的CADmium执行顺序，但不改旧历史文档。

| 阶段 | run / 输出 | 当前证据 | 状态 / 下一步 |
| --- | --- | --- | --- |
| M0 | 固定v1.0 worker四文件、UID map物理分离 | 8046 CAD/3644唯一文本/7934原查询，固定SHA | 已审；评分不读UID/qrels |
| G1 | idea01_text2cad_g1_profile_20261001 | b16通过，b32六logit元素失败保留 | 已结束；不重试b32 |
| G2 | idea01_text2cad_g2_b16_20261002 | 双卡16200对，26254独立检查/415 OBS SHA，38.28394小时ETA | 通过 |
| G3 | idea01_text2cad_g3_probe_20261002 | 双卡804600对；3219147独立检查/130本地及OBS SHA；28.85179小时ETA | 通过；Kimi报告de02bb4a/2b006501已核 |
| G4 | idea01_text2cad_g4_full_20261002 | 计划完整29319624对；保守取38.28394小时，硬七天双卡墙钟 | 原实现52+6 CPU及七核心AST通过；两处生命周期Important拒绝，激活最小R1修复/GPU0 |
| G5 | 完整原始矩阵审核→独立CPU评价 | 恢复7934查询，UID R@K/MedR/重复上限/基线与CI | 待G4；不看test改方法 |

Codex结果准入/代码审查都在reviews/，Kimi新实现/运行/结果报告在submissions/；原始结果本地与共享OBS，Git不含运行产物。正常作业每30分钟只读监控，真实阻塞讨论，不扩预算或重复评分。
