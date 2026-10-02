# Text2CAD G3 R2：卡时绑定与同卡串行时间轴

2026-10-02。从R1报告HEAD `da476b235c6310140fd2c9afa6622bd5be02f146` 开始。上位G3计划与R1合同继续有效，新的第二次审查见 `IDEA01_M1C_G3_CODEX_CODE_REVIEW_R2_2026-10-02.md`。本轮真实模型/BERT/NPU预算0，只本地CPU。

## 只修以下两项

1. ETA要求卡时授权与实际账本cap绑定：每卡 `authorization.card_hours*3600` 等于该卡snapshot `max_card_seconds`，两卡授权一致，最终账本 `max_card_seconds` 等于两snapshot。保留最多12卡小时的上限；后续实际授权将固定12/43200。必要时只扩验证器返回字段/传参，**不改SharedBudgetLedger算法**。合法第一完成卡peer pending继续通过。
2. 在真实timing reader按生产顺序验证单卡query时间轴：生产todo按全局text_index升序，每条下一query的start必须不早于前一条query落盘end，拒绝逆序/重叠/同起点等矛盾。保留每条E2E及invocation边界、跨卡80%门、每层最慢query/冷启动/1.30/七天门。不要用排序窗口或扩大invocation范围掩盖非法输入。合成合法fixture应形成可串行的真实时间轴；七天门两侧测试的假时长变大时，同步重建合法spacing/账本区间，而非放宽新检查。

保留既有chunk/live资产/load/resume及producer→reader检查。不新增真实forward、预热、NPU测试、下载、依赖安装或SSH。score、shift/mask、full272/FP32/b16/尾14/无cache trim、原8046图库/3644文本和固定选择SHA `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985` 均不变。

## 必要CPU证据与交付

- 合法fresh/peer pending+最终账本、真实producer→reader路径通过。
- 授权360秒而snapshot43200/实际3000必须拒绝；final cap3001而snapshot43200必须拒绝；两卡授权不一致必须拒绝。
- 同卡50条start全相同、相邻重叠、逆序必须拒绝。正常连续/有间隙串行窗口通过，双卡真实并发门仍有效。
- 只跑probe的受影响测试与必要不变项复核，无关全套和G1/G2真实重跑均不需要。CPU输出/源差异/完整实现SHA放新的 `TEXT2CAD_G3_R2_IMPLEMENTATION_2026-10-02.md`，旧报告不覆盖。
- 新报告更正账本实际路径：`$RT/budgets/text2cad_m1c_<approved_sha>.json`；最终ledger取回后随run原始证据归档，CPU ETA仍必须 `--final-ledger`。准确命令以新报告为准。

Kimi只本地修改必要probe/测试文件及自己的新submission，exact stage实现commit、报告commit、SHA回填后停止review_required，不push、不改Codex文档。不做其它重构。Codex只复审新增两门及合法路径，通过后另写G3 pilot_authorized/完整approved SHA/804600对/最多12卡小时/6小时墙钟。
