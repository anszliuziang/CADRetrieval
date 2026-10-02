# Text2CAD G3：R2修复复审通过与运行批准

2026-10-02。实现 `98ba9f41c8106e87c2e925677f09fe23f0992411`，报告回填/本轮完整批准运行SHA **`487d02b9ac92f5e2338718e45b1f023c308dd332`**。相对实现提交仅报告改动。基于R1已关闭四项的核验，本次只复审R2新增两门与合法路径，没有扩大无关测试或重复真实模型前向。

## 裁决和证据

**批准fresh G3两卡验证。** Root逐行审diff，独立运行10项针对测试通过（13.42秒），另复算26项CPU见证全通过、真实模型对数0。独立只读reviewer也运行10项全通过，确认无Critical/Important/Minor、未搁置行为。原七个评分/加载/计费核心AST与base一致、固定盲选SHA不变。

- 授权card_hours×3600与本卡snapshot cap、两卡共同cap及final ledger cap相等，授权360而snapshot43200/final3001矛盾均拒绝；合法peer pending+完整final保留。
- 单卡query按生产text_index顺序串行，50条同起点、相邻重叠、逆序均拒绝；正常连续与有间隙串行通过。大时长合法结构的ETA为710606秒，正确拒绝七天门，没有通过fixture放宽1.30/80%/604800门。
- 真实producer→reader路径仍通过；所有完整chunk、实际valid counts、账本/stat/成功invocation、固定资产/load、fresh/resume隔离门保留。模型score/shift/mask/FP32/full272/b16/尾14/无cache trim、原8046图库未变。

Root外置证据 `g3_probe_cpu_r2_codex_20261002/`；reviewer证据 `g3_r2_readonly_review_98ba9f41_lga_zvnq/`；Kimi66项/19子测试与79项/3子测试原始输出也读取归档。本地与OBS证据目录 `runs/m1c_g3_cpu_review_r3_20261002/`，7份原始文件逐项OBS核验exit0；SHA256SUMS SHA `07a891e080d1a527b964b5b9c7c2efccdbd99c6dd03c120a088c79bf34f8fc5c`。

## 固定运行范围

- 上位合同 `IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md`，R1/R2修复合同继续约束证据。实际执行补充 `IDEA01_M1C_G3_EXECUTION_2026-10-02.md`。
- run `idea01_text2cad_g3_probe_20261002`：每片固定50文本，每条完整8046 CAD；402300对/片，总**804600对**，每片25150批；各卡仅一次fresh score调用，**最多12卡小时合计/6小时墙钟**，两卡独立timeout21600。不从旧G2扣账或复用余额，不另做warmup/验证前向。
- 原选择SHA `f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985`，不依据分数挑样本；不读qrels/UID映射/原CSV/旧score。
- 2026-10-02约15:40只读服务器预检确认checkout08a0183工作树干净、两卡健康且无运行进程、runtime4TB余量、prospective run及本轮ledger不存在。首个预检末行因SSH stdin换行语法失败（exit2），ledger另用独立只读命令exit0确认；不将失败预检整段标为exit0。启动前Kimi重新核验空闲/Git/目录。

此批准只让真实G3测量进入结果审查。G2保守38.28394小时保留；G3成功、完整原始结果/SHA和Codex独立复算ETA≤7天后，才自动制定批准正式全量G4。100文本不会替代3644×8046的完整矩阵。
