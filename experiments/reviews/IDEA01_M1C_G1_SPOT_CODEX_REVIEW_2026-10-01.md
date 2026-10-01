# Text2CAD G1 spot：Codex 结果审查

判定：通过单对可运行性与分数实施验收。批准代码 `24570dd892202be4b5e31a4bdc12cb23dc258d57`，实际同 SHA；NPU0 只执行 1 对。不能据此评价检索精度或外推七天时长。Kimi 原始报告保留于 submission。

Codex 从云脑独立取回 bench JSON（SHA `c5e08075c834273562b0b4ed5326f356f0f59d7ceadf5317813c1609e687e07c`），直接以 `math.fsum` 复算逐项 log probability：x 33 项合计 `-19.92079045623541`，y 8 项合计 `-23.18826524168253`，总分 `-43.10905569791794`。计数、总分、terminal x 的 START 标签和位置、terminal y/padding 排除、有限性、代码绑定及预算 11 项检查全部通过；检查 JSON 留在本地 run 的 `codex_check/`。logits `[271,2,267]` 全有限。ledger 扣1对、33.64卡秒，加载冷启动32.64秒。

共享 OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g1_spot_20261001/` 的 7 项 SHA 清单由 Codex 另行通过 `ssh cvteam14` 验证，全部成功；原始结果、命令、预算和复核脚本齐备。Kimi CLI 正常退出0，无 API错误。未发现源码修改或计划外前向。

两点限定：Kimi 报告写的 UTC 时段有20秒偏移，实际 JSON 的窗口为 **2026-10-01 04:03:27.316739–04:04:01.508327 UTC**（北京时间12:03:27–12:04:01，约34.19秒）。其 y 头均值较低的“高熵解释”只是推测，本轮只接受数值事实。其下一步建议直接 G2 不被采纳：先完成既定 G1 十对批/单项一致性与可选优化等价门。

下一门合同为 `plans/IDEA01_M1C_G1_BATCH_2026-10-01.md`。它使用完整 SHA `f0d348568f60ac2956deeb058aca7f4873afad79`：相对已审24570dd仅文档/授权改变，源码、脚本、配置、依赖和LF属性无差异，由 Codex 核对；为阶段预算使用独立 ledger，不修改或删除此前 spot 的账目。下一次批准最多90对、1卡小时，仅 npu:0，不批准 G2 或检索评分。
