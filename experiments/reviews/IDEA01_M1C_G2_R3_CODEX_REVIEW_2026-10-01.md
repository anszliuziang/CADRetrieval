# Text2CAD G2 R3 最终代码审核

- 实现SHA：`03db50f7e49f4713ce0b96ce0d48d4e57ad39f67`；报告回填HEAD：`598dd40b0cf6f8ce87003225c0ba3ee434abb450`。
- **通过本地代码审核，可进行零前向CPU资产预处理。NPU与704对门尚未授权。** R1/R2承重缺口已逐轮修复，原始提交和报告保留。
- 独立代码审查代理确认两项实际接线：HBM reset进入peer异常通知范围；正式eta-g2先核各自唯一invocation、同host/共享budget_file、对应shard、单卡、finalized和int0退出，再估算。
- Codex独立重跑：`D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_bench_protocol.py experiments/tests/test_text2cad_speed_bench.py -q`，**108 passed、25 subtests passed，26.57秒，exit0**；日志在checkout外 `CADRetrievalIdea1_runtime/g2_protocol_r3_cpu_20261001/codex_independent_r3.txt`。`git diff 59ffc944..598dd40b --check` 通过。
- R2受影响runner测试已独立通过（154/18），R3未改runner、模型、score、容差或旧配置，不再扩大重复测试。

## 下一门的权限

1. push本地已审实现与审核文档，服务器检查干净，Git fetch/detach至上列598dd40b完整SHA。
2. 仅运行CPU `audit-text-lengths`，从固定BERT tokenizer生成checkout外派生资产，不构造BertModel，不占NPU；取回并核SHA。
3. 仅本地以干净worker清单和真实审计文件生成G2配置，审查完整矩阵权重、样本、分片和64对；再提交Git。
4. 配置和完整运行SHA审定后，另在ACTIVE_TASK授权单卡704对/1卡小时的G1 profile。当前production仍reference，G2与完整矩阵均未授权。

完整矩阵29,319,624对及两卡七天门不改变。实际NPU并发、数值和HBM行为仍须后续真实运行验证，CPU通过不代表真实吞吐已验证。
