# Text2CAD G2 R2 审核

- 实现：`55e30d156c6c4c793d1ba5aea146dd19b8c7b9da`；报告回填：`59ffc9444a77d7c6f6a7e55ec08095783c92fbe9`。
- 独立审码已核：共同/各分片计划绑定，两卡 reference 配置、完整批/rep和窗口，严格selection、实际token audit、每cell暖身及同源预算、失败原始logits、gate失败账本与逐头报告。上述R1问题已修。
- Kimi受影响三文件 CPU 测试154 passed、18 subtests passed；Codex独立重跑同三个文件：154 passed、18 subtests passed，75.21秒，exit0。日志在checkout外 `CADRetrievalIdea1_runtime/g2_protocol_r2_cpu_20261001/codex_independent_r2.txt`。没有任何真实前向或新NPU评分。
- 新G2六档预算：每卡480/960/1980/4020/8100/15780，合计31320；两卡62640。G1 profile仍704。

## 最后两项修复

1. `text2cad_speed_bench.py:1281` 的 reset_peak_device_memory 仍在try外。NPU统计异常应进入现有peer signal失败处理，不让另一卡只等180秒超时。最小修复移入try并用CPU fake sync测试原异常保留且FAILED通知发生。
2. 正式ETA入口应复用已有budget证据，不增加重复身份字段：每份result按invocation_id找到唯一记录，核非空host相同、budget_file相同、shard_id对应npu设备、cards_used=1、finalized=true、exit_code=0。当前结果已包含全部字段，尚未自动检查。此项是运行证据检查，不改变数学估计或额外增加forward。

源代码/配置仍只在本地修改，先交付短修复及针对性CPU测试，审过再push和CPU资产核验；本review不授予NPU额度。R1/R2原报告保留。
