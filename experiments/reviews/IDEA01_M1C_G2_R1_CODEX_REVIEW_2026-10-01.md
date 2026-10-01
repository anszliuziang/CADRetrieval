# Text2CAD G2 R1 实现审核（2026-10-01）

- 审核对象：实现 `fca6e977834f2f022ae1740354fbfe14a7bbba8d`，报告回填 `4fc2825e4126e5d4757332a07d2de0c4c99e1b84`；起点 `32b83f76068b53b1bed12026acd36f58fa96f9cf`。
- 合同：`experiments/plans/IDEA01_M1C_G2_PROTOCOL_REPAIR_2026-10-01.md`。
- 结论：**需要修复，尚不批准部署、真实前向或 G2。** 生产相邻批、固定文本分片和慢卡 ETA 的主体方向正确；以下检查缺口可使测速失败或不充分的证据进入 ETA。全部核查为只读/CPU，没有运行真实模型。
- 审核方法：Codex 读源码及提交报告；独立代码审查代理预审 Task1–4；Codex 已只读核对当前 torch_npu 的 `max_memory_allocated(device=None)` 与 `reset_peak_memory_stats(device=None)` 签名。行号以 R1 提交为准。

## 必须修复

1. **双卡 barrier 使用不同绑定。** `text2cad_speed_bench.py` throughput 块分别计算各 shard 的 `tier_plans_sha256(plans)`，却把它放进要求双方相同的 binding；两卡文本和 cell 不同，第一次同步会拒绝。共同 binding 应使用固定 shard 顺序的完整双卡计划哈希，各自 local plan SHA 单独保存、核验。同步点也应使用双方共同的 tier/text-tier/CAD-layer/rep 编号。
2. **显存统计没有指定设备。** `peak_device_memory_bytes` / `reset_peak_device_memory` 没有向 NPU API 传入 device，且此路径没有设置当前卡；npu:1 可能测到默认卡。显式传设备；真实 NPU 采集失败不得把 None 当有效 HBM 证据。
3. **ETA 未完整验证两卡。** `estimate_sharded_eta` 只检查 env0 的 float32/full-length/uncached。必须分别核卡身份、所有 reference 配置、代码/模型/评分/清单/选择/计划绑定，拒绝重复或缺失的 tier/cell。
4. **ETA 可接受不完整或错误 profile。** 当前只要求 reps 非空，未要求至少三次且编号唯一；窗口核验只遍历卡0，卡1重复/多余点可被 dict 覆盖或忽略；未从 selection 核每卡 local plan SHA、逐批真实大小与 pair 顺序。重建审定计划，要求双卡点集合完全相同、完整 reps 和批明细一致。
5. **selection 输入隔离回归。** `_diff_path` 忽略额外键，`verify_g2_selection` 允许额外顶层项；整个 selection 被 worker 读入。必须严格限定顶层、嵌套及 provenance 字段，拒绝塞入 qrels/evaluation_map 等额外输入。
6. **token audit 未绑定真实文件。** runtime 仅检查 token audit SHA 长度并用 selection 自带 token 表重算自身。增加执行时实际审计文件的 SHA、schema、texts/BERT provenance 与内容对照；选择构造、G1 profile、G2、ETA 使用同一已核资产。
7. **只有首 cell 暖身。** 合同要求每档/每cell 至少两次 warmup，但 `run_production_batch_tiers` 与 `plan_production_forwards` 只枚举首 cell 的前两批。对每 cell 明确枚举两次暖身并同步调整预算、测试和命令中的对数。原报告“每cell”与括号中的“首个cell”矛盾。
8. **非有限失败缺原始证据/账本退出状态。** 详细评分在保存 variant logits 前可因非有限值抛错，导致失败张量没有保留。保留带 variant、pair 和批编号的失败 logits 再报错；普通 gate_failed 返回3时，ledger 不应先记录 exit_code=0。通过和失败路径都核实账本最终状态。
9. **正式 throughput 可以完全省略 barrier。** CLI 允许同时不提供 sync-dir/group，ETA 也不要求 sync.enabled。正式两卡 G2 强制同时提供两个参数，ETA 验证两卡使用同一 group、共同 profile。共同 step key 本身已正确归一，不需要改为新的命名。

## 同轮补齐的交付内容

- profile 逐对报告补 `n_valid_x/y`、有效序列/真实 token 数及逐头/总分差；汇总差取三项，避免两头变化抵消。原门本身已核三分数，不能据此声称原门绕过。
- 原 R1 报告留存，新写 R2 修复报告。统一运行变量与完整已核资产路径，去掉 `...`。**tokenizer 审计可以在远端生成派生资产；selection 是配置，必须在本地生成、提交后 Git 同步。** 不在服务器生成新配置再回传入库。
- 只运行受影响协议/速度/runner 的 CPU 测试；缺 pyarrow/transformers 的历史全套测试缺口不需要此轮安装依赖或反复全套运行。
- Codex 独立重跑当前协议/速度测试：`D:/miniconda/python.exe -m pytest experiments/tests/test_text2cad_bench_protocol.py experiments/tests/test_text2cad_speed_bench.py -q`，82 passed、6 subtests passed，13.05秒；`git diff 32b83f7..HEAD --check` 通过。这些已有测试通过不消除上列未覆盖的缺口。

## 下一步

Claude Code/Kimi 在本地修复、CPU回归并提交，不 push、不部署、不运行真实前向。Codex 重审通过后再推进 tokenizer 资产核验、固定64对的704对 G1，以及真实双卡 G2。完整29,319,624对与七天门保持不变。
