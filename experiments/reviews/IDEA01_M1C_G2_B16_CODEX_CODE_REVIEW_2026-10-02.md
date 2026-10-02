# Text2CAD batch16双卡测速：Codex启动前审核

2026-10-02。实现SHA `4a22261399e84eeed6656ea3a17f8e2a6b8b84d6`；Kimi报告回填HEAD `08a0183c86afc7198eed1c798d289d4b6f0c4a42`。

## 裁决

**允许按本轮B16计划启动两卡throughput，不允许其它batch档、trim/cache、G1重试或正式矩阵。** 用户已同意B16路线。唯一源码差异为SyncBarrier默认poll从0.2降至0.001秒；模型、评分、shift/mask、FP32、选择配置、预算/ETA数学与已审R3一致，timeout/peer失败/绑定/真实窗口80%门仍保留。

Codex独立复核：

- 原CPU双进程见证：到达差0.03秒、工作窗口0.10秒，重叠0%。修复后使用相同脚本、仅改新输出名，独立测得重叠 `0.9999119559113278`；未混入同步等待或延长工作窗口。Kimi自己的after见证为0.994752，同样通过。
- 受影响 `SyncBarrierTest` 与 `RepWindowOverlapTest` 独立执行 **13 passed，6.42秒**。Kimi现有两文件完整CPU套件为110 passed/25 subtests；从原工具返回核对，非仅引用报告。
- 原固定配置选择[16]，按每cell实际样本批独立计算 `2*前两批pair数 + 3*全部样本批pair数`，每卡8,100、总16,200；每卡33 cell，批尺寸只有16及真实尾批14。此前固定清单1187项核验不重做，配置及token审计指纹未变。
- 两卡公共计划SHA `e88343f660ab955dccb1844befd7029362cedd7d89b0dfc49b431d8fc831a603`；shard0本地SHA `fed82ddfc9f898cdcddbc81cbac9246ee93a9c0808fade1f69301258b40fa661`；shard1 `75302a04e0854733a20a594379387a70eccf422e1ddb646d5dccac813e5c3dc6`。
- 环境spec规范JSON SHA前缀仍10c18c17，复用既有已验证环境；无安装/升级/新环境见证前向。SSH只读检查两卡健康且无运行进程；runtime所在磁盘约4TB可用，run目录尚不存在。实际启动前Kimi须再核一次。

## 对Kimi报告的限定

报告§5关于Windows sleep最坏粒度与Linux“预期不差”的推断不能当准入证据。其中 `0.03+0.016=0.046` 秒并不满足0.10秒窗口的80%门；到达差本身也不等于barrier返回后的窗口起点差。只采用实际CPU见证和随后真实NPU窗口，不推断跨OS保证。不覆盖执行者原文。

测速的generic `environment.shard_id/n_shards` 仍来自未使用的matrix RunnerConfig默认0/1；真实throughput分片由device→ledger shard、sync.shard_id、cell及pair清单证明。结果审核不能把generic默认字段当真实分片，也不能把实际两卡独立分片解释成单卡运行。启动时显式 `--batch-size 16` 消除generic batch_size默认8的歧义；实际分批仍由 `--batch-tiers 16` 决定。

## 启动及验收

运行完整SHA定为 `08a0183c86afc7198eed1c798d289d4b6f0c4a42`，仅在ACTIVE_TASK写入pilot_authorized并push后有效。16,200对/2卡小时、两个进程各一次、每进程cards_used=1、timeout3600；新per-SHA共享账本，不沿用G1余额。

两卡原始rep/score文件、峰值HBM、真实窗口、账本/退出码、运行前后干净Git、SHA归档后，Codex独立按完整矩阵权重和较慢卡重算ETA。每点80%并发门和七天604800秒门不改变。G2通过后才自动推进G3稳态复核；本次不批准G3或全量。

CPU证据：checkout外 `CADRetrievalIdea1_runtime/g2_b16_cpu_20261002/`，含before/after/codex三个见证及 `codex_affected_tests.txt`。原G1 batch32失败保留，不重新命名为通过。
