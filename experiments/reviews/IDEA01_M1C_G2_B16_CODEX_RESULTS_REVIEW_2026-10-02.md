# Text2CAD G2 batch16：Codex结果审核

2026-10-02。原始run `idea01_text2cad_g2_b16_20261002`；运行完整SHA `08a0183c86afc7198eed1c798d289d4b6f0c4a42`，授权文档SHA `fe1b2d4fb6de5e0ba01dffe916bc400b6dae6f8d`。执行者报告 `TEXT2CAD_G2_B16_RESULTS_2026-10-02.md` 原文保留，首次提交 `42a5105a9f6d33f068774d33d090cb63ed78f37e`，回填 `3aa3d9d9bdb29b7fa7ab0bcf953d48df1bd7fe05`。

## 裁决

**G2通过，可进入G3本地实现与CPU审查；本文件不批准新的NPU前向或正式全量。** 固定batch16、FP32、完整272行、无trim/无文本缓存。G1 batch32的6个logit元素失败保留，不因G2成功改判。

Codex从原始文件独立核查26,254项，全部通过；审核脚本未导入生产评分器或ETA实现，零模型前向。检查范围：实际输入/代码/授权绑定、两卡真实分片、批尺寸16及尾14、33 cell×3 rep/卡、12,024条计时score的有限性与有效字段计数/和/均值关系、共享账本、99组真实工作窗口及全量分层权重和ETA公式。没有G2 raw logits，未声称本次重新计算了logits的log_softmax；评分数学以此前G1原始logits审核为依据。

| 证据 | 结果 |
| --- | --- |
| 两卡实际退出/账本 | exit 0，均finalized；8,100对/卡，共16,200，零额外重试 |
| 累计卡时 | 168.71卡秒，低于7,200卡秒上限 |
| 真实窗口重叠 | 99/99通过80%门；最低0.992359358095925 |
| 峰值HBM | npu:0 1,893,842,944 bytes；npu:1 1,893,712,896 bytes |
| 全量实际尺寸 | 3,644唯一文本×8,046 CAD =29,319,624对 |
| 卡0全量计算/冷启动 | 96,891.37569162523秒 /33.73913043085486秒 |
| 卡1全量计算/冷启动 | 105,983.62824119325秒 /33.423819813877344秒 |
| 保守墙钟 | max(两卡计算+各自冷启动)×1.30 =137,822.16767930926秒 =38.283935466474794小时 |

原始ETA SHA256 `d354d5ad1bfce254ada0acf81e0ebcc84b773e46c2a69a61bc3b9f417312f76d`，独立重算逐项一致，低于604,800秒。采用各cell三rep中最慢每批E2E及实际分层生产批数；未用单卡吞吐理想乘二。短rep的写盘行为不能完全代表每条文本输出整个图库，G3须实测这部分长路径；38.3小时尚非全量实际完成时间。

## 归档与证据位置

本地 `experiments/runs/idea01_text2cad_g2_b16_20261002/`、云脑runtime同名run以及共享OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g2_b16_20261002/`。Codex独立核对原始SHA256SUMS的415项本地哈希，并在cvteam14执行无管道掩盖的 `LC_ALL=C sha256sum -c --quiet SHA256SUMS`，exit 0。原始manifest SHA256 `dcd3d914c6aedd583c14896934eeff75612c80058f88f3611dd4624ee26e5529`。归档仅用cvteam14存储，无GPU占用。

独立脚本与JSON另放本地/OBS的 `codex_check/`，不改原始415项manifest：`text2cad_g2_b16_codex_audit_20261002.py`、`CODEX_G2_B16_CHECK_20261002.json`，新manifest SHA256 `47bbe29636545ab90b8abcaffffc0c6dff2747c24d7f1635f127e5f9951de9e0`，两项OBS严格SHA校验exit 0。CPU同步前后见证另在 `m1c_g2_b16_cpu_20261002/`，8项已独立核验。

## 对执行者报告的限定

1. generic `environment.shard_id=0/n_shards=1` 两卡相同，为测速入口未使用的matrix配置默认值。真实分片以sync、device→ledger、cell/pair清单为准；本次已独立核查，而非仅由跨卡文件不相同推断。
2. 报告“失败与异常：无”适用于实际评分。归档计数的中文locale误判及读取/断言后处理错误曾出现并修正，未导致评分重跑或源码变化；不应把整个执行过程表述为没有任何操作错误。
3. 重复rep文件一致只证明本次设备/输入的重复性，不能保证所有输入或更大batch；epoch窗口不是独立的NPU内核计时。七天估计仍须G3真实全图库路径和较慢卡验证。
4. **没有检索质量/R@K结论。** 本次仅证明batch16两卡可同时稳定测速，并给出可推进持续运行验证的时间估计。

下一步为 `IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md`：固定盲选100文本，仍遍历完整图库，先Kimi实现/Codex审码及CPU核验，随后独立阶段授权。无新问题需要用户决策。
