# Idea 01 M1c Stage B2：Codex 预审查

日期：2026-09-30。状态：**暂不接受 B2 为 G1 入口；保持 `implementation`、真实前向预算 0。** Codex 独立重跑相关五个 CPU 套件，`197 passed`，但静态审查发现以下运行准入缺口。Kimi 的 `TEXT2CAD_IMPLEMENTATION.md` 是实现者交付，不替代代码审查。

## 阻塞项

1. **pair manifest 含本机路径，云脑无法逐字节重建。** `text2cad_pair_index.py:669-670` 把 `query_path`、`gallery_path` 的字符串写进 `pair_manifest.json`，并把该文件 SHA 当成固定指纹；当前 `text2cad_pair_manifest_expected_20260930.json` 记录了 Windows `.cache\\m0_real_run_v3\\...` 路径，交付 §8 又要求云脑以不同绝对路径重建并获得相同 SHA。这在输入字节完全一致时也不可能。要求 manifest 只绑定输入内容 SHA 与稳定逻辑名称；本地用两套不同路径的同字节输入重建，断言 manifest 和全部 worker 文件哈希相同，重新生成并提交固定指纹与选择配置。
2. **前向入口允许外部预期指纹替换。** `text2cad_frozen_runner.py:1350-1353` 的 `--expected-pair-hashes` 可以指向 checkout 外任意 JSON；`text2cad_speed_bench.py` 的 `_gated_setup` 同样如此。这样修改候选/文本清单后再给相应伪指纹，生产授权门仍可放行。要求所有 forward 模式将其 resolved 路径硬绑定到已审 checkout 内的 `experiments/configs/text2cad_pair_manifest_expected_20260930.json`，或移除该覆盖参数；`pair_manifest.json` 自身 SHA 也须验证。干运行的可测试入口可单独保留，但不得成为 forward 的生产旁路。增加临时文件伪指纹拒绝测试。
3. **预加载拒绝项会导致全矩阵不完整而仍标成功。** `ZipCandidateStore.preload()` 把不合格候选记入 `rejections`，`FrozenMatrixScorer.score_chunk()` 对其所有 pair 写 `rejected`，矩阵 CLI 仍可汇报 `status: ok`。本轮固定 test 已独立证明 8,046/8,046 可评分，主合同要求精确全矩阵；任一候选结构拒绝必须在首次 forward 前阻断矩阵和 G1/G2 测速入口，保留失败报告。可保留诊断性拒绝清单，但不能将非全覆盖运行称为完整评分。增加合成坏候选拒绝测试。

## 须在本轮修正的边界

4. **两卡卡时按进程占卡数计。** 每个 `npu:0`/`npu:1` 进程实际只占 1 张卡，但交付 §8 的两条并行命令各写 `--cards-used 2`，共享账将同一墙钟扣成 4 卡时。`cadmium_runner.authorize_run` 的字段语义也是单进程占卡数。要求矩阵和测速 CLI 的当前单设备模式固定 `cards_used=1`，文档中的两卡并行命令各写 1；测试总计两卡同时运行一小时扣 2 卡时。
5. **优化仍须有实际准入门。** 矩阵 CLI 现允许直接传 `--trim-mode batch_max`/`--text-cache`，即使真实 G1 `trim-check`/`cache-check` 从未通过；`bfloat16` 亦可直接选用。当前 B2 可以只批准默认 `trim=none`、`cache=off`、`float32` 参考路径，生产矩阵入口先拒绝其他组合；真实等价/精度结果经 Codex 审核后另加绑定获批 SHA、资产、设备和 dtype 的授权字段或证据文件，再重审启用。测速入口可保留明确的 G1 对照模式。
6. **CAD ZIP 被重复解析 8,046 次，测速前置开销不可接受。** `ZipCandidateStore.preload()` 逐候选调用 `_load_one()`，后者每次重新 `zipfile.ZipFile(self.zip_path)`，而固定 ZIP 有 528,161 个目录项。Codex 在云脑以 pinned ZIP 做零前向计时：单次打开/解析目录约 3.93 秒；仅此操作乘 8,046 就约 8.8 小时，且每次测速模式都会重来。要求 `preload()` 一次打开 ZIP、循环 `archive.read(member)`，保留逐候选验证和失败报告；增加测试断言一次 preload 的 archive 开启次数为 1。此项不改变评分数学。

## 已通过的证据与保留项

- Codex `D:\miniconda\python.exe -m pytest experiments/tests/test_text2cad_pair_index.py experiments/tests/test_text2cad_frozen_runner.py experiments/tests/test_text2cad_speed_bench.py experiments/tests/test_text2cad_model_adapter.py experiments/tests/test_text2cad_vector_scorer.py -q`：`197 passed in 38.59s`，exit 0。尚未做真实 forward。
- B1 全长默认路径新增可选长度参数，静态 diff 显示 `None` 仍取 271；CPU 合成等价测试通过。真实 NPU 等价仍是 G1 门。
- `pair_manifest.json` 自身 SHA 的加载器路径已在 Kimi 最终文件中正确合并进 expected hash 映射；上述第 2 项针对的是可替换预期指纹文件，不能因为自哈希校验存在就放行。
- 矩阵 CLI 是整文本分片运行，单对/十对 pilot 必须走独立测速 CLI；当前结构满足这个分工，后续须继续核预算扣账与数据隔离。

修复后 Codex 会独立重跑相关测试、跨路径重建、零前向 dry-run，并审核最终 diff。未在 `ACTIVE_TASK.md` 写完整批准 SHA、卡时和 pair 上限之前，不允许任何真实模型 CPU/NPU 前向。
