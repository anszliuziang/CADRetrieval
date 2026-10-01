# Text2CAD M1c B2：Codex 复审与 G1 交接

日期：2026-10-01（北京时间）。本报告替代 2026-09-30 B2 预审的运行判定。Kimi 的 `TEXT2CAD_IMPLEMENTATION.md` 保留为原始交付记录；其中旧 manifest SHA、每进程 `--cards-used 2` 和共享 shard 输出目录不得用作当前运行命令。

## 判定

当前实现可以进入 **G1 单对 spot**，前提是本轮 CPU 测试全部通过、提交并推送完整代码、远端重建清单与零前向审计通过，随后由 Codex 在 `ACTIVE_TASK.md` 单独批准完整代码 SHA、1 个评分对和 0.5 卡小时。此前所有授权均不适用。G2、十对对照与全量评分尚未获批。

独立复审 `/root/b2_code_review` 检查了当前工作树：此前 8 项问题已修复，未发现阻塞 G1 单对的新 Critical/Important 问题。它只做静态只读审查，未执行真实模型。Codex 核对了实际调用链和修复；外部评审不替代以下验证证据。

## 本轮修复与证据

- 清单仅记录输入内容 SHA，跨主机路径不影响内容。远端审计进一步发现 Windows 默认 CRLF 与 Linux LF 的字节差异，已在 writer 明定 LF，并以 `.gitattributes` 固定仓库 JSON 配置的 LF。真实完整本地 v4 重建与远端 v3 构建的全部 worker SHA 完全相同；当前完整清单 SHA 为 `62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557`。G1/G2 选择配置重新生成，当前 SHA `dd07ec91782fa8156fa490f176d0fe0e3fa118488e687ce7bf14a04af29d834a`。旧 CRLF 指纹已作废。
- `evaluation_map.json` 的 SHA 独立登记在 expected 配置的 evaluation 字段，评分 manifest 和 loader 不再打开它。测试删除映射文件后仍可完成 worker 加载；评分器拒绝将映射加入评分 artifacts。
- forward 入口固定仓库内已审指纹、原版 checkpoint 与非 BERT 172 键清单；默认 FP32、全长、无文本缓存。矩阵入口拒绝未批准的 BF16、裁剪和缓存；候选任一预加载拒绝即在前向前停止。ZIP 在一次 preload 中只开启一次。
- 每个单设备进程计 1 张卡。bench 与 matrix 在同一张卡使用同一 ledger identity，两张不同卡可以并发，同卡双运行被拒绝。card ledger 在模型迁移/加载前登记，失败时也 finalize。
- stale heartbeat 不能证明进程死亡；仅 Linux 同主机 PID 确认消失后自动回收，计卡时间至恢复时刻。Windows 保守保留，不调用 `os.kill(pid,0)`；该 API 的平台差异由 [Python 官方文档](https://docs.python.org/3/library/os.html#os.kill) 核验并由专项测试覆盖。
- exact chunk 恢复须有完整 dense 索引、有限分数、有效计数和与 header 一致的评分对数；rejected/null/NaN 行不能被跳过。矩阵两进程分别写 `runs/<run_id>/shard0` 和 `shard1`，每进程 `--cards-used 1`。
- ETA 拒绝同文件、同 invocation、同 NPU 的重复测速结果，绑定共同代码 SHA、checkpoint、清单、选择、评分定义及成功 batch 档。`--batch-tiers 16,32` 不能绕过 large-batch 门。

实现主体由 Kimi 交付。预审与独立审查后的上述修复由 Codex 在本地完成；此前 Kimi 修复调用长时间无进展后已停止，未修改原始实现报告。所有修改仍须通过 Git 进入服务器。

## 验证记录

完整 CPU 测试命令：

```powershell
D:\miniconda\python.exe -m pytest -q experiments/tests/test_text2cad_pair_index.py experiments/tests/test_text2cad_frozen_runner.py experiments/tests/test_text2cad_speed_bench.py experiments/tests/test_text2cad_model_adapter.py experiments/tests/test_text2cad_vector_scorer.py
```

最终重跑 **209 passed，7 subtests passed，112.24 秒，exit 0**。日志：本地 checkout 外 `D:\projects\IdeaSpark\CADRetrievalIdea1_runtime\b2_cpu_tests_20261001.log`。首次重跑的 209 项通过，另 1 项失败来自测试 fixture 未提供新增 device 参数，已修正 fixture；这不构成真实前向验证。

上述全套后发现并修复 LF 问题；专项测试先在 Windows 复现 CRLF 失败，修复后通过。受影响的 pair-index 与 speed-bench 套件重跑 **58 passed，4 subtests passed，10.86 秒，exit 0**，日志 `b2_lf_tests_20261001.log`。独立复审确认全部 writer、Git LF 属性与新指纹同步，未见新增 G1 阻塞。失败的云脑零前向审计证据保存在 `runtime/runs/m1c_b2_zero_audit_20261001/`，其 SHA 门拒绝发生于任何真实前向之前；不得覆盖这些失败记录。

本轮只读 SSH 核验成功：远端工作树为空，HEAD `76dd70aad5fe1531d3d39df0feb75afb669c659d`，两张 Ascend910 健康，无 NPU 进程。新代码尚须 fetch/detach，不可从旧 SHA 运行。启动前在 checkout 外创建 `runtime/budgets`、`tmp`、`hf`、`xdg` 目录。

后续已在干净 detached `24570dd892202be4b5e31a4bdc12cb23dc258d57` 完成实际远端 LF 重建和零前向 dry-run，exit 0，172 张量严格/精确相等、8046 候选全覆盖、3644 文本无截断；报告 SHA `92bd8f1663d9f0a16f681ea5fa05d65d04add9b9825ca9e2a2918eacfc3f369a`。评分目录的 UID 映射已移走，所有必需 runtime 目录已建立。详见资产状态文档的 2026-10-01 段。因此 Codex 在单独授权提交中放行此完整 SHA 的 **G1 spot 1 对、0.5 卡小时**；G2 两项阻塞和其它阶段仍未放行。

## G1 命令（授权后才执行）

服务器 checkout `/home/ma-user/work/CADRetrieval`；`RT=/home/ma-user/work/CADRetrieval_runtime`，Python 为 `/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python`。远端先确认 worktree 干净，再 fetch/detach 获批完整 SHA。输出目录必须新建且不能覆盖已有结果。

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RT/hf XDG_CACHE_HOME=$RT/xdg TMPDIR=$RT/tmp \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
  experiments/scripts/run_text2cad_speed_bench.py --mode spot \
  --checkpoint $RT/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth \
  --bert-dir $RT/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b \
  --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip \
  --pair-manifest-dir $RT/pair_manifest_v4 \
  --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json \
  --selection experiments/configs/text2cad_g1g2_pair_selection_20260930.json \
  --out-dir $RT/runs/idea01_text2cad_g1_spot_20261001 \
  --device npu:0 --cards-used 1 --batch-size 1 --repo-root /home/ma-user/work/CADRetrieval
```

本轮不读取 qrels、不计算 R@K。验收是有限 logits、正确 shift/有效 x-y 因子、终止 x 因子被计入而 terminal y/padding 排除、172 键及 BERT 固定资产加载证据、预算仅扣 1 对、日志/结果 SHA 和远端 worktree 干净。失败即保存证据并回本地修复。

## G2 尚待修复的两项

1. 当前全局 measurement window 含冷启动及收尾，不能证明各长度层的实际测速并发。G2 前必须记录实际 cell/tier 测量窗口并核验双卡并发；必要时同步各层起点。
2. 当前 `--overhead-seconds` 可低于实测冷启动总和，须以实测开销为下限。七天门使用完整 `29,319,624` 对，至少 `1.30` 余量和精确公式（零其他开销下门槛约 `63.0200` 对/秒）。

因此本报告只允许下一步准备 **G1 单对**，没有批准 G2 或全量作业。
