# IMPLEMENTATION — 统一 G5 CPU 评价入口与外部基线预检（2026-11-15）

计划：`experiments/plans/IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md` §2。范围：本地 CPU 实现、测试、已有分数回放、资产预检；**新增模型前向 0**。

## 改动文件

| 文件 | 作用 |
|---|---|
| `experiments/src/text2cad_g5_eval_entry.py`（新增） | 统一评价入口：身份/覆盖/数值门 + 排名 + 指标 + 分层 + 配对共享抽样 bootstrap + 证据落盘 |
| `experiments/scripts/run_g5_sum_replay_20261115.py`（新增） | SUM 回放 CLI：读认证 G4 chunks → 导出外部分数工件 → 经统一入口重算 → 对历史 G5 逐查询对账 |
| `experiments/tests/test_g5_eval_entry.py`（新增） | 12 项针对性测试 |
| `experiments/submissions/idea01_g5_external_baselines_20261115/` | 本报告 + RESULTS + ASSET_PRECHECK |
| `experiments/runs/idea01_g5_sum_replay_20261115/`（新增，Git 外） | 回放运行原件 |

## 实现方法与关键假设

1. **复用而非重写**：入口 `import text2cad_g5_cpu as g5`，直接调用已审计的 `link_inputs`（冻结查询/图库/qrels/映射连接）、`score_ranks`（float64、精确同分按 cad_index 升序）、`rank_metrics`、`bootstrap_statistics`、`duplicate_ceilings`、`manifest_entries`、`load_chunk`、`require_certificate`。历史正式 CLI（`run_text2cad_g5_cpu.py`）一字未改，作为原运行证据保留。
2. **分数接口**（计划 §1）：`ScoreSet` 两种来源——(a) 内存矩阵 + 声明行列身份（`from_matrix`，矩阵形状必须与声明轴一致）；(b) 外部 JSONL 分块文件（`from_jsonl`，header + 每文本一行 `cad_indices`/`scores`，逐行身份表与声明轴逐一比对——错位/短行在加载或评估时拒绝）。`direction ∈ {higher, lower}`（lower 先取负再排名）；dtype 声明记录，排名一律 float64。
3. **门**：文本轴必须恰为 0..3643 全集（缺文本拒绝）；cad 轴全库模式必须恰为 0..8045 密集集，子库模式必须恰等于显式声明的 `cad_subset`；每 (文本,候选) 恰出现一次；分数全部有限（NaN/inf 拒绝）。子库模式下正例不在子库的查询在计算任何指标前冻结剔除，kept/dropped 计数写入结果。
4. **配对臂**：`paired_sum_set` 声明同一轴（不同轴拒绝），两臂在同轴重排，共享 `default_rng(seed)` 抽样做文本组 cluster bootstrap，输出配对差与 95% 百分位区间（R@1/5/10/20、MedR）。子库模式下组 id 重新密集化（空组文本剔除）。
5. **证据**：每个输出目录写 `query_ranks.jsonl`（query_id/text_index/正例 cad_id+cad_index/rank）、`text_top20.jsonl`、`results.json`（schema `text2cad-g5-external-eval-v1`，含全部输入指纹与 zero_npu/forwards=0）、`SHA256SUMS`。
6. **回放外部化工件**：SUM 分数先落盘 `sum_scores_full_axis_f64.npy`（原始 <f8 字节）+ 身份 sidecar JSON（轴/方向/dtype/来源 manifest 与证书 SHA/矩阵 SHA256），再从磁盘重新加载（`np.fromfile` + SHA 复核）经 `ScoreSet.from_matrix` 进入评价——证明外部工件路径端到端可用，评价端不依赖读取方的内存。
7. **对账**（脚本内置硬门）：逐查询 (query_id, 正例 cad_id, SUM rank) 与历史 `query_ranks.jsonl` 三元组全等，要求 7,934/7,934；hit@1/5/10/20 = 16/73/130/207、R@K 小数与历史 `metrics.json` 全等、MedR = 3411.5、长度四分位切点 = [27,40,70]——任一不符即非零退出。
8. **已知边界**：内容级"行列值互换但身份声明自洽"的错位不可由评价端单独检测——协议以分数来源指纹（sidecar SHA）+ 回放对账覆盖；JSONL 逐行身份表可检测排列/短行类错位（有测试）。

## 复跑命令

```
python -m pytest experiments/tests/test_g5_eval_entry.py -v
python experiments/scripts/run_g5_sum_replay_20261115.py --repo . --data-root . \
  --runtime-root D:/projects/IdeaSpark/CADRetrievalIdea1_runtime \
  --output experiments/runs/idea01_g5_sum_replay_20261115
```

实际退出与逐条对账结果见 RESULTS。历史 G4 chunks 读取一次（29,319,624 行 JSON，本地 CPU 实测约 6–8 分钟）；评审可复跑，但**不得覆盖已交付的 run 目录**（入口要求输出目录不存在）。

## 已知缺口

- 入口当前要求文本轴全集（计划口径：覆盖缺口在候选侧用子库模式表达）。若未来某外部方法连唯一文本轴都不全（应不发生——查询是给定 7,934 条 L0），需扩展声明。
- OpenShape/MM-CAD/CLIP 的运行依赖均未安装（见 ASSET_PRECHECK §5），本轮零下载零安装；启动前需一次授权的依赖安装。
