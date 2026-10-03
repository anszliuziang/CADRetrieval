# G5 固定 CPU 评价实现交接

状态：review_required；真实检索指标尚未执行。执行者 Codex，用户已明确授权接手。完整 G4 证书 SHA `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6` 为前置门。

## 文件与协议

- `experiments/src/text2cad_g5_cpu.py`：纯 stdlib/NumPy，原 query/UID/strip 文本/成员/SHA 连接；每组完整8046候选的 sum 主表与 mean 诊断精确排序；cad_index 升序打破完全相等 tie。
- `experiments/scripts/run_text2cad_g5_cpu.py`：需干净 detached 完整批准代码 SHA、独立授权 ref（ACTIVE pilot_authorized/G5/零评分/零NPU）；最终 matrix certificate、八个冻结输入和全量原始清单/实际chunk SHA均绑定；拒绝旧输出。
- `experiments/configs/text2cad_g5_cpu_frozen_20261003.json`：所有预注册输入 pins、seed20261003、5000抽样次数和两 score/三基线固定。
- `experiments/tests/test_text2cad_g5_cpu.py`：针对统计与完整性边界的手算 toy。

代码不 import scorer/torch/BERT，不创建新精确分数矩阵；每chunk读取现有JSONL并验证其原始哈希，再提取全部正例rank和两score臂top20。所有7934原query等权；组bootstrap有放回抽3644text，每次保留组内全部原query并重新计算分母/普通中位数，所有方法共享draw，paired sum gain CI针对两个长度基线/seeded随机。

基线 n_valid_x+n_valid_y 升/降两臂来自固定长度文件，seeded随机每唯一text独立排列并被重复query复用；analyticK/8046单独说明。重复组正例UID互异验证后给固定ceilings及最小MedR；固定group size/unique-repeated/无监督gallery长度quartile分层，空层null。

## 已执行验证

- 初始18项测试在缺实现时全部真实 RED；实现后18通过。
- 增加真实临时Git授权/错误输入chunk/空层测试；发现缺失授权文件传播 Git 子进程异常类型，局部改为明确 ValueError（准入本来拒绝，未改变门）；最终 **21 passed、actual exit0**。
- 正式固定输入只做 SHA/UID/文本/成员连接 preflight，7934/3644/8046全部通过，无真实排序/检索指标。
- checkout外证据 `g5_cpu_targeted_tests_20261003_stdout.log`、同前缀 stderr/exit.json、`g5_cpu_input_linkage_preflight_20261003.json`。不重复G1/G2/G3或旧全量数值审核；用户已明确限于必要新门，不跑无关历史全套。

## 正式运行方式（尚未授权）

独立审核后先提交明确CPU授权；本地新干净detached批准worktree，只从其Git内脚本运行一次：

```
python -B experiments/scripts/run_text2cad_g5_cpu.py --data-root D:/projects/IdeaSpark/CADRetrievalIdea1 --runtime-root D:/projects/IdeaSpark/CADRetrievalIdea1_runtime --output <新的G5原始输出目录> --approved-sha <完整审核SHA> --authorization-ref <完整CPU授权提交SHA>
```

设置PYTHONDONTWRITEBYTECODE=1、tmp/cache外置；无需装包/下载/远端新Git或占卡。外层独立 operator 捕获真正子进程退出，最后固定manifest包含全部7934ranks/3644top20/metrics/bootstrap/strata/input及codefingerprints/stdout/stderr/退出。运行健康仅读进度，不重复启动。

审码及正式评价后还须独立raw正例rank/全部指标/基线ceilings和统计逻辑审核、本地与OBS完整SHA以及事实/解释/建议报告；本提交不声明研究精度或全部任务完成。
