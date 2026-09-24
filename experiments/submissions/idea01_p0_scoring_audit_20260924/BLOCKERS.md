# BLOCKERS — idea01_p0_scoring_audit_20260924

**本轮无阻断项。** 60/60 对评分成功，资产哈希、边界、有限值、源码树洁净全部通过。

## 残留不确定 / 需 Codex 知悉但不阻断 P0 收尾

1. **家族与近重复未量化**：UID 链闭合（110,263/110,263），但 UID 不代表设计家族；val 20 个候选间、以及 train↔val/test 的近重复/几何等价未测量。影响：P1+ 若宣称 family-disjoint 必须先补此项。建议见 RESULTS.md C.3。
2. **数据许可**：IEEE-T-ASE 仓库（HF model 类型）license 元数据为空。内部研究核验不受影响；任何公开再分发（含论文附件中的大段程序/文本）需先补许可链核查。
3. **EOS 行为**：`<|im_end|>` 在三条件下 logprob 均约 -21（模型几乎不预测结束）。当前如实计入 sum（符合"原样 output+EOS"合同）；若 Codex 认为这会稀释语义信号，可在 P1 把 EOS 项拆为独立诊断列（逐 token 证据已支持离线重算，无需重新跑模型）。
4. **env spec 哈希口径**：`environment.json` 记录的 env-spec 文件哈希前缀 `b6cdd884` 与准备期文档的 `10c18c17` 不一致；前者是远端 checkout 内 env json 文件本身的 sha256 前缀，后者口径未在文档中写明。环境实际版本已逐项记录（Python 3.11.10 / torch 2.7.1+cpu / torch_npu 2.7.1.post1 / Transformers 4.53.1），建议统一直接记录包版本而非文件哈希。
5. **tokenizer 警告**：对 >1024 样本 transformers 打印 model_max_length=1024 信息性警告（该 SFT checkpoint 的 tokenizer_config 上限与官方训练过滤一致）。本实现不截断，仅标记；无实际影响。
6. **environment.json 覆盖写**：`run_p0_smoke.py` 每次调用重写 `environment.json`，最终保留的是 pair 1–59 调用的版本；两次调用仅 `--pair-ids` 参数不同，SHA、设备、精度一致。若需每次调用独立环境记录，下一轮脚本应加运行级子目录。

## 已发生并已处理的小事件（非阻断）

- 本地彩排与远端审计结果与 Codex 只读初测完全一致，未发现新的映射/程序例外。
- 无 OOM、无非有限值、无边界错误、无哈希失败、无源码树污染、无超时。
