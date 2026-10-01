# Text2CAD 固定64对 G1：Codex 独立结果审核

日期：2026-10-01。审核者：Codex。执行者原报告保持不改。

## 1. 裁决与当前状态

**交付证据可审核；原数值门确实失败，未批准后续前向。**

- 获批运行 SHA：`a2219e25b7f018aa12e820616466672fd8d70729`。
- 授权文档 SHA：`c45cf32e7937d7e4d963796c6fbfd7b0b48322b5`；独立从 Git 对象读取 ACTIVE_TASK blob，其 SHA256 `9a07bf6febc1e7601b5d58cf859614a67423b4bca9901a7d2e1dd72f65650523` 与 runner 记录一致。
- 唯一 invocation：`20261001T062948Z-4fb6b1fc`，exit=3，账本已 finalized。
- 实际完成 reference_b1、batch16、batch32 各64对，共192/704对；后续8个 trim/cache 变体没有运行。未自动重试、未改容差、未扩大运行。
- 本地 Kimi 报告提交为 `a5ee50aaeb1b7a14ebaf892f1c8754cb89fb2771`，SHA 回填提交为 `b18b44c5de986e86d75a48aad6ca9dd4d57f0d8f`。

本轮没有 R@K，没有正式双卡吞吐/七天 ETA；不能据此宣称检索有效或无效，也不能启动完整矩阵。

## 2. 独立复算方法与证据

本地 CPU 审核脚本不导入项目 scorer、不加载模型、不进行新的模型前向：

1. 检查运行 SHA、授权 SHA、固定配置 SHA、候选索引文件与官方 CAD ZIP SHA、实际变体顺序、每档真实批尺寸、失败后停止及账本/退出码。
2. `weights_only=True` 读取官方 ZIP 中原始 CAD 向量，以及3份完整有效行原始 logits。独立确认272×2向量、START/terminal、连续有效前缀、逐头 padding mask；按 `vec[1:L]` 推导目标与逐头有效计数。
3. 独立 FP32 `log_softmax`/目标 gather，再以双精度 `math.fsum` 求逐头和总分；核对报告中的192对分数及计数。审核 CPU torch=2.12.0+cpu，原前向 torch=2.7.1+cpu/torch_npu=2.7.1.post1；二者 FP32 log_softmax 的少量实现差异不冒充同一数值结果。
4. 从原始 FP32 logits 转 double 后逐元素复算固定容差：`1e-5 + 1e-4*abs(reference)`；score 容差仍为 `1e-4 + 1e-6*abs(reference_score)`。不放宽、不忽略超差元素。

**2,816 项完整性/数学/报告一致性检查通过；原门失败得到独立复现。** 这里的检查通过是证据审核通过，不是把失败的数值门改成通过。

审核文件在原始 run 的独立 `codex_check/` 子目录：

- `text2cad_g1_profile_codex_audit_20261001.py`
- `CODEX_RAW_LOGITS_CHECK_20261001.json`，SHA256 `d85c9b66922fc257ecff5887de7399ba2481a0222e0d867d3b5dea8bc6a7521f`
- `SHA256SUMS`（仅本子目录的审核资产，不改原始产物清单）

原始 JSON：`bench_profile-consistency_20261001T062948Z-4fb6b1fc.json`，SHA256 `0da99847d794bbfb5460a11d168afb5000b84862e5c556fd52802e3230607c00`。

## 3. 数值结果

每档比较64对，共2,516,208个 logits 元素。

| 档位 | logit 超差元素 | 最大差值/容差 | 逐头/总 score | CPU复算最大score绝对差 |
|---|---:|---:|---|---:|
| batch16，全长，无缓存 | 0 | 0.955790 | 原容差内 | 3.2764e-5 |
| batch32，全长，无缓存 | 6 | 2.051194 | 原容差内 | 6.3088e-5 |

CPU复算与 NPU 报告中的逐头/总分差异最大为2.3544e-6，均在原score容差内；表中 CPU 重算差值与 NPU JSON 原数值略有不同，不能混写成同一实测。

batch32 仅第7对（零基索引，text=160，CAD=164，212个真实CAD行）出现超差。详情如下；行号是预测行零基索引，class 是输出类别索引：

| 预测行 | 头 | class | 实际目标 | 此头此行参与score | 差值/容差 |
|---:|---|---:|---:|---|---:|
| 70 | y | 54 | 0 | 否 | 2.051194 |
| 83 | y | 171 | 11 | 是 | 1.386669 |
| 83 | y | 100 | 11 | 是 | 1.035872 |
| 136 | y | 263 | 0 | 否 | 1.076864 |
| 136 | y | 37 | 0 | 否 | 1.066425 |
| 136 | x | 28 | 3 | 是 | 1.031155 |

需要更正执行者报告 §7 和最终摘要的“6行”措辞：**6个元素分布在3行，其中5个y头、1个x头；3个元素所在头/行不参与评分。** 原始 JSON 与代码的 `n_logit_violations` 实际统计元素数量，原结果未算错。另一半位于参与评分的头/行，虽非目标类别，其 logits 仍进入 softmax 分母，不能说六个全都无影响。

第7对 CPU重算总分 reference=-327.9101461931132，batch32=-327.9101331937127，绝对差1.29994e-5。全部64对的逐头及总score均通过原容差。这支持“此样本的评分漂移很小”；不足以证明全库排序或 R@K 不变。

## 4. 对代码、环境和交接的判断

- 运行代码与此前R3已审版本一致，批32确实是两批32对；没有沿用旧十对测试的假批尺寸。本轮未出现新的 shift/mask/count、NaN、OOM、截断、账本或泄漏证据。
- fail-fast 按合同工作：先保存导致失败的原始 logits，再停止，非零退出，并按实际192对计费。剩余512对是历史额度余额，不是新的授权。
- 2026-10-01 只读再次 SSH 核验：远端 HEAD仍是批准 SHA，`git status --porcelain` 为空，两张 NPU 健康且无运行进程。
- 泛化环境字段 `environment.batch_size=8` 是未使用的CLI默认值；实际 profile 批尺寸由 reference/variants 与 `real_batch_sizes` 记录。不能把这一字段当作本轮实际批尺寸。
- 所有原始 logits 有限。stderr 有 torch_npu 安装目录 owner 和驱动版本解析警告；没有证据把微小批差异归因于这些警告。当前未定位到具体算子/精度设置/归约顺序，不能断言根因。
- 前向张量dtype为FP32，不等同于已记录所有NPU内部计算精度设置。若另行排查32批，应明确采集有效精度设置并一次只改一个因素；本轮没有批准此诊断运行。
- 原始11项产物在本地与共享OBS的 SHA256SUMS 全部独立核对通过，清单自身 SHA256 `2d232f224ad39713edd7aaf27d6ae9cefa652d70a67ac10aa05070cc807514bd` 与三方记录一致。CPU审核另行归档，不覆盖原始文件。

## 5. 下一步建议，等待用户讨论

**建议先保留原容差，用已通过的 batch16、完整CAD序列、无文本缓存配置做代表性双卡测速。** 这条路线不依赖32批、trim或cache通过；先得到可靠的全矩阵保守 ETA。若它已满足七天上限，就可以继续完整精确评分的准备。

实施前须另写具体子计划与受限测速配置，经本地代码/配置审核、Git提交后再在 ACTIVE_TASK 激活；不能用现有六档计划或704门的余额直接继续。测速也须确认使用同一已验证计算路径，并遵守真实生产批界、两个固定文本分片和完整预算。

另一个选择是先定位batch32的算子/精度差异，再复验。这可能提高吞吐，但当前尚不知道16批是否已足够快，诊断耗时也未知。

不建议看到失败后直接放宽 logits 容差或删除检查。原始 logit 逐元素门比最终score门严格，且包括部分不参与score的头/行；今后若要从研究目标出发重审验收定义，应明确解释并另作前瞻验证，保留本次失败，不把它改写成通过。

## 6. 数据位置与恢复入口

- 原始本地镜像：`D:\projects\IdeaSpark\CADRetrievalIdea1\experiments\runs\idea01_text2cad_g1_profile_20261001\`
- 云脑原始目录：`/home/ma-user/work/CADRetrieval_runtime/runs/idea01_text2cad_g1_profile_20261001/`
- 共享OBS：`cvteam14:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g1_profile_20261001/`
- 执行者报告：`experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_G1_PROFILE_RESULTS_2026-10-01.md`

**恢复时先读本审核与 ACTIVE_TASK。当前没有新模型前向授权，也没有全量作业在后台运行。**
