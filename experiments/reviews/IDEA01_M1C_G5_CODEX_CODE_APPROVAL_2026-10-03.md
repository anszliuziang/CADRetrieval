# G5 固定 CPU 评价代码批准

批准代码完整 SHA：`e2d823aa284c800906ed37c0a2d0272e052de96b`。审查范围从 `b022770c9d091e31a089076193a09cebf58f4858` 到该 SHA；无 Critical/Important/Minor。仅批准既有 G5 预注册的一次本地 CPU 评价，不授权新模型评分或更改协议。

## 证据与判断

Root 21 项必要测试、固定输入连接 preflight 通过。唯一 fresh `g5_cpu_review` 独立重读实际代码，21 项针对 CPU 测试真实 exit0，独立展开 query 的参考实现核对全部10种 toy 文本组计数和5000次 bootstrap 的五方法 recall/MedR/paired CI，变化分母与重复四分位边界均一致；真实未授权/dirty CLI exit1 且未创建输出。Root 另以100次 toy draw、四种组大小核对2600项直接展开统计，全部一致。

独立见证 `CADRetrievalIdea1_runtime/g5_fresh_review_20261003/review_e2d823_001/review_witness_manifest.json` SHA `369621fd59cdfdd35716dc23544de40c081a97d076db4027f0602b7be5833e0b`。未读取真实 matrix 或计算真实检索指标。

sum 主表与 mean 诊断分离；完全相等 score 按 cad_index 升序；固定 UID/文本/成员和全部 SHA 连接；7934原查询等权；5000次文本组抽样保留组内全部query并重新计算分母和普通中位数；五臂共享draw，paired gain CI正确。拒绝 partial 证书、未授权 SHA、dirty checkout、错误输入/chunk和旧输出。代码仅 stdlib/NumPy，无 scorer/torch/BERT 前向。

## G4 额外证据归档已完成

最终矩阵证书仍为 `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6`，full_matrix_approved=true。新的失败现场/诊断/v2运输/独立OBS审核/最终连接证书组成299文件、2686355bytes小型证据包；manifest `49c536daecefb727d3dd4abcb7919f4d2b4987be7a74366b95f67f0422999e31`。本地及 `cvteam6:/data/obs_data_liuza/CADRetrieval/runs/m1c_g4_codex_final_archive_proof_20261003` 全SHA/覆盖独立通过、无extra，真实tar/ssh/remoteSHA/独立byte witness/controller全部exit0。保留旧部分目录和operator stop，不覆盖Kimi草稿。

## 正式执行合同

另提交 ACTIVE pilot_authorized/完整批准 SHA；干净 detached 本地worktree从该Git脚本一次运行，外层捕获实际退出、argv/PID/日志。全7934rank、3644top20、原始指纹/CI/基线/分层与真实退出均归档，本地及OBS逐SHA。评价后独立从原raw复算全部正例rank与指标再写研究结论。本批准不声明检索精度。
