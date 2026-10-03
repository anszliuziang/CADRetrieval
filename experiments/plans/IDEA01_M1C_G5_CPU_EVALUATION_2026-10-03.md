# G5 CPU 评价预注册（归档关口通过后才激活）

2026-10-03，Codex接手后冻结；本文件写入时尚未读评价qrels或计算检索精度。当前只允许准备规则，不授权实际评价。完整 OBS 归档及 Codex 最终矩阵审核是前置门。

## 任务与数据

用已完成29319624对/3644唯一strip文本×8046CAD的原始分数，恢复7934原nonempty L0查询；Protocol A官方8046 UID图库。评价映射、原query/gallery/qrels只在CPU评价入口读取，评分不重跑。输入字节以固定pair_manifest_expected、M0冻结manifest和G4全量审核SHA绑定；逐query检查文本相等、strip、SHA、UID唯一正例及cad_index/member连接，不能从报告猜标签或按指标排除样本。

没有人工新增正例。这里评测登记UID检索，不能把未命中UID的CAD判定为几何/语义错误。IMMA发表数字对应不同gallery协议，不能直接并表称同benchmark复现。

输入路径：本地`.cache/m0_real_run_v3/query.jsonl`、`gallery.jsonl`、`qrels.jsonl`，其冻结SHA分别a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd、09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30、f0677928ae2412ca4bcd8952c95d89f0568c5bb214c7bb53e82e0059555c87d6（来源M0 Codex review，不从本轮指标选择）。映射用checkout外runtime/pair_manifest_v4/evaluation_map.json SHA a6f5b19a76a14aef4eab03879ae9f0f872b9e5a30e4bda5ba9ea8a45c0ed1875。worker四文件用runtime/pair_manifest_v4_worker及pair_manifest_expected既有pins；qrels未复制进worker。本阶段正式激活前不读标签/计算精度。

## 冻结排序和指标

- 主表：`score_total`（两个原生输出头logprob sum），降序。诊断：`mean_total`独立表，不据test表现选择主方法。
- 完全相等score固定 `cad_index` 升序tie；不做epsilon合并，不随机破tie。
- 每原query记录正例 rank（1-based），R@1/5/10/20、MedR（普通中位数，偶数平均中间两rank）、MeanR作为补充。主汇总按7934原query等权，不能把3644文本组等权混入主表。
- 每唯一text完整排序8046CAD，只读该chunk。保存正例原queryrank及每text两score臂top20，不另保存/重复评分新matrix。
- 固定无监督查询无关长度基线：`n_valid_total` 升序、降序各一行，cad_index升序tie。有效长度从pinned cad_length_index独立取得；不能使用正例文本/分数来选长度规则。
- 随机期望：R@K=K/8046，期望rank=(8046+1)/2；后者是随机rank的期望，不冒称有限样本MedR测量。另报告seed20261003每唯一text独立均匀排列的固定随机实测，重复query复用同text排序。
- 重复组不可辨识上限：在每text正例UID互异前提下，R@K上限=sum_t min(K,n_t)/7934。同时报告理论下界MedR的合并rank集合 {1,...,n_t} 的中位数，并如实说明只代表登记UID目标。
- 固定分层：group size1、2–5、6–20、21–100、>100，空层记录n=0而不造数字；另unique vs repeated。按有效CAD长度四分位分层使用gallery全体分位点（与score/qrels无关），边界重复按值比较；先冻结再报告。

## 统计

5000次percentile95% CI，seed20261003，**抽样单位是3644唯一text组**，有放回抽3644组；每次保留被抽组内所有原query，重新计算按query加权分母、R@K及MedR。sum/mean/长度/随机共享抽样；报告sum相对两个长度基线和随机的paired R@K差值CI，不能把mean更高说成验证集选优。相对随机的倍数以点估计/analytic K/N计算，不将极小分母倍数代替实际百分点增益。

不做token归一化参数搜索、CPN/空query前向、阈值调整、额外训练或new model。现有一次frozen run的CI是查询组不确定性，不是模型重训seed变异。

## 代码与证据门

Codex本地实现独立CPU模块/CLI，不import scorer/torch/BERT；纯numpy/stdlib。新增代码独立审码，toy测试涵盖tie、恢复重复query/多UID、qrels缺失/重复拒绝、长度排序、ceilings、group bootstrap权重/median和rank方向。正式入口须完整 approved SHA 与最终G4review证书，拒绝partial/full_matrix_approved=false、输入SHA错、旧输出目录；精度只运行一次，不因结果低重跑或调方法。

保存评价源码/配置/输入/全量chunk原件指纹、命令/PID/真实exit、全部7934queryrank/3644组top20、指标CI分层、时间、bootstrapseed与规则；本地和OBS完整manifest逐SHA。Codex独立从raw分数复算至少全7934 positive ranks及全部主指标、基线/ceilings，并审统计逻辑；支持结论由结果决定，不承诺优于随机或SOTA。

## 研究交付

新CODEX报告区分事实、解释和可检验后续建议。核心判断是冻结Text2CAD原生sum是否在该Protocol A exact UID库超过随机/长度捷径；即便成功也不推出所有CAD生成器、CAD语义理解或纯MLE likelihood成立。未获新授权不自动做额外NPU实验。完成归档+评价+独立审核+报告后结束Text2CAD监控。
