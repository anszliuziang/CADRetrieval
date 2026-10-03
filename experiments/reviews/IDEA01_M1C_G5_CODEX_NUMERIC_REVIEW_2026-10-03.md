# G5 原始结果独立数值审核

数值关口通过；完整交付仍待新的 G5 OBS 归档及最终组合 review。未重复正式评价或改变方法。

正式CPU评价从批准 `e2d823aa284c800906ed37c0a2d0272e052de96b`、授权 `4ad109025ea28557b47f8f467bfd8962882ccf34` 干净worktree一次执行；PID7240实际exit0、545.908秒，完成7934query ranks/3644组sum和mean top20/5000文本组bootstrap。15payload/15660013bytes、LF manifest `c98e205626a9d95d4830f3c628518f5f5e34f215c80b2079948dc0e54b19be95`，postGit为空。

Root 单次独立 witness SHA `007aa82bdc682a35e2b8c0e25bc2b153ca0f2a86786e4d9b9554578fcd6c3fc9` 已先审码及941toy检查。正式 witness PID19068实际exit0、552.323秒；**873364 checks，all_passed=true**。全部29319624原始score行再用于结果审核，读取时与原manifest和生产chunk fingerprints逐SHA相连；原G4数值审核未重跑。

- 所有7934query ×5方法 = **39670正例rank**：sum/mean独立阈值计数 `1 + strictly better + exact tie smaller cad_index`，长度和固定随机独立复算，全部一致。
- 全3644组两个score臂的 **145760 top20候选记录**：独立heap排序全部一致。
- 全部主/诊断/基线点指标、重复组ceilings/最小MedR、11固定分层和长度边界全部一致。
- **5000实际文本组抽样**直接展开为原query序列，重新分母/普通median，全部methodCI和pairedCI一致；draw SHA `634a14b7bc600ce5dd72a30b5c28bf16943c0f462835e917faa1def5b48a6cf6`，分母6258–10396。
- witness不import生产评价模块/scorer/torch/BERT，零模型forward。12payload/1261898bytes，独立证据manifest `4b3927e2a56971c60ecc90ee892383ec7b3f707982470da90f7c71377a67a970`。

数值事实：sum R@1/5/10/20分别0.2016637/0.9200907/1.6385178/2.6090245%，MedR3411.5；mean仅诊断。sum各top-K相对两种长度基线及固定随机的paired95% CI下界均>0；实际绝对UID命中仍低、长度分层差异大，不能推出语义检索或实用效果。完整表与结论另写最终报告。
