# Text2CAD G2 真实预处理资产与配置冻结

2026-10-01，Codex完成；**CPU tokenizer与清单计算，零模型前向、零NPU评分**。

## 运行证据

- 本地已审R3报告HEAD：`598dd40b0cf6f8ce87003225c0ba3ee434abb450`；远端从干净checkout通过Git fetch/detach此完整SHA，执行 `run_text2cad_speed_bench.py --mode audit-text-lengths`。执行前后 `git status --porcelain` 为空；没有远端源码编辑。
- 使用云脑现有隔离Python环境与固定BERT目录，只构造BertTokenizer，不构造BertModel；HF离线、no bytecode、cache/tmp指向runtime。
- 真实token资产：`/home/ma-user/work/CADRetrieval_runtime/derived/text2cad_text_token_lengths_20261001.json`，SHA256 **`9abfd5b403d0f117ca70d512b4b5ab3a8020d8b5ec08b8b2ad7ad895f62627e8`**。
- 取回本地 `D:\projects\IdeaSpark\CADRetrievalIdea1_runtime\derived\` 后重算SHA一致。3,644条索引连续、覆盖全部唯一文本；含special tokens的实际长度4–98，零超过512、无截断；provenance中`model_loaded=false`，texts/BERT指纹与已审资产一致。

## 本地配置生成与独立审核

只在**本地**运行已审 `--mode select-g2`，输入为干净 `pair_manifest_v4_worker` 与取回的token资产；生成到本地checkout外，审核后复制进入Git的 `experiments/configs/text2cad_g2_pair_selection_20261001.json`。没有在服务器生成配置。

- 配置273,366字节、LF，SHA256 **`fe42096c5ff63e8532bcb80f9c2cdb2175eb10f5d00679013c36899d58571dce`**。
- Codex独立脚本不import模型/评分器：由真实长度重新计算排序/分位、相邻CAD窗口与十层、两段各1,822文本分片、各cell权重和每cell暖身预算；**1,187项检查全部通过**。
- 六档1/2/4/8/16/32每档完整pair和严格29,319,624；完整批数分别29,319,624 / 14,659,812 / 7,331,728 / 3,665,864 / 1,832,932 / 918,288。8046不能整除4/8/16/32时，tail分别为2/6/14/14；tail单独计权，不补重复CAD。
- 64对为8×8真实长度分位网格，包含4-token/19行最短与98-token/212行最长组合；原spot/ten_pair与旧G1配置完全一致。
- 每卡G2六档计划评分480/960/1980/4020/8100/15780，合计31,320；两卡62,640。尾层只有一个真实窗口的shortfall如实记录（24项），为尾层定义所致，不降低真实批或补样。

## 归档与复查

- 本地原始镜像：`experiments/runs/m1c_g2_cpu_preprocess_20261001/`；含token资产、配置候选、`CODEX_SELECTION_CHECK_20261001.json`、独立审核脚本、原CPU stdout/stderr与SHA256SUMS。
- 共享OBS：`/data/obs_data_liuza/CADRetrieval/runs/m1c_g2_cpu_preprocess_20261001/`。通过cvteam14只做存储操作，不占其GPU；六份文件在OBS按SHA256SUMS逐项验证成功。
- tokenizer日志另保留云脑/local `runs/m1c_cpu_token_audit_20261001/`。R2/R3 CPU测试原始日志仍在本地checkout外对应runtime目录。

## 下一门

代码和真实配置已满足704对扩展G1的输入准入。Codex另提交ACTIVE_TASK授权**单卡npu:0、固定704对、最多1卡小时**并钉完整运行SHA；通过原始logits独立复算后，再规划实际生产profile与G2。当前尚无新增模型评分、吞吐、七天ETA或R@K结论。
