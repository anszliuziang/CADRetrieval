# Text2CAD G3：100条代表性文本对完整图库的持续运行验证

日期：2026-10-02。前置：G2 batch16双卡已退出0，Codex独立核查26,254项通过，99组rep最低窗口重叠0.992359，完整矩阵保守ETA为38.28394小时（含慢卡、冷启动、1.30系数）。本合同先授权本地实现/CPU验证，NPU额度0；审核后再在ACTIVE_TASK批准具体SHA与运行预算。

## 1. 目的与不变条件

检验长时间运行、每query实际8046条输出/原子落盘、断点文件和未见文本长度对吞吐的影响。这里只复核全量是否可在七天内完成，不根据试验分数调整模型、score、query或gallery。G1 batch32失败保留；本轮仅batch16、FP32、完整272行输入、无trim/无BERT cache。

完整官方A仍为8046 CAD、3644唯一文本，恢复7934原查询的指标留到正式全量评价。G3只计算100种唯一文本×8046 CAD=804600对；每卡固定50种文本、402300对。这个子集不能替代完整29319624对矩阵。

## 2. 事前固定的盲选清单

Kimi在本地实现纯CPU selector，读取已有四文件worker清单、已核验token审计和已冻结G2配置；不加载tokenizer/BERT/生成模型，不读取qrels/UID映射/原CSV/任何模型分数。

- 每个正式文本分片[0,1822)、[1822,3644)内，按 `(实际BERT token长度, text_index)` 排序，取位置 `round((1822-1)*j/49)`，j=0..49。排序并去重后必须每片50条，包含每片最短与最长，保留原全局text_index，不重排或缩减worker图库。
- 全局token长度排序的三层定义与G2相同：`min(2, 3*rank//3644)`；清单记录每个probe的text层、长度、固定分片以及每片各层完整文本数。每片每层须至少一个probe，不能缺层后用其它层替代。
- 新配置 `experiments/configs/text2cad_g3_probe_selection_20261002.json`，LF、明确schema/规则/原清单四指纹/token审计指纹/G2配置指纹、100索引及50+50分片。只在本地生成并Git提交；不可提交空配置/假SHA，不在服务器生成配置。
- Codex从固定源数据独立复算清单/分层/预算及SHA，未通过前不能NPU执行。

## 3. 最小实现范围（Kimi本地，CPU预算）

1. 新增一个纯CPU probe清单/验证/ETA模块和必要薄CLI，或在现有模块加入小型明确入口；避免重写评分器。selector拒绝覆盖输出，验证器拒绝错schema、重复/越界/错分片/错资产指纹、非官方图库尺寸或读取额外qrels文件。
2. 现有 `text2cad_frozen_runner.py` 增加可选 `--probe-selection`，生产score只接受本轮Git内固定路径及严格重算验证。没有该参数时完整矩阵行为保持不变。G3保持原PairSpace/global pair_index及相邻CAD批界，`texts_per_chunk=1`；只筛选当前分片清单中的query chunk，每个选中query仍遍历全部8046 CAD。不能用max-pairs截断、重建100条小manifest或改全局text索引冒充原图库。
3. 输出和resume必须绑定probe清单SHA/运行scope；混全量、外来probe清单或非本shard文件时拒绝，不能复用未经检查的chunk。保留原完整chunk密集pair顺序/状态/计数/代码/资产/score绑定核验，不覆盖已有文件。
4. 原始每query计时记录包含全局text_index/分片、真正start/end epoch、score/序列化/原子写盘完成的E2E、实际pair数/批数/有效和失败数。第一query不另外加模型warmup；预算只能804600评分对。计时必须包括实际chunk落盘，不能用进程总启动窗口冒充稳态评分窗口。
5. 准确保留模型/候选加载冷启动、NPU指定设备峰值HBM、累计对数/卡时/账本、真实结束状态与代码/资产/清单指纹。两进程各cards_used=1，固定npu0/shard0及npu1/shard1，native模型/score/shift/mask/FP32不改。
6. 新CPU ETA入口读取两个成功的真实probe summary和对应chunk timing。核各50条、8046/条、没有缺层、同host/不同device/对应shard、完整成功账本和同代码/模型/score/probe指纹；两卡的从首个query开始至末query写盘结束的实际评分窗口重叠至少较短窗口80%。不能理想倍乘单卡。
7. 每卡每个全局text层，取该卡该层probe **最慢完整query E2E** × 该卡该层完整query数，加该卡实际冷启动；两卡取max、乘1.30，严格比较604800秒。明列每层数量、probe/最慢query、各卡成本、额外开销与原始输入；不因超限改系数。G2/G3两个门都通过才能自动准备全量。

## 4. 必要CPU验证与代码交付

只测新增风险：选择可复算且盲于qrels；原全局索引和完整图库；生产16/尾14；完整与probe规划/resume隔离、缺行/重行拒绝；每chunk真实计时包括写盘；只对选中chunks算预算；错资产/设备/分片/账本/窗口拒绝；较慢卡+每层最大query+冷启动+1.30的ETA。用小型假数据/假模型，不进行真实模型/BERT前向，不安装依赖。

复用已有受影响runner/protocol测试即可，不要求无关全套重跑。交 `TEXT2CAD_G3_IMPLEMENTATION_2026-10-02.md`：改动、CPU测试与实际清单、完整实现SHA、准确双卡/CPU ETA命令、已知限制。只提交自己src/scripts/configs/tests/新submission，不改Codex计划/ACTIVE/CLAUDE/reviews/旧报告，不push；review_required后停。Codex审源代码、独立验CPU清单/受影响测试后才push及批准NPU。

## 5. 后续运行上限（当前未授权）

审核后预定run_id `idea01_text2cad_g3_probe_20261002`，输出在云脑runtime的run/card0、card1，各一次score调用；总804600对、最多12卡小时/6小时墙钟，timeout21600秒。启动两进程可错开3秒避Git fetch锁；必须保留真正并发窗口。当前正式全量不授权，G2余额/账本不可复用。

源/配置/执行脚本仅本地编辑、Git到 `yunnan-share2:/home/ma-user/work/CADRetrieval` 检出完整批准SHA运行，源码树始终干净。既有环境 `/home/ma-user/work/envs/cadretrieval-idea01-torch271` 不变；资产/缓存/结果均在checkout外。Kimi等待/轮询单次不超过30秒，长作业启动和状态确认后可交接给本线程heartbeat，不保持一个150秒以上的阻塞sleep。

保存启动命令/PID/环境/sha、raw stdout/stderr/退出码、100个完整query chunks/timing、共享账本快照和失败/partial证据；生成SHA256SUMS，取回本地并归档 `cvteam14:/data/obs_data_liuza/CADRetrieval/runs/<run_id>/`。cvteam14只存储不用GPU；不依赖 `/data/data_dev`。源目录不删，未知目录/脏Git/资源被占/错误/OOM/NaN/超预算/错hash均停止，不自动降批/截断/改清单/重试。

G3成功且Codex重算保守ETA≤七天后，按用户已获授权自动制定/审核/批准完整矩阵G4。若超限或有新的科学/数值/环境阻塞，先报告给用户；不把100-query结果当正式全量完成。
