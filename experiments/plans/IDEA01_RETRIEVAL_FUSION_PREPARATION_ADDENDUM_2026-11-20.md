# 表 3 CadTranslator–Fusion 360 几何准备补充约定

本约定落实三表计划第四节的数据准备。输入是锁定预清单中的 6,757 个候选和 6,502 条查询。图库最终索引按规范 Fusion UID 升序分配。脚本提供本地 CPU 的 `extract`、`validation-index`、`replay`、`freeze` 四个入口。 [几何准备入口](../scripts/prepare_retrieval_benchmark_fusion_gallery.py)

## 重放与保留规则

- 读取 CadTranslator 发布的原始 HDF5 和短句，逐件校验预清单中的原始命令 SHA-256 和去首尾空白后的原文 SHA-256。候选以三表计划原始命令查重后的清单为入口。
- 用 DeepCAD 发布代码的字段语义独立重放 HDF5。单侧使用有符号方向，对称在两侧各使用一次深度，双侧分别沿法向和反法向使用有符号距离。双侧模式需要两个有效的棱柱，任一侧深度为零时在 `h5_replay` 阶段记录 `HDF5 two-sided extrusion has a zero-depth side`。使用 HDF5 成品 B-rep 的 bbox 最小点及最长边，将几何平移并等比缩放到 0.75 的最长尺度。浮点转码与量化转码共用此坐标框架。
- 逐件执行严格轮廓解析、HDF5 参考重放、Text2CAD 浮点重放、8 位张量校验、量化后重放和几何比较。继续使用 E1 的 272 步上限、有效 B-rep、相同实体数、双向 P99 各至多为参考对角线 1%、相对体积差至多 5%，以及圆切轴内部材质检查。保留每阶段失败与深度模式标签。 [固定 DeepCAD 查看器](https://github.com/rundiwu/DeepCAD/blob/f7b350790a3bce0267a5345339fd09be0250c8cb/cadlib/visualize.py#L36-L54)
- 使用 E1 已审计的 train 159,049 和 test 8,046 个 `cad_vec` 指纹做候选查重索引，按官方 validation 8,925 个 UID 补齐。SHA-256 命中后从发布的 PTH 原件读 `cad_vec`、`flag_vec`、`index_vec` 和 `key_padding_mask`，四个张量逐值全等时排除。每条排除记录保留全部源划分及匹配 UID。
- 保留通过几何和源端查重的无短句候选。将接受件的 HDF5 参考 B-rep、Text2CAD 量化重放 B-rep、输入张量及指纹保存在每 UID 子目录。主清单中的 `member` 是相对运行目录的量化张量 NPZ 路径。几何参考适用于本次 HDF5 表示，发布包的 Autodesk 文件名对照尚无公开资产。

## 顺序、重跑和交付

1. `extract` 使用既有 h5py 环境，校验冻结清单与发布压缩包。可重复提供 `--pilot-uid` 首先校验固定的小批输入。
2. `validation-index` 读取官方 split 和发布的 `cad_seq.zip`。小批使用固定 validation UID。完整运行读取 8,925 个 validation 向量的一阶指纹，复用已有 train 和 test 指纹。
3. `replay` 在一个本地 cadclip CPU 进程内顺序处理。入口校验实际 Text2CAD 修订、输入及源索引，按对象原子写入证据。续跑复核已存张量、B-rep 和绑定摘要后继续缺失对象。整份 ledger 汇总候选数、阶段及挤出模式。
4. `freeze` 在整套预清单和三套源划分覆盖齐全后发布 `gallery.jsonl`、`gallery_fields.npy`、`texts_ct_fusion.jsonl`、`queries_ct_fusion.jsonl`、`qrels.jsonl` 与 `freeze.json`。每个文本分组按精确原文 SHA 排序，每个查询仍以自己的 UID 为唯一正例。小批运行的产物为独立诊断，正式图库等待完整运行核验。

源端 64 件及 8 件双侧补充的检查展示了几何和量化的实际覆盖。双侧补充中 7 个可比较样本的浮点转码全部保留 HDF5 几何，量化后有 4 件通过。源端压力估计约 3.9 至 6.0 小时，每件实际耗时也会随复杂度改变。正式提交前由执行方审查目标小批的覆盖和墙钟，再启完整数据准备。冻结的 48 条源背景名单摘要随图库清单记录，计分阶段按最终图库另算各候选背景分。 [源端及输入预检](D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_retrieval_benchmark_fill_20261120/fusion_preflight/FUSION_PREFLIGHT.md)
