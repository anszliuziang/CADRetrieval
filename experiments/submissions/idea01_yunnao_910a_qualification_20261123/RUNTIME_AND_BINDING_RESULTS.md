# 910A 准入与两种生成器的 40 对几何辨认

执行日期为北京时间 2026 年 11 月 23–24 日。本轮先调通新机器，再执行已量测的两开发家族题库。实验只使用已登记的开发题、既有历史对照及训练侧的工程样例。

## 运行环境

| 主机 | 已验收配置 | 本轮用途 |
|---|---|---|
| `yunnao-910a` | 8 张 32 GiB 的 Ascend910ProA。驱动 24.1.0，CANN 8.0.0，Python 3.11.10，PyTorch 2.4.0，torch-npu 2.4.0.post2，Transformers 4.53.1 | Text2CAD FP32 原生评分、两个命令双塔的 CPU 对照、小规模训练运行检查 |
| `yunnao-share4` | 已运行过项目的 910C，CANN 8.3.RC1，PyTorch 2.7.1，torch-npu 2.7.1.post1，Transformers 5.23.0 | 保留原 BF16 配置，运行 HierCAD-14B 单件打分和批布局对照 |

安装环境在 `/home/ma-user/work/CADRetrieval_runtime/envs/cadretrieval-910a-cann80-torch240`。检出的代码和运行产物分别在 `worktrees`、各运行控制目录中。用户原来克隆在 `/home/ma-user/work/CADRetrieval` 的 checkout 保持原状。Text2CAD 模型、BERT 的五个文件、命令向量压缩包、v1.0 标注、官方划分和 Minimal JSON 都已放入同一 `work` 下的标准 `CADRetrieval_runtime/models` 与 `datasets` 路径。每个文件核了发布哈希。迁移后的历史见证包含 149 项经 SHA-256 核验的原件。

安装初期，服务器从公开 PyPI 取 SciPy wheel 的阶段长时间未完成。终止自己这一下载进程，使用本地取回并逐一核 SHA-256 的 22 个 ARM wheel 离线补齐。首次 CPU 回放在导入 BERT 时遇到 `einops` 缺失，来自 Transformers 的 NPU attention 集成导入路径。补齐后独立环境的 `pip check` 与 BERT 导入均通过。没有改系统驱动或 CANN。

### 精度与硬件证据

- 镜像默认环境的掩码注意力探针最大绝对差为 `9.793×10⁻⁴`。给当前进程设置 `ACL_PRECISION_MODE=must_keep_origin_dtype` 后，镜像同探针差值为 `4.768×10⁻⁷`。独立执行者在新环境第 4、5 张卡复现，FP32 掩码注意力与 CPU 最大差 `3.576×10⁻⁷`，带平滑的交叉熵最大差 `4.768×10⁻⁷`，两项退出 0。
- 首个 FP16 探针的线性层通过，但更新量为零。用实际梯度复算发现 CPU 预期更新在半精度舍入后也是零。改成独立、适当缩放的权重抽样后，新环境完成反向与真实更新。最大梯度 `0.210815`，最大权重变化 `0.00415039`，对 CPU 更新的最大差 `0.00012207`，退出 0。
- 双卡 HCCL 的两个 rank 都读到预期求和值 3.0，退出 0。BF16 能力查询退出 0，返回 `false`。这组证据分别覆盖了做过探针的卡和已测试的操作。
- Text2CAD 用完整 272 行、单配对、原始 FP32 读出回放 16 条 910C 历史记录。输入与标签哈希、文本 token、有效字段计数逐项相等，NPU 最大 S0 差 `2.8885901×10⁻⁵`，满足冻结的 `max(0.001, 10⁻⁶ × |S0|)`。同一机器的 CPU 回放最大差 `0.041714536`，超出容差。实验分数采用通过的 NPU 配置。
- 同一设置的批量 16 和尾批 14 均通过两次真实前向，60 个分数的最大差 `2.6833266×10⁻⁵`，退出 0。两形状的首次调用分别用时 308.3 秒和 231.5 秒，重复调用分别为 5.45 秒和 4.91 秒。时长来自实测小面板，每次调用都包括已固定的文本和候选处理。
- CANN 在非零 offset 投影上产生警告。用同尺寸线性层将权重切片的 storage offset 固定为 65,536，复现同一警告。切片和 offset 清零的拷贝输出逐元素相等，对 CPU 最大差 `7.7486×10⁻⁷`，探针退出 0。原模型的警告和数值回放原件都留存。

复现入口见 [运行手册](../../plans/IDEA01_910A_RUNBOOK_2026-11-23.md)。版本与精度设置分别依据 [Ascend PyTorch 兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) 和 [CANN 8.0 编译选项](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_1371.html)。

### 实际训练路径

Text2CAD 四级文字源和 CadTranslator 短句源各完成 3 次真实 FP32 AdamW 更新，进程退出 0。每步固定 4 对，全部 UID 来自已登记的官方训练划分。BERT 参数冻结，CAD 端使用两头分别掩码求均值的交叉熵、0.1 标签平滑、原配方学习率和梯度裁剪。每一步记录了非零更新量和 AdamW 内部步数，共用过 24 对次，工程探针权重未存储。固定批次后两步用时约为 T 17.5 秒、CT 4.4 秒，首步明显更慢。这是单卡、固定输入的真实见证，同源匹配重训和验证面板结果仍待补齐。[作者训练代码](https://github.com/SadilKhan/Text2CAD/blob/669a81472ede2269837c22a4d00070eaac360c82/Cad_VLM/train.py) · [原生训练损失](https://github.com/SadilKhan/Text2CAD/blob/669a81472ede2269837c22a4d00070eaac360c82/Cad_VLM/models/loss.py)

两来源的 4,096 件开发训练清单、512 条验证目标、2,048 件验证图库和 48 条背景文字已复制到 910A 的 `CADRetrieval_runtime/evaluation_inputs/in_domain_pilot_20261123`。原清单 SHA、官方 train/validation 归属和图库包含全部验证目标的条件已核验。固定 BERT 分词器上，L3 有 95/512 条超 512 token，最长 1,124 token。L0–L2 及 CT 这批验证文字全部在窗内。此步骤只核验输入，正式同源验证需使用符合训练协议的检查点。

## 评分协议

每对 CAD 有两种等价程序历史，各有两种叙述先后与两套措辞。每种展示对两条描述、两件 CAD 交叉取四个原生分，40 对共 320 种展示，每个生成器共 1,280 个主分数。主读数为固定两件 CAD，各自选对自己的描述。还记录同一句话选 CAD、四格差 $D$、并列和缺失。每对的严格联合成功需要八种展示全部正确。

题目由程序构造，几何真值来自前一轮对原始、Text2CAD 量化后以及 HierCAD 目标重放的独立 OpenCASCADE 测量。两种发布检查点是开发参照。词袋同文计数和 SpectraBRep 同几何特征的对照以已核实的输入恒等性判定，单列于实际前向的模型旁。

[固定执行方案](../../plans/IDEA01_910A_AND_DEV_GEOMETRY_RECOGNITION_2026-11-23.md)

## 40 对开发题的结果

「每种展示两件都选对句」是固定两件 CAD 选描述的双正确。「每种展示双句都选对件」固定两条描述选 CAD。最后一列以一对题为单位，要求全部八种程序历史、叙述顺序和措辞的组合都答对。

| 读出或对照 | 两件都选对句 | 双句都选对件 | $D>0$ | 八种全部答对 |
|---|---:|---:|---:|---:|
| Text2CAD 原生总分 | 20/320 | 0/320 | 162/320 | 0/40 |
| HierCAD 原生总分 | 122/320 | 76/320 | 168/320 | 2/40 |
| 命令双塔，Text2CAD 四级训练 | 0/320 | 0/320 | 163/320 | 0/40 |
| 命令双塔，CadTranslator 短句训练 | 3/320 | 7/320 | 155/320 | 0/40 |
| 首个属性与首个建模槽位匹配 | 160/320 | 160/320 | 160/320 | 0/40 |
| 忽略放置的局部草图距离 | 0/320 | 0/320 | 0/320 | 0/40 |
| 无序词袋，按文本输入恒等性 | 0/320 | 0/320 | 0/320 | 0/40 |
| SpectraBRep，按 CAD 输入恒等性 | 0/320 | 0/320 | 0/320 | 0/40 |

两种生成器的 1,280 个主分数各自齐全。四组题目沿用已独立量测的几何真值，输入身份逐格核验，双向指标独立复算。HierCAD 的两个八展示成功都在 V 形缺口的大小–部位题。全部方法的八展示标签互换成功均为 0/40。因此 HierCAD 的条件标签互换参照 $N$ 是 1/40，其他方法为 0/40。重复展示共享同一对几何和文字，观察仅限两工程家族。

Text2CAD 在四个粗孔形易题上固定 CAD 双正确为 3/4，反向为 1/4。HierCAD 对应为 4/4 和 0/4。Text2CAD 的每道易题两件相差 11 个有效字段。易题上的粗孔形信号与绑定题上的低联合成功率同时出现。槽位规则在半数单次展示里两件都答对，全部八种展示的成功仍为 0/40。四格差和单次展示各提供一部分行为证据。

Text2CAD 首段的 25 分钟时限已实到，退出 124，留下 80 个完整展示。CPU 的回放超出原容差。NPU 续跑在同精度、同布局下再次通过历史回放，脚本用时 2,653.35 秒（约 44 分 13 秒）补齐其余展示与易题，实际退出 0。80 条原展示中的 320 个分数逐格相等，最终共有 320 个唯一展示。

## HierCAD 的实测与布局对照

在 910C 上，16 条历史单件分数完全复现。40 对的 1,280 个单件主分数全部有限，另测同一小组的 32 个批量分数。总墙钟 601.9 秒，辨认与布局对照段 414.1 秒，峰值分配约 31.0 GiB，峰值保留约 43.6 GiB。抽查的 8 种布局有 1 种改变了「两件各选对句」的严格判定，逐 token 平均分差的最大绝对值为 `0.0006536381`。主表统一保留单件分数。

四个粗孔形易题上，HierCAD 固定 CAD 时均能让两件选对描述（4/4）。固定文字改为选 CAD 时，双句都选对为 0/4，两句都偏好同题的双圆孔程序。该易题区分两个圆孔与一个圆孔加一个长圆孔，后者程序多 211–260 个有效目标 token，也伴随其他几何变化。这组数据记录了原生总分的候选偏好，长度与几何效应仍混在一起。

## 版本与原件位置

| 原件 | 执行源版本 | 远端位置 |
|---|---|---|
| 910A 隔离环境计算见证 | `dcf2994126c054006444721f59e9fb024dce4b73` | 910A 的 `qualification_20261123/independent_runtime_witness` |
| Text2CAD 历史回放与首段 | `e43faf6fb5316effe2fab45d15aa1129db521e2c` | 910A 的 `qualification_20261123/project_replay_and_recognition` |
| Text2CAD 续跑与计数复算 | `0d4aa86f372c11c85b07f4d6c150f59479454700` | 910A 的 `qualification_20261123/development_finalizer` |
| 两支命令双塔的 CPU 对照 | `bc31bc840c9be4a2732165fa4a754a803f733cde` | 910A 的 `qualification_20261123/command_dual_recognition` |
| HierCAD 主辨认与实际布局对照 | `a64b0b2aa272b02eedbdf1dd47a3af3c1a0b9fb8` | share4 的 `runs/idea01_geometry_dev_hiercad_20261123` |
| HierCAD 同一组粗孔形易题 | `9f11befa3c6984c152fa063bf54fdcc43f6b65f6` | share4 的 `runs/idea01_geometry_dev_hiercad_coarse_20261123` |
| Text2CAD 训练步骤见证 | `c22f71a156e675a79990c8155f722b43a733d60d` | 910A 的 `qualification_20261123/train_optimizer_witness` |
| Text2CAD 全批与尾批检查 | `68b559ad2b5391eae3878fa6439fa5cf43bd084d` | 910A 的 `qualification_20261123/batch_shape_witness` |

表中路径均从各主机的 `/home/ma-user/work/CADRetrieval_runtime` 起算。

### 本地可复核原件

下列目录位于 `D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_yunnao_910a_qualification_20261123_control`，已按远端清单逐文件验证。重复保存的初段和续跑结果分开记录。四种学习方法的逐对结果、分母和分数原件可从相应运行目录复查。

| 证据目录 | 清单 SHA-256 | 文件数 |
|---|---|---:|
| `runtime_and_dual_evidence` | `c3011dabe38d3fec76e25e257a44db148a9edff4cdb84b8f393abd47906cef51` | 130 |
| `native_and_training_evidence` | `46478793cd2e036f985ca2798a9f916e64fe18063b8cba63206fc8971b23c7ba` | 78 |
| `audit_and_batch_evidence` | `5df2fc36e60476f67ca8ba10d050c5f25d04722c4e08bbb32ad7c11aff50ccf8` | 24 |
| `idea01_geometry_dev_hiercad_20261123_evidence` | `6fb496b6b7955c9f1b14805e164894c50a829bb988d3afc236562e2d4ffbe34e` | 21 |
| `idea01_geometry_dev_hiercad_coarse_20261123_evidence` | `18a3b4a121bba06e577ddd6332bb632c1e7fe69dfe1a6dde2ad21674b5110938` | 14 |
| `idea01_hiercad_layout_and_identity_audit_v2_20261123` | `c9b74599ccf7e43481240b7790249a9dee3a82c428db448bce832396536b53a6` | 5 |

## 剩余正式研究

同源比较还需要核实发布权重的训练曝光与选点，或按同一协议训练 T 四级与 CT 短句的生成检查点。公开训练代码与配置可以核对模型、损失和默认设置，发布权重的逐 UID 曝光及实际选点依据尚未核实。已经备齐验证面板，正式同源验证、封存家族检验以及同一提示的自由生成与辨认四格对照待做。[作者训练配置](https://github.com/SadilKhan/Text2CAD/blob/669a81472ede2269837c22a4d00070eaac360c82/Cad_VLM/config/trainer.yaml) · [论文实验方案](../../manuscript/EXPERIMENTS.md)
