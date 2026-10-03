# 真实 OBS 接收双条件诊断审核

2026-10-03约17:20北京时间。结论：prepared真实小型接收通过，可执行同审核代码的完整v2归档；完整矩阵OBS审核尚未通过、没有检索指标。

实际代码`8bc4eeabea57f416982a286add9b65ba35f57ed1`，授权ref`f86d15b3bb262944ebaf3542c3a720d94d3827a8`，controller20928/exit0，script SHA b9d93d2ce4db319212be1a8456c9c7629a63abb61f07bc5ad3ba9bfb0075073a。新入口cvteam6，仅共享OBS存储，零NPU。旧cvteam14首次SSH banner timeout没有任何远端payload写入；新两条件均真实执行一次。

## 实际对照

| 条件 | 8路发送exit | 8路接收exit | payload完整性 |
| --- | --- | --- | --- |
| implicit父目录 | 全0 | 0,0,2,0,0,2,2,2 | 16项只10项存在，6项缺失；存在的文件SHA无错 |
| prepared父目录 | 全0 | 全0 | 16项/1048576字节全部SHA通过，无缺失/未知文件 |

implicit四路stderr明确ENOENT，不是SSH警告；失败具体路径为card0/file02、05、06、07.bin，card1/file13、14.bin。由于pipeline退出非零，未把implicit称SHA通过。prepared所有父目录在启动任何receiver前显式mkdir且跨SSH test -d，所有pipeline双方0、远端sha256sum -c真实0。

fixture清单SHA81bab0445167616c59238aaa6ba4267bae4cc770531cddf098c33c98419a6804。Root独立只读Py3.8 witness按本地原件比较远端逐文件SHA/size/17文件覆盖（16payload+manifest），prepared全通过/exit0；implicit只11文件（10payload+manifest），6缺失，无未知extra/错误bytes。独立验证不通过transport布尔值推断完整。

全部原始证据：checkout外`CADRetrievalIdea1_runtime/g4_obs_receiver_diag_v2_20261003/`：state/controller_exit、两condition的prepare/stream argv/PID/stdout/stderr/exit/transport_result、remote_sha256 exit、独立witness/command/stdout/exit；OBS两个toy原件在`runs/m1c_g4_obs_receiver_diag_v2_20261003/`。失败及成功都保留。

## 原因与适用范围

在实际fuse.obsfs上，用相同8路transport和相同文件集合复现了新父目录隐式创建时的ENOENT；显式预建父目录消除了该次对照中的失败。这足以定位并修复本归档的首次目录创建并发问题。尚未追踪obsfs/GNU tar内部竞态，不能声称证明所有挂载故障或每次并发都会失败；不以本小样例替代完整原件SHA。

下一阶段同代码、明确full模式，在新的独占`idea01_text2cad_g4_full_20261002_recovery_v2_20261003`目录运输既有3666payload/14858752230字节。全部双方退出0、完整OBS独立3666 SHA/coverage与旧117289444项原始数值审核连接才批准matrix/G5；不重评分、改指标或占卡。
