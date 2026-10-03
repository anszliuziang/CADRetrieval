# Text2CAD 最终交付审核：通过

2026-10-03 UTC（北京时间10月4日）。全部授权研究及报告、原始存储和退出证据交付结束。固定科学口径、方法及指标不变，无新增评分、评价、模型前向或NPU操作。

## 已有科学结果

G4完整29319624对、117289444数值检查、源端/本地/OBS3666payload SHA全部通过，矩阵证书SHA `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6`。

G5一次正式CPU评价和一次独立数值审核（873364检查）通过；完整70payload24521948bytes原件OBS包manifest `0adec648178550648b36dfd6c372b9f1f3a1d08c3d6cfc373b0be9f446c09125`，实际运输、SHA、独立byte/controller退出均0。评价证书SHA `6421e331aa574a8337779982a39d812ddd340d94a8729deb5c533749684a097b`保持不变。

最终科学报告仍为CODEX_TEXT2CAD_FULL_RETRIEVAL_RESULTS_2026-10-03.md，SHA `002cf6562bac2013c1a33060834f03fe81c5e97fb073ec8324f3d6f677535491`。主表SUM R@1/5/10/20为0.201664/0.920091/1.638518/2.609024%，MedR3411.5；仅支持当前Protocol A弱登记UID排序信号，宽泛语义命题partial、收窄命题同家族fresh判定provisional yes，不宣称实用、语义、因果、SOTA或泛化。

## 最后证明包的实际失败与恢复

证明包为62payload108659bytes，manifest `b555cf47d3dad7ab465c616b2b19a3ca00d0735ac340c54dbf24c05c3bf21f1b`，OBS目录 `runs/m1c_g5_final_delivery_proof_20261003`。本地报告/certificate/完整semantic trace原字节绑定和8路sender/receiver实际0、remoteSHA0已通过。

原cvteam6独立reader在OBS元数据请求等待后，读取`project/experiments/ACTIVE_TASK.md`返回ENOENT，reader/controller实际1。完整失败stderr与原controller_exit保留在checkout外`runtime/g5_final_delivery_proof_control_20261003`，不改成成功。

只读检查显示4-A6000同共享fuse.obsfs可正常读取同一对象（6172bytes，SHA `0c69c0184f523fc5a0b8e3b11b37c1b4de02265e198d416cf33671852f96db83`），本证明包及主要G5清单SHA完全匹配。故障范围定位到cvteam6挂载读取路径，内部obsfs原因未确定；未重新挂载、删除或覆盖旧目录。

通过4-A6000仅作为存储入口，原已审核只读witness SHA `a22d785a6f66d620d31fc13446ebff659273b0c7ac9a75e576aa16ea619902d3`一次恢复执行，reader/controller实际0：全部62文件/108659bytes/SHA/coverage通过，unknown_extra=[]。未重新运输该62包，未重启评分或评价。恢复命令、输出、实际退出在`runtime/g5_final_proof_cross_mount_read_20261003`；原失败与恢复记录明确分开。

## Closing receipt

新的独占OBS目录为 `4-A6000:/data/obs_data_liuza/CADRetrieval/runs/m1c_g5_final_closing_receipt_20261003`。仅使用普通系统mkdir/tar/sha256sum上传原始元数据，未写服务器项目源码。

- `CLOSING_RECEIPT.json`：73977bytes，SHA `11dfa50b87ed01c19cfdea3fcba4e046091dc45d3f8a609a609fb1aef4e8be76`。
- LF `SHA256SUMS`：SHA `d8d307bcc47eaf6a39db51de0bf5fdb75beb145c4f667a1cecf0ed944d2c7640`。
- Receipt包含88份最后证明包原命令/两端运输退出/原失败/备用读取退出/launcher/诊断/源manifest原字节（base64）、size及SHA。
- 该receipt的prepare、sender、receiver、remoteSHA、原固定witness独立byte、controller全部实际0；核验1payload/73977bytes，无unknown extra。控制原件在`runtime/g5_final_closing_receipt_control_20261003`，本地数据包在同前缀`bundle_20261003`。

Closing receipt绑定最后证明包真实退出；其自身运输原件与最终Git记录留存，不递归制造新的归档证明任务。managed CPU worktree此前已可恢复归档，Kimi草稿、旧错误目录和所有原件保持。

唯一archive reviewer独立只读复核Ready，无Critical/Important：恢复门未弱化，88个内嵌记录解码后与原件逐字节一致、SHA/长度及控制目录覆盖一致，报告和certificate指纹相同。Root随后确认receipt上传的全部实际退出、SHA及byte关口通过；review记录在`runtime/g5_final_delivery_closing_review_20261003.json`。

ACTIVE现为reviewed、所有运输授权false、approved SHA none、零NPU/零新评价。完整授权工作结束，删除text2cad监控；后续新研究须另行授权。
