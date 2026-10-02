# Text2CAD G4：正式全量准入

2026-10-02。正式代码批准完整SHA：56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719；源码e8ef0cdd0349f255d8a299ae1b32fbf3fcdd1263与R3 CPU审查相同。新增授权文档在origin/master独立提交，运行checkout保持该批准SHA；不能将旧checkout内ACTIVE当授权来源。

## 门与实际证据

1. G2 batch16真实双卡准入：26254项独立检查/415个OBS哈希；保守ETA38.2839355小时。
2. G3完整804600对、两exit0/finalized，3219147独立检查/130本地及OBS SHA；ETA28.8517907小时。正式取较大38.28394小时（含至少1.30余量），均小于七天。不是实际完成时长保证或检索精度结论。
3. G4启动器原两生命周期与R2两完成边界均修复，Root24/fresh27针对CPU通过、七旧评分核心AST/full argv不变；唯一fresh reviewer无剩余Critical/Important。
4. Linux实际22项必要标准库CPU test、零fail/error/skip、torch_free=true，exit0。测试前后Git干净，HEAD56ac0f8，两张芯片OK/无NPU进程。实际test与controller文件SHA和批准Git blob逐字节匹配。不是Windows/mock替代。
5. 两真实树均在test finally清理前采集：deadline父PID3662290/desc3662292已gone、组empty；正常父exit0 PID3662355/desc3662357已gone、组empty。前者两父exit−15，后者父exit0但诚实记录persistent后代清理异常failed；控制器两场景均结束于各自deadline内。fixture内部批准SHA为临时测试Git身份，不是生产代码身份；外部CPU_META明确实际代码56ac0f8。

## 原始证据及已保存失败

Linux原始16文件本地/OBS run m1c_g4_posix_cpu_20261002，manifest a29c8d9819dc60422a74135c61ad555e18649b0a99af2fdd007e01b83b2f2a60；Root独立源绑定/原始状态/退出核验与运输记录9文件run m1c_g4_posix_codex_check_20261002，manifest 7e95817b6363e0a8c96a9d1118efc96d4adde94cbd461a377b3c4368c0124f5f。两边逐SHA/OBS exit0通过。R3本地2文件e56c2afbbe3bdda93e55ea92f4a64c73ba557691bd9edc63460b256bac4e375b及Kimi5文件eaf351f322f8318d787f2e2babc8514a03d0533106b7a75eb261d71d2d9eb650亦已核。

首次SSH stdin的Windows CRLF在Bash第一条set处失败，未执行任何远端操作/CPU test；失败原件保留。随后按既有SSH运输规约去CR，实际22test只执行一次，未重试评分、未改环境或源代码。父子树两场景是真实Linux；临界deadline/不可杀/权限错误案例仍标mock，不能扩大成所有故障的实测保证。

## 正式批准范围

允许按IDEA01_M1C_G4_EXECUTION_2026-10-02.md，通过已审Git控制器启动全3644×8046=29319624对，仅npu:0/1两卡、两次fresh scorer，固定FP32/full272/b16尾14/无trim-cache/无probe-token-audit-resume。共同604800秒墙钟/336双卡合计卡小时=1209600卡秒，无预热、重试或空查询。必须ACTIVE另列pilot_authorized/同一完整SHA及精确预算，并Git推送、远端fetch读取最新授权。

本审查批准运行不代表运行已开始或模型达到检索精度。启动后须真实controller/scorer PID、账本invocation、完整chunk/timing、stderr及干净Git验证；正常运行只读，不重复评分。完成后完整矩阵/实际预算/退出与归档复审，再独立CPU UID评价。
