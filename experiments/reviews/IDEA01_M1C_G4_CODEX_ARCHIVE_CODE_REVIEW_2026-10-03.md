# G4 Codex归档程序审核与执行批准

2026-10-03。审查范围 ff12ea325659df261c20a14121583cd29c41af50..3ffd22254831e2cad07878c835026086fd5af05f；单一fresh只读reviewer `/root/g4_archive_review`，无实现代理/重复审查席位。用户明确要求Codex接手，Kimi独占执行约束已被用户覆盖。

## 独立结论

固定目标独占mkdir、source LF manifest/路径/完整SHA/数量字节、双方真实exit/no retries、Git blob绑定符合合同。6 supplied tests通过；reviewer在Windows实际bsdtar八路toy roundtrip覆盖17 payload+manifest、含空格路径，全部字节通过；缺manifest/LF故障及目标冲突拒绝且sentinel未改。当前script 9819 bytes/LF与Git blob相等，core.autocrlf=true未造成实际差异。

唯一Important：receiver Popen失败时sender已创建，但launch.json未写，真实sender PID/双方argv遗失。无Critical/Minor。reviewer未检真实OBS/原评分数值/独立Start-Process启动/G5，理由分别是只读权限、已审核复用、尚未执行、范围外；这些由Root实际运行/既有证据/后续门负责，不视为已通过。

## 最小修复与验证

Root增加实际Python sender+不存在receiver反例，旧代码出现fault_launch.json缺失；修复在spawn前写argv、sender创建后立即保存PID、receiver创建后保存PID，终态保留两argv和实际PID/exit。针对7项测试重新全部通过（包含四种真实两端exit组合及unsafe paths子例），不重跑scorer/G1–G4无关门。Root独立system bsdtar gzip→Python流式tar读取也通过，LF payload与manifest字节不变。

完整修复提交/批准实现SHA：`f9c149fe1a8d616680d778b585f16ede2dc7a5d7`。证据checkout外runtime/g4_archive_code_review_20261003/tests_stdout.log、tests_stderr.log、tests_exit.json，真实exit0/7 tests。审查后只改这项故障记录，未改变分数/输入/环境或原始评分SHA。

## 执行批准

批准上述Git blob的归档代码执行一次：Windows独立Start-Process Hidden，最多8 gzip tar流；新OBS recovery目录独占创建，旧部分目录保留；零NPU/前向/新增模型数据。ACTUAL_HEAD允许比实现SHA多授权/计划文档，但实际运行script必须逐字节等于批准Git blob，tracked tree干净。Start-Process启动后独立核PID与持久state，确认工具会话结束后存活；不是以短期exec session当监督。

运输成功后还须实际八路双方exit0、远端3666 sha256sum真实exit0、独立OBS3666 byte/覆盖witness通过；此前 full_matrix_approved=false，G5不启动。任何真实失败保留目录和日志停下交流，不自动重试。
