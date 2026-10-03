# G5 完整结果与证据归档批准

Git实现SHA `3e2c9701d4c990ed21fb3bd55b123eb8f1939095`；实际脚本 `experiments/scripts/archive_text2cad_g5_results_20261003.py` byte SHA `d0f745db9cc729acb457f3a2026a772f1ae8b317bfc2758eec35734cdca5a299`。只授权一次既有结果资产归档，不授权再次评价或评分。

唯一 archive reviewer 首审原 prototype `2255c742b3ee900bb0e9dd13e12e18dc849209ea34d19e63737bd89f1601a356` 找到三个 Important：receiver启动失败丢sender真实退出记录；顶层文件检查未拒绝嵌套unknown；副本未重新绑定原input pins。均在任何真实G5运输前修复，原审查记录保留。统一finally先回收自身所有已spawn子进程再落真实退出；源与复制后原manifest/所有SHA严格检查；副本input与执行module/config raw字节pin及批准Git canonical bytes相连。CRLF仅用于Git canonical比较，实际执行原字节另存，不冒称两种hash相等。

Root 8 项针对CPU见证全部通过，包括receiver-missing、sender/receiver post-spawn写证据故障、peer非零、正常退出、nested extra、复制后损坏；在测试cleanup前观察已创建进程均不存活，实际退出记录正确。证据在runtime `g5_archive_boundary_tests_20261003.json`、测试源码在Git `experiments/tests/test_g5_archive_operator_boundaries_20261003.py`。唯一reviewer复审Ready，无Critical/Important/Minor。未真实传输/改原资产/算新精度。

新独占OBS目标 `cvteam6:/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g5_cpu_20261003`，本地control `g5_complete_archive_control_20261003`，bundle `g5_complete_archive_bundle_20261003`。范围为已完成formal15payload、independent audit12payload、固定八输入、批准执行代码、review/toy证据与协议；不递归父runtime、不重新运输G4矩阵、不混CPUfixtures。显式预建所有父目录；真实两端退出0、完整remote SHA0、独立只读Py3.8 byte/覆盖/无extra全部通过后才最终交付批准。
