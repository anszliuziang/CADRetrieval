# G5 一次正式 CPU 评价启动

2026-10-03T15:07:01Z，本地隐藏 operator PID6612，真实评价子 PID7240。此前新 managed worktree `C:/Users/liuza/.codex/worktrees/text2cad-g5-cpu/CADRetrievalIdea1` 确认为干净 detached `e2d823aa284c800906ed37c0a2d0272e052de96b`；独立 CPU 授权 ref `4ad109025ea28557b47f8f467bfd8962882ccf34`。code/auth/最终证书准入检查 exit0，无评价。

实际评价唯一 argv 在 `CADRetrievalIdea1_runtime/g5_cpu_execution_control_20261003/child_launch.json`，脚本 `experiments/scripts/run_text2cad_g5_cpu.py`，data-root 原repo、runtime-root外置、output `experiments/runs/idea01_text2cad_g5_cpu_20261003`，approved-sha/auth-ref均完整。使用既有 Python3.13.5/NumPy2.3.4，不安装包；PYTHONDONTWRITEBYTECODE=1、cache/tmp外置，0NPU/0模型评分。

startup确认源码module SHA `0fdce6ae27794d1f0756026cdfd69dbd4448f7df085666ac7b9a1c1659e1e0c8`，config `f75356ae738f40faf756747e6c61aec5ad38d6696cfda34a47dcf04612737b77`，matrix certificate `afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6`。正式输入及每chunk pin均由已审CLI检查。

运行健康时仅读 progress、stderr、实际PID和exit，不再次排序/评分、不看test调方法。operator捕获真正子退出，复制完整日志/命令/退出/postGit并生成LF SHA256SUMS；完成后独立raw正例rank/指标和本地OBS归档。启动成功不表示完成或检索精度。
