# BLOCKERS — idea01_p1_conditional_sensitivity_20260925

本门（第一门补正）只有一条阻塞，影响第二门资产核验的一条路径，不影响本次代码补正与 CPU 复核的完成。

## B1（对应 Codex 审核 R3 末段）：910C OBS 挂载 `/data/data_dev` 仍有不可中断 D 状态复制进程

- **现象**：P0 补证复制进程 `cp -r .../review_supplement /data/data_dev/CADRetrieval/runs/idea01_p0_20260924_r01/`（PID **2602998**）自 P0 审核记录时处于不可中断 `D` 状态；2026-09-25 补正期间在服务器上以只读 `ps -p 2602998 -o pid,stat,wchan,cmd` 复核：**仍为 `D`（wchan `request_wait_answer`）**，挂载仍未恢复。
- **影响**：`/data/data_dev`（910C OBS）上的 P0 补证副本**无法完成 SHA 核验，不能记为已通过**；第二门若要求在该挂载做资产/结果核验，此路径当前不可用。
- **我未做的事**（按明确指示）：未在 `/data/data_dev` 上启动任何新的读取或校验进程（避免再次卡死）；未尝试 `kill -9`（D 状态不可中断，只能等挂载恢复）；未把 910C OBS 副本写成"已核验"。
- **已核验的替代证据**：P0 完整结果的三份副本——服务器运行目录原件、本地镜像、三机共享 OBS `/data/obs_data_liuza/CADRetrieval/runs/idea01_p0_20260924_r01/`（P0 审核记录 78/78 通过）——仍然有效；第二门评分读取的模型/数据**运行副本**由 `run_p1_matrix.py` preflight 在载入权重前自动做 SHA256SUMS 核验（本提交新增）。
- **建议（供 Codex 决定，非授权）**：等挂载恢复后收尾 PID 2602998 并补核 910C OBS 副本；或由 Codex 明确修订第二门资产核验路径（例如以"运行副本 preflight 核验 + 共享 OBS"为权威路径），并在 `ACTIVE_TASK.md` 写明。

无其他阻塞。
