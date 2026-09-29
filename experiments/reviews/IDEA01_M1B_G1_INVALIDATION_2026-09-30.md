# M1b G1 首次真实前向：结果作废与修复准入

运行：`yunnan-share2`，单卡 `npu:0`，代码 `b6a46b2d4800fe5b12cda89c2404dbb9ec05205d`，授权提交 `552478b07ddd185d359ee8f4e3dab1207778993a`，run_id `m1b_g1_20260930_0258`。固定输入 SHA `501756a7b5ce0e2dc15c08ffabf37d0c8517b3fb4eef2740aff75e4dee812db7`，本地原始副本 `experiments/runs/m1b_g1_20260930_0258/`（Git 忽略，含输入、输出、summary、budget ledger、stdout/stderr）。

## 观察与裁决

- 程序退出 0、计费 1 对，336 个目标 token；设备端 sum log-prob `-99.7238085467121`，已审 CPU 核心 `-99.7238192885667`，绝对差约 `1.07e-5`。这只说明两套**评分数学**在那次 logits 上接近。
- `stderr.log` 中 PEFT 明确警告大量 `Found missing adapter keys while loading the checkpoint`；该次模型的 CADmium LoRA 是否有效载入无法成立，**结果不可用于 CADmium 检索或吞吐外推**。
- 对 SHA 已核验的官方 adapter safetensors 只读查键：共 504 项（252 A、252 B），形如 `base_model.model.layers.0...lora_A.weight`；PEFT 0.14 警告中的模型键形如 `base_model.model.model.layers.0...lora_A.default.weight`，差一个 `model` 路径分量，adapter 名由 PEFT 插入。原 `PeftModel.from_pretrained` 没有校验缺键，故报告“成功”是代码缺陷。
- 额外发现原 CLI 的 `--budget-file` 允许调用者通过新路径重新开始 1 对额度。这违反跨 run 的批准上限。发现后立即把 `ACTIVE_TASK.md` 改回 `implementation`、预算 0，并推送撤销提交 `40cc36d`。不再使用旧授权做任何前向。

## 修复准入

本地修复须对官方 504 键做**内存中**一对一映射，不修改已核验的 checkpoint 文件；构造 PEFT 模块后校验映射键集与实际 LoRA 参数键集完全相等、形状相容、加载返回值无 LoRA 缺键或意外键。预算 ledger 只能落在本项目固定绝对路径，不能通过输出目录、checkout 或参数变更重置。先跑 CPU 测试与**无前向**的真实 base/adapter 加载诊断，确认 504 项匹配，经过新的代码 SHA 审查和授权提交后，才可对同一固定输入重新跑一次 G1。

批量测速与全量评分仍未获授权。首次 G1 的原始记录保留，不覆盖、不算有效数据点。

## 后续核验更新

上述键名映射修复经云脑 CPU、零前向实测：504 个 LoRA 参数已严格匹配，但实际载入后的 252 个 B 参数仍全部为零。进一步下载并按官方 LFS SHA 验证 1.5B、7B、14B，四个公开 adapter 的 B 张量均全零，详见 `experiments/assets/CADMIUM_LORA_CHECKPOINT_AUDIT_2026-09-30.md`。因此“修复 loader 后重跑 CADmium G1”的原建议取消；只有获得非零、来源可核验的新 checkpoint 才能恢复此臂。
