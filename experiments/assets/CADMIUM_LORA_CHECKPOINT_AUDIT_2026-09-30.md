# CADmium 官方四种尺寸 adapter 的有效性核验

## 结论

截至 2026-09-30，从 Hugging Face 官方 `chandar-lab` 仓库当前 revision 下载并按官方 LFS SHA256 复核的 **1.5B、3B、7B、14B 四份 LoRA adapter 中，所有 `lora_B` 张量均为全零**。四份文件均只含 LoRA A/B 张量，没有其它可产生非零增量的权重。因此其标准 LoRA 更新 `ΔW = (α/r) B A` 恒为零；这些已发布文件无法用作训练后的 CADmium 生成评分器。此结论针对下表的精确 revision/文件，不推断作者私有 checkpoint 或未来修订。

| 模型 | 仓库 revision | 官方 LFS SHA256 = 本地实测 SHA256 | A / B 张量数 | 非零 B 张量数 |
| --- | --- | --- | ---: | ---: |
| `chandar-lab/CADmium-1.5B` | `c33af3cacb2e68ca71b681986332c39ba18cdb4c` | `5ceac7667a3bd00f631754f07650af004095f460b1dd28eea6d77f4079f85592` | 196 / 196 | 0 |
| `chandar-lab/CADmium-3B` | `6c6a79d160595207804171be56a037883b369abd` | `531090f702d6b4b0680821b524c95f61cf5d208ee035b23caeabc2e8ec20634b` | 252 / 252 | 0 |
| `chandar-lab/CADmium-7B` | `b012a0bd61caaa0405d2de7c66e59296a6da7442` | `0a6300632a756681f9c0a74ea6f8a3a7583243444a1aaa0807a9950c7ffb5ec8` | 196 / 196 | 0 |
| `chandar-lab/CADmium-14B` | `3440df41a273fba4edca3eca97cae02a2e8ff364` | `191a0649894ebf957d82fa3329d267238ced98b8a7179c380c52fed4ba698a9f` | 336 / 336 | 0 |

官方 revision 与 LFS SHA 来自 `HfApi.repo_info(..., files_metadata=True)`；3B 仓库历史显示当前 `6c6a79d` 只在前一权重提交后上传 tokenizer，权重 LFS SHA 未换。可独立复跑 `experiments/scripts/audit_lora_safetensors.py <adapter_model.safetensors>`；该脚本流式计算整文件 SHA，并逐一检查 `lora_B` 张量的非零元素数。原文件都在本地 Git 忽略的 `.cache/cadmium_{1p5,7b,14b}_model/`、`.cache/cadmium_model/`；权重不入 Git。

官方仓库：[1.5B](https://huggingface.co/chandar-lab/CADmium-1.5B)、[3B](https://huggingface.co/chandar-lab/CADmium-3B)、[7B](https://huggingface.co/chandar-lab/CADmium-7B)、[14B](https://huggingface.co/chandar-lab/CADmium-14B)。在线文件列表本身不显示“B 全零”；上表这一点来自下载后逐张量的本地检查。

## 首次 G1 的独立证据

在首次 3B 单对前向，旧 PEFT 0.14 loader 因路径差一层 `model`，报告 504 个 adapter keys 未加载；同次设备/CPU 数值一致只说明打分实现一致，结果已作废，见 `experiments/reviews/IDEA01_M1B_G1_INVALIDATION_2026-09-30.md`。本地修复在内存中严格映射这 504 键。用云脑**CPU、零前向**加载实测 14/14 资产 SHA、504 个 LoRA 参数均匹配，但 252/252 个实际加载后的 B 参数仍全零。由此排除“只是 loader 错误造成全零”的解释；问题在官方源文件自身。

新 runner 增加零 B 检查，当前这些 checkpoint 将在任何前向前被拒绝。`ACTIVE_TASK.md` NPU 预算保持 0；不对 3B 继续 batch 测速，不把其 base Qwen 分数标为 CADmium。若将来作者修复文件，须重新锁定 revision、SHA、键名与非零 B，并重做 G1；不能在旧 run 上续算。

## 后续分支

Text-to-CadQuery Qwen2.5-3B-SFT 曾通过真实模型短输入前向，但其训练 UID 与官方 Text2CAD test 有 4,581 项交叉，只能在预先固定的无交叉子集 C（或原生 held-out benchmark）声称 held-out；不能拿其官方 A 全集分数与 IMMA 同榜。原版 Text2CAD 生成模型可覆盖 A，但多头、label smoothing 与 910 适配尚待证明。选择下一仪器后，仍须按独立审查 SHA → 小规模正确性 → 代表性 batch/two-card 吞吐 → 加 30% 余量核算七天上限的门槛推进。
