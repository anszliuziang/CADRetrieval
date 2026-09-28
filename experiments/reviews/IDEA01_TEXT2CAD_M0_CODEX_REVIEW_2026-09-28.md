# Idea 01 / Text2CAD M0：Codex 代码与数据审查

审查提交：`b440f2d6447ac6b088ea0168ab1a84629810ffee`。范围仅为本地 CPU 的 query/gallery/qrels 清单和 UID 交叉审计；**没有模型前向、GPU/NPU 评分或检索精度**。当前 M1 评分器、远端运行 SHA 和 pilot 预算均未批准。

## 实现来源与审查处置

GLM-5.2 通过本地 Claude Code 编写 M0 初稿与合成测试。Codex 首次运行得到 5 通过、4 失败；GLM 修正后 9 项通过，但真实数据揭示其未对 L0 去首尾空白，并且初稿把正例 UID 写进 query、把配对摘要存在性写进 gallery。Kimi-K3 的窄补修调用在读取文件后持续无响应，未写入代码。Codex 据 [冻结协议](../plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md)修复这些问题、补隔离断言并编写独立的训练 UID 交叉审计。上述过程不是对初稿的自动放行。

## 已核证事实

固定输入及官方 SHA256 见 [资产审计](../assets/TEXT2CAD_BENCHMARK_ASSET_AUDIT_2026-09-28.md)。使用 `.cache/hf_venv/Scripts/python.exe` 运行 `experiments/tests/test_m0_manifest.py`：**9 passed**。随后用 `experiments.src.idea01.m0_manifest` 对五个官方固定输入运行一次，输出在 Git 忽略的 `.cache/m0_real_run_v3/`，退出码 0，所有预期检查通过。

| M0 结果 | 数值 |
|---|---:|
| 官方 train / validation / test UID | 159,049 / 8,925 / 8,046 |
| v1.0 CSV 中 test UID / 非空 L0 query / qrels | 7,946 / 7,934 / 7,934 |
| 缺 CSV UID / 空 L0 | 100 / 12 |
| `str.strip()` 后不同 L0 / 重复组 / 最大组 | 3,644 / 639 / 371 |
| 单正例 UID 的 R@1/2/5/10/20 理论上限 | 45.93 / 53.98 / 64.31 / 72.13 / 80.40% |
| `cad_seq` / minimal JSON / CADmium 原生 JSON 的 test 覆盖 | 8,046 / 8,046 / 8,046 |

未去空白的原始字符串有 4,010 种、693 个重复组、最大组 320、R@1 上限 50.54%；差异只来自首尾空白。主协议仅 `strip()`，不改变大小写和内部空白。当前 `query.jsonl` 字段为 `query_id,text,text_sha256,n_chars,level`；`gallery.jsonl` 字段为 `cad_id,cad_seq_member,minimal_json_member,cadmium_native_present,n_representations`；配对 UID 只存在 `qrels.jsonl`，重复组大小只在评估用 `audit.json`。后续评分器必须只把 query 的 `text` 和当前 gallery 候选表示传进模型，不得读取 audit、qrels 或原始 CSV。固定 query ID 顺序经稳定哈希打散，与 gallery 的 UID 排序不直接对应；这属于软件输入隔离，不能替代后续评分器的代码审查。

冻结输出 SHA256：`query.jsonl` `a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd`；`gallery.jsonl` `09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30`；`qrels.jsonl` `f0677928ae2412ca4bcd8952c95d89f0568c5bb214c7bb53e82e0059555c87d6`；`audit.json` `7ee759780ba884465f37776a2c03a0395eca1ac467f9a6821663b60510728041`。文件本体为受许可数据派生物，不进入 Git。

`m0_leakage.py` 对官方 split、CADmium-ds 三份 parquet 与既有 Text-to-CadQuery P0 UID 清单复核，完整结果在 `.cache/m0_real_run_v3/leakage.json`，SHA256 `5c2f6f88c0c11a98ab3f6992aed6fdf81bc3c82744264478d408ae873667776b`，`integrity_issues=[]`。CADmium train/validation/test 分别为 118,299/8,925/8,046 UID，均落在官方对应 split，彼此零交叉。Text-to-CadQuery train 的 99,236 UID 与官方 test 交叉 **4,581**；其 validation/test 与官方 test 分别交叉 259/247。排除其训练 UID 后，官方 test 剩 3,465 CAD、其中 3,416 有 v1.0 L0 query；若同时排除其 validation UID，剩 3,206 CAD、3,158 query。未做家族级近重复排查，以上仅为精确 UID 层面的无泄漏条件。

## 裁决与下一门

**M0 清单与精确 UID 审计可用于后续评分器开发。** 这不是 IMMA 的已发表 8,023 项过滤协议，也不是一对多语义 qrels。CADmium 权重的实际训练配置、各表示的有效目标、原版 Text2CAD 双头 logits/EOS/mask，以及评分器是否真正隔离 qrels 仍须 M1 审核。`ACTIVE_TASK.md` 维持 GPU/NPU 预算 0、`approved_sha` 无；不得据此启动模型评分或正式检索。
