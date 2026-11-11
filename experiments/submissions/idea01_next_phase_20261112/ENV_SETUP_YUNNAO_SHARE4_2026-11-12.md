# T-A 交付：yunnao-share4 环境准备与资产就位（2026-11-12 夜间）

执行者：GLM 5.3（2026-11-11 接手治理下自授权的环境准备任务；**零 NPU 前向、零评分**）。机器状态、命令、失败与修复全部如实记录。

## 1. 只读侦察结果

| 项 | 结果 |
|---|---|
| NPU | 4× Ascend 910C（npu-smi 25.5.1，chip 4–7），全部 OK，**侦察时无任何运行进程** |
| CPU/内存 | 320 核 / 1005 GB |
| 工作盘 | `/home/ma-user/work` 4.0 TB（已用仅 851 MB） |
| CANN | 8.3.RC1（V100R001C23SPC001B235） |
| conda 基础环境 | `PyTorch-2.7.1`、`python-3.11.10`（与 share2 同源镜像） |
| Git checkout | `/home/ma-user/work/CADRetrieval`，干净，HEAD `80f53480…`（交接提交），origin=GitHub 镜像 |
| 网络 | 走代理可达 huggingface.co / hf-mirror.com / github.com |
| OBS 挂载 | **无**（无 fuse.obsfs/nfs）——资产须自备，不能依赖共享 OBS |

机器日期 2026-11-11 16:19 UTC（北京时间 11-12 00:19）。

## 2. Python 环境就位

从基础环境 `PyTorch-2.7.1` 克隆项目隔离环境（checkout 外）：

```
/home/ma-user/work/envs/cadretrieval-glm-torch271
python 3.11.10 / torch 2.7.1+cpu / torch_npu 2.7.1.post1
torch.npu.device_count() == 4（零前向设备核验通过）
transformers 4.53.1 / numpy 1.26.4
```

版本与 share2 历史 env 完全同版。**观察记录**：侦察时基础 env 首次 `import torch_npu` 打印过一条 `ERR99999 UNKNOWN application exception`；克隆 env 中不复现，4 卡可见，根因未追查（仅记录，不影响使用）。

## 3. 资产传输与核验（全部 SHA 逐项通过）

来源：本地已核验的 SHA 固定副本（`.cache/text2cad_official`、`.cache/bert_large_uncased`、`.cache/idea01_assets`、`.cache/m0_real_run_v3`、runtime 的 pair_manifest_v4*）；传输脚本 `CADRetrievalIdea1_runtime/glm_share4_upload_20261112.sh`，落位与 share2 同构。实测带宽约 25–35 MB/s，约 4.6 GB 用时约 3 分钟。

| 资产 | 位置（`/home/ma-user/work/CADRetrieval_runtime/`） | 核验 |
|---|---|---|
| Text2CAD v1.0 权重 | `models/Text2CAD-v1.0/422d4c93…/Text2CAD_1.0.pth` | sha `235b137e…` ✓ |
| BERT large uncased | `models/bert-large-uncased/6da4b6a2…/`（5 文件） | 5/5 `sha256sum -c` ✓ |
| Text2CAD 数据 | `datasets/Text2CAD-v1.0/422d4c93…/`：`cad_seq.zip`（`9f3a8bb8…`✓）、`train_test_val.json`（`4ea6db90…`✓）、`text2cad_v1.0.csv`（`32427ca7…`✓）、`minimal_json_0000_0099.zip`（`7e9a29ea…`✓） | 全部 ✓ |
| IEEE-T-ASE（CadQuery 线） | `datasets/IEEE-T-ASE/4091ce30…/` | 自带 SHA256SUMS 6/6 ✓ |
| 评价输入 | `evaluation_inputs/m0_v3/`（query `a0d8504f…`✓、gallery `09782bd6…`✓、qrels/audit/leakage）+ `pair_manifest_v4/` + `pair_manifest_v4_worker/` | pins ✓ |

**传输事故（如实记录，均已修复）**：① bert `model.safetensors`（1.34 GB）首次 scp `Connection closed` 中断，重传成功并核验；② minimal_json zip 首次失败为脚本 mkdir 只建了 `misc` 未建 `misc/minimal_json`，补建后成功核验。

**未传输（记录决定与理由）**：`qwen2.5-3B-SFT`（5.8 GB，仅 CadQuery P0/P1 线需要，按需再传）；Text2CAD 官方源码镜像（Git 里已有 pinned SHA 记录 `669a8147…`，运行前如需克隆再取）。

## 4. 当前限制与下一步

- 分级纪律不变：share4 的 origin 是 GitHub 镜像，可用于代码同步/阅读；**授权评分仍校验 CNB 可信远端**，迁移执行远端须先本地审核提交。
- 本机**无 OBS 挂载**：未来运行的重要产物要回传本地+共享 OBS 须经 cvteam 入口（连通性未重测）。
- 尚未做：评分 runner 的零卡预检与首批真实前向（须随 T1/B 阶段的正式计划另行授权）。
- 环境指纹：`experiments/configs/idea01_yunnao_share4_env.json`（本次写入）。

## 5. 结论

share4 已具备：4 卡空闲 910C + 同版项目 Python 环境 + Text2CAD 主线全部资产（权重/BERT/数据/图库/评价输入）+ CadQuery 线数据集。任何真实模型前向仍为 0，等待下一步正式计划授权。
