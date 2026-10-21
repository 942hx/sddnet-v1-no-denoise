# 复现和验收边界

唯一活动配置为 `configs/a0.yaml`。`models/` 保留 A0 模块名和参数名，`datasets/`、`losses/`、`engine/`、`utils/` 提供共享实现；`scripts/` 只有薄入口和原有数据审计工具，没有复制训练循环。运行命令见 [README](../README.md)。

## 环境和安装（仅供未来新环境）

本轮使用原有 testmodel，没有执行以下安装命令。若将来创建新的环境，须由用户自行决定并验证 GPU 驱动和官方包可用性；不要在现有 testmodel 上重装：

```powershell
conda create -n testmodel python=3.9.25
conda activate testmodel
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -c .core-constraints.txt
```

完整原环境锁定仅保留于 `results/A0_best88/provenance/environment/requirements-lock.txt`，不是安装清单。LPIPS 自带的 Alex 线性层位于当前 lpips 包中，官方 AlexNet backbone 在项目 `cache/torch/checkpoints/`。这两份权重的来源/hash 在原始 `lpips_status.json` 中记录；不得换成随机权重掩盖依赖失败。

## 训练与续训

新训练要求新输出目录和 CUDA；仅运行 MILL train/validation。100 轮与 best88 是不同概念。原始 resolved 配置保留旧路径；新的配置仅改变可写输出位置，训练 contract hash 仍为 `94d4b888a62d6758132f8b69d940d880479b3a6bb4a96f838b9a4ddb2710c9fd`，architecture hash 为 `3e6a458a9af85543d55e8662052d7f5f0227e0c144066f91a00a712ed3293bcc`。

续训只允许同 run 的完整 `checkpoints/last.pth`，需要 optimizer、scheduler、AMP scaler、RNG、sampler、loader generator 及 manifest/contract 完整匹配，且在完整 epoch 边界；拒绝 best、smoke、冻结目录和已完成 100 轮的状态。原始 `last_epoch100.pth` 已在本轮真实 CUDA Trainer 中恢复并核对状态，执行 optimizer update 数为 0；没有创建第 101 轮。

## 本次小规模验证

在删除旧实现之前用旧代码+原始 best88 生成临时 golden，与 A0-only 实现比较 raw enhanced/Fs/Fd/c_hat。8 例包括 256×256、255×257、1×1、400×600，以及真实 MILL、LOLv1、LOLv2 Real、LOLv2 Synthetic 输入。全部 CPU FP32 逐值一致，绝对/相对误差均 0；state_dict 148 项、12,270,696 参数保持，strict load、初始化/RNG、loss 和采样增强等价通过。

实际运行 CLI help、只读 verify_release、MILL 1 图 CPU 评测、真实单图推理、含中文空格路径的两张合成图文件夹推理。测试仅在项目临时工作区创建合成夹具/临时状态，没有覆盖原始实验输出；验收成功后临时工作区删除。自动测试含临时模型的少量更新、梯度/AMP/恢复/日志保护检查，不等于正式训练。

最终测试数量、失败修复情况、数据全量核对、Git 边界和未验证项以 [release_verification.json](../results/A0_best88/maintenance/release_verification.json) 为准。CUDA AMP 单元测试和 last 状态恢复与 CPU 数值等价分别记录；没有做 CUDA 旧/新逐值对照，没有完整 100 轮复训、全量验证或四集重测，不能据本轮测试新增性能结论。

可自行运行现有测试（不启动正式训练；会在指定项目临时目录生成测试夹具，运行后应清理）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest tests -q -p no:cacheprovider --basetemp .cleanup_work/pytest_manual
```

Windows junction 保护测试需要操作系统允许在项目临时目录创建 junction；权限不足应报告测试失败原因，不能静默跳过并称通过。本轮已实际运行该测试。若 pytest 环境没有配置 CUDA，CUDA 用例会按测试逻辑跳过，并需在汇报中注明。
