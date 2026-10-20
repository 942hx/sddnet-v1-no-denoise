# 旧版v1无去噪 · SDDNet-v1 legacy

此分支是旧版独立实现，模型标识为 `model_revision=sddnet_v1_fixed_100e_1`。保留对应训练源码，未混入新版模型；没有独立末端去噪块或小波模块。

仓库仅包含代码、配置、依赖清单、测试和说明，不含数据、权重或实验输出。新版在“新版v1无去噪”分支，checkpoint不兼容，请勿混用。Git操作见 [GIT_GUIDE.md](GIT_GUIDE.md)。

以下为此版原始运行说明，硬件信息是当时的本机验证环境；权重、数据和运行生成的审计/日志需自行准备，未随Git上传。

---

# SDDNet-v1：MILL 100 完整 epoch 实验

本工程按根目录 `Codex_SDDNet_v1_100epochs.txt` 实现。SDDNet-v1 包含共享编码器、结构分支、退化分支、亮度等级预测、条件融合和重建解码器。没有小波或独立末端去噪块；`forward(x)` 只接收低光 RGB，不接收 GT/等级标签。

## 安装与环境检查

Windows、conda 环境 `testmodel`。保留已有且兼容的 Python / PyTorch；本机保留 Python 3.9.25，已安装官方稳定 `torch 2.5.1+cu121` / `torchvision 0.20.1+cu121`，与实际 RTX 4060 Laptop GPU 驱动兼容。实际显存为 8188 MiB，并非 16 GB。系统内存为 16 GB。

从项目根目录执行：

```bat
scripts\setup_env.bat
```

脚本使用 `conda run --no-capture-output -n testmodel`，每个步骤检查返回码。仅安装缺失依赖，不自动替换现有 torch/torchvision；存在不完整或不兼容核心环境时明确报错，由用户决定调整。完全缺失核心包时采用 [PyTorch 官方安装组合](https://pytorch.org/get-started/previous-versions/)：

```bat
conda run --no-capture-output -n testmodel python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n testmodel python -m scripts.install_dependencies
```

`requirements.txt` 不含 torch/torchvision，避免普通索引替换 CUDA 构建。`requirements-lock.txt` 来自当前环境的真实包版本，不含私有索引地址或凭据；重建时先按官方 CUDA 索引安装锁定的 torch/torchvision，再安装其余锁定包。安装器保留已有可工作的 NumPy/Pillow，实际版本以锁定文件和环境报告为准。

CUDA 检查实际执行矩阵运算和小型卷积前向/反向，并核对 `sys.executable` / `sys.prefix`。CUDA 不可用时禁止正式训练，不会自动切到 CPU 训练 100 轮。

## 数据目录与审计

```text
data/
  MILLs-DSLR/{train,validation,test}/{input,gt}/
  LOLDataset/{our485,eval15}/{low,high}/
  LOLv2/Real_captured/{Train,Test}/{Low,Normal}/
  LOLv2/Synthetic/{Train,Test}/{Low,Normal}/
```

花括号是目录示意。代码使用 pathlib，相对路径以实际项目根目录为基准，不硬编码中文绝对路径。数据按需解码，不把全套图像预加载进 RAM；不移动、修改或删除原始数据。

MILL 从 stem 最右侧 `-` 解析等级 1…10：`Scene1_a-3.png → scene_id=Scene1_a → gt/Scene1_a.png`。完整保留 `_a` / `_b`。每张低光图按明确场景名匹配 GT；LOL 按完整 stem 匹配，绝不按排序 zip。无法确定的匹配直接报错，不猜测。命名规则不同的数据应先提供经确认的显式映射，再调整审计入口，不能绕过失败项。

审计会完整解码 PNG/JPEG 等图像，检查 RGB 格式、尺寸、重复路径、缺配对、未使用 GT、MILL 跨 split 场景与重复 GT 字节，并生成含文件 SHA256 的 CSV。字段包含 `dataset,filename,split,scene_id,illum_level,input_path,gt_path,width,height,input_sha256,gt_sha256`。CSV 路径相对各自数据集根目录（LOLv2 两个子集分别以 Real_captured / Synthetic 为根）。清单自身的 SHA256 写入审计报告和 checkpoint。

任务文件明确的官方损坏排除项 `MILL/test/Test13-9.png` 被记录在排除清单中，无论源文件存在还是已经缺失均说明；不删除文件。其他问题不能静默跳过。本机有效 MILL 数据为 train 300 / validation 120 / test 79；test 第 9 级为 7 张，其余等级各 8 张。具体数目以每次真实审计为准。

LOLv1 / LOLv2 Train 仅审计，绝不进入本实验训练。

## 统一入口与正式训练

Anaconda Prompt 中从项目根目录执行：

```bat
conda activate testmodel
python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml
```

也可始终明确环境：

```bat
conda run --no-capture-output -n testmodel python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml
```

流程依次为环境检查 → 完整数据审计 → 独立冒烟检查 → 100 轮训练 → 同一验证最佳权重的四套测试 → 性能统计 → 报告。子进程全部使用 `sys.executable`，检查返回码并实时保留日志。

默认配置：patch 256、triplet batch 1（一次前向 3 张 RGB）、累积 2 步、workers 0、pin memory、AMP、AdamW、lr 1e-4、weight decay 1e-4、CosineAnnealingLR 至 1e-6、梯度裁剪 1.0。最后不足一个累积组时按实际 triplet 数归一化。记录实际 optimizer 更新次数与 AMP 溢出跳过次数。每个 epoch 全部 300 张训练低光图各作为 anchor 一次，无虚拟扩增 epoch，无 debug 截断。

Positive 为同完整场景、不同等级；negative 为不同场景、与 anchor 相同等级。Anchor/positive 共用裁剪和几何增强，negative 与其 GT 独立成对增强。不做亮度/gamma/color jitter。a/p/n 拼成一个 3B batch，三组共同参与重建与等级监督。

```text
z_s     = normalize(GAP(Fs).float(), p=2, eps=1e-8)
L_scene = mean(relu(||z_a-z_p||2 - ||z_a-z_n||2 + 0.2))
c_gt    = (illum_level-1)/9
L_illum = L1(c_hat, c_gt)
L_rec   = L1(enhanced, gt) + 0.2*(1-SSIM_RGB(enhanced, gt))
L_total = L_rec + 0.1*L_scene + 0.05*L_illum
```

SSIM 用可微 pytorch-msssim，data_range=1、win11、sigma1.5、K=(.01,.03)、nonnegative=False；SSIM、特征距离及损失归约用 FP32。这是适配独立分支的 MILL 辅助监督变体，不是原方法逐字复现；归一化等级预测不是物理 Lux 估计，也不能证明学到了噪声图或严格解耦。

逐轮完整验证 MILL validation，FP32、batch1、原尺寸、无随机裁剪或 TTA，以逐图等权平均 PSNR_RGB 选择 `best.pth`。无提前停止，scheduler 每完整 epoch 更新一次。只在正式训练前确认 OOM 后允许 256→192→128，并记录原因/最终配置；不会在训练中改变 patch 或删改结构。

## 冒烟检查、断点恢复与进程锁

可单独运行全部必要预检查：

```bat
python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml --preflight-only
python -m unittest discover -s tests -v
```

真实检查包括完整 triplet 语义、采样图、实际 `[3,3,256,256]`（或明确记录的 OOM 调整尺寸）CUDA 前向/loss/反向/optimizer step、各输出和梯度有限、两个分支非零梯度、`[1,3,257,321]` 原尺寸输出、同图 SSIM≈1/LPIPS≈0/PSNR=Infinity、scikit-image 对照、完整 checkpoint 恢复后下一步参数一致、随机状态和不足累积组归一化。冒烟在独立进程和 `smoke/` 中，不污染正式初始化、日志或轮数。

恢复累计到第 100 轮：

```bat
python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml --resume outputs/sddnet_v1_mill_100e/checkpoints/last.pth
```

checkpoint 原子写入，包含模型、optimizer、scheduler、scaler、已完成 epoch、global step、实际更新/溢出次数、best 指标、配置、数据清单哈希及 Python/NumPy/Torch/CUDA/DataLoader RNG。中断后从上一个完整 epoch 恢复；未完成 epoch 重跑，不算已经完成。恢复时验证数据清单与训练配置。相同软硬件下恢复随机流，但不强制 CUDA 位级确定性。

正式目录已有训练记录时必须明确 `--resume` 或选择新的 run_dir，不能覆盖。`run.lock` 记录 PID 和进程创建时间，拒绝同一目录重复运行。查看：

```powershell
Get-Content outputs/sddnet_v1_mill_100e/run.lock
Get-Process -Id <锁文件中的PID>
Get-Content outputs/sddnet_v1_mill_100e/train.log -Tail 30
```

确认旧进程已退出后，在原启动或恢复命令末尾加 `--clear-stale-lock`；代码仍会拒绝清理活进程的锁。不要手动反复杀进程或在已有训练时重复启动。关闭整机/终端可能中断前台任务，可靠权重保留在 last.pth。

## 训练结束后的评测

自动使用同一个 `checkpoints/best.pth`，依次测试 MILL test、LOLv1 eval15、LOLv2 Real Test、LOLv2 Synthetic Test。四套输出均记录相同 checkpoint SHA256，不能用它们选择权重。跨 LOL 测试不是在 LOL 单独训练的结果。

仅重测（不创建 optimizer、不运行训练或训练冒烟）：

```bat
python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml --eval-only --checkpoint outputs/sddnet_v1_mill_100e/checkpoints/best.pth
```

指标协议固定为 FP32、eval+inference_mode、batch1、原始 RGB `[0,1]`、整图前向、无 TTA/额外裁边/GT 曝光修正。只去掉模型内部 padding，在未量化输出上逐图计算并等权平均。原图作为 input baseline，报告增强前后差值。指标不包含 PSNR_L，PSNR_RGB 不能冒称 MILL 论文指标。

LPIPS 使用 [官方 LPIPS](https://github.com/richzhang/PerceptualSimilarity) 的预训练 AlexNet v0.1、frozen/eval、输入 `2*x-1`，只作评测，不加入训练。官方 AlexNet 和线性权重的文件哈希会记录；下载失败、计算失败均写 `NA` 和真实原因，绝不使用随机网络冒充预训练指标。权重缓存位于项目 `cache/torch/`。

MILL 按 1…10 级报告数量、input/output PSNR/SSIM/LPIPS、delta PSNR、PSNR 下降比例及归一化等级 MAE。LOL 没有同类标签，MAE 写 NA。MSE=0 的数学 PSNR 记 Infinity，不误填有限数。

性能统计固定 `[1,3,400,600]`，FP32，预热 20 次、计时 100 次、逐次 CUDA 同步。计时仅包含模型前向，排除磁盘 IO、CPU/GPU 拷贝、指标和保存；输出 mean/median/p95 时延、FPS、精确参数量、实际显存容量、峰值训练及推理 allocated 显存。MACs/FLOPs 未统计时写 NA 和原因。整图评测 OOM 会明确失败，不能悄悄 resize 或切换 tile。

## 输出

固定目录 `outputs/sddnet_v1_mill_100e/`：

- `environment_report.txt`、`environment.json`、`requirements-lock.txt`、`config_resolved.yaml`。
- `data_audit.json` / `.md`、`metadata/*.csv` 和清单 SHA256。
- `smoke/smoke_results.json`、`triplet_sampling.png`、独立恢复测试权重。
- `checkpoints/best.pth`、`last.pth`、`epoch_010.pth` 等每 10 轮权重。
- `train.log`、`history.csv`、`tensorboard/`、阶段日志、`pipeline_status.json`、`training_status.json`。
- `metrics/summary.csv`、`per_image.csv`、`mill_per_level.csv`、`model_profile.json`、LPIPS 状态/权重哈希。
- `figures/loss.png`、`validation.png`、训练后 `mill_per_level.png`。
- `enhanced/{mill,lolv1,lolv2_real,lolv2_synthetic}/` 和各数据集按稳定文件名顺序选择的前三张 input/output/GT 对照图。
- `report.md`：根据真实 checkpoint、history 和评测文件生成；训练期间也明确标记实际完成轮数、未运行的指标及原因，不把运行中报告当作完成结果。

```bat
tensorboard --logdir outputs/sddnet_v1_mill_100e/tensorboard
```

旧 `runs/smoke/` 来自先前版本，只供历史流程验证，不能作为当前固定架构/100 轮实验结果。正式入口不会加载那些权重。第二版计划独立加入末端去噪块做对比实验；本次不实现 V2，后续须保持数据划分、种子、训练轮数、验证选择和评测协议一致，并报告额外复杂度。
