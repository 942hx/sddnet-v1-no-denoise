# SDDNet-v1 · A0 · best epoch 88

这是当前研究版本的唯一活动实现：`arch_revision=v1_clean`，A0，seed42，bounded 调制，`residual_gate=false`。无独立末端去噪、无小波。正常 forward 只接收低光 RGB；GT 和真实照明等级不进入正常推理。

模型、初始化、损失、固定 triplet 采样、同步增强及训练/指标协议保持冻结 A0 行为。活动配置只有 [configs/a0.yaml](configs/a0.yaml)。A1/A2/A3、standard、PK、半难挖掘和 residual gate 的活动实现已移除；历史混合报告仍保留为 A0 选型证据。

## 环境

本机已验证环境：`C:\software\anaconda3\envs\testmodel\python.exe`，Python 3.9.25，PyTorch 2.5.1+cu121，torchvision 0.20.1+cu121，CUDA 12.1，cuDNN 90100，RTX 4060 Laptop GPU（8 GiB）。本次未安装、升级或重建环境。

```powershell
conda activate testmodel
Set-Location -LiteralPath 'E:\结构亮度结构解耦\V1新版无去噪'
```

入口校验环境名为 `testmodel`。其他必要依赖见 [requirements.txt](requirements.txt)，Torch 约束见 [.core-constraints.txt](.core-constraints.txt)。完整原环境锁定记录仅作证据，见 [provenance/environment](results/A0_best88/provenance/environment/)。未来新环境安装方法及本轮验证边界见 [复现说明](docs/reproduction.md)。

## 权重与记录

默认权重：`results/A0_best88/checkpoints/best.pth`。

```text
SHA256 a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083
```

训练完成 100 轮；依据 MILL validation 选中第 88 轮，不能将 best88 写成只训练了 88 轮。同 run 的完整末轮状态保留为 `results/A0_best88/checkpoints/last_epoch100.pth`，并非默认推理权重。两文件保留原字节与元数据，不重新序列化；严格加载，不使用 `strict=False`。

全部 A0 原始记录统一位于 [results/A0_best88](results/A0_best88/README.md)，包括训练历史、验证、四集评测、逐图/分等级指标、照明诊断、预览、冻结与配置/环境证据。原始日志、CSV、报告及 resolved 配置保持原字节；其中历史路径通过 [path_map.csv](results/A0_best88/maintenance/path_map.csv) 查找，不能直接照搬为当前命令。

## 数据与训练

原始 `data/` 保持原样：`MILLs-DSLR/`、`LOLdataset/`、`LOLv2/`。配对、划分、官方排除项和审计位置见 [数据说明](docs/data.md)。路径相对当前项目目录解析；评测可通过 `--mill-root`、`--lolv1-root`、`--lolv2-root` 覆盖。训练根路径在唯一活动配置的 `data` 中设置，不需要修改源码。

训练只使用 MILL train/validation；不会自动执行 MILL test、LOL 或完整四集评测。100 轮、patch256、triplet microbatch1（3 图）、accumulation2（完整更新 6 图）、AdamW、cosine、AMP、尾窗与原 A0 相同。保留重建、场景 Triplet、照明 L1、空间一致性和前 5 轮 warmup，具体数值以配置和原始 contract 为准。

以下是可用接口；本次仅验证训练帮助、参数/路径保护及恢复逻辑，没有运行正式训练或续训。每次新输出路径必须尚不存在：

```powershell
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new
# 仅恢复同一未完成 run 的完整 last；不要对冻结结果原地续训。
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new --resume outputs/A0_seed42_new/checkpoints/last.pth
```

已经完成 100 轮的 `last_epoch100.pth` 仅作完整复现证据；CLI 会拒绝继续第 101 轮。训练后只导出本 run 的 best validation。

## 评估与推理

```powershell
# 只读发布检查：SHA、strict load、CPU 小尺寸无梯度 forward
python -m scripts.verify_release --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth
# 本轮实际验证的范围：MILL 1 图小规模评测；完整结果不能由它替代。
python -m scripts.evaluate --config configs/a0.yaml --dataset mill --device cpu --max-images 1 --output outputs/MILL_one_image_new
# 完整评测接口，本轮未执行；必须由用户显式运行。
python -m scripts.evaluate --config configs/a0.yaml --dataset mill --output outputs/MILL_test_new
python -m scripts.evaluate --config configs/a0.yaml --dataset lolv1 --output outputs/LOLv1_test_new
python -m scripts.evaluate --config configs/a0.yaml --dataset lolv2_real --output outputs/LOLv2_real_test_new
python -m scripts.evaluate --config configs/a0.yaml --dataset lolv2_synthetic --output outputs/LOLv2_synthetic_test_new
python -m scripts.evaluate --config configs/a0.yaml --dataset all --output outputs/four_tests_new
# 单图：已在本轮以此输入实际验证；文件夹接口同样已用两张合成图验证。
python -m scripts.infer --config configs/a0.yaml --input "data/LOLdataset/eval15/low/1.png" --output "outputs/单图 推理_new" --device cpu
python -m scripts.infer --config configs/a0.yaml --input "data/LOLdataset/eval15/low" --output "outputs/文件夹 推理_new" --device cpu
```

评测和推理默认严格加载上述 best88，允许显式 `--checkpoint`，但仍须满足同一 A0 contract/数据指纹。输出只能在当前项目内部新目录，不能指向 `data/`、`cache/`、`results/` 或 Git 元数据，拒绝 `..`、符号链接/junction 和已有目录。中文及空格路径已实测。

保持 FP32、batch1、原尺寸整图 RGB；模型内部保留 raw 输出，评估外层 clamp 到 `[0,1]`，量化前计算指标。无 GT 均值对齐、额外 crop、resize、tile 或 TTA。PSNR_RGB 是逐图 RGB MSE 的 PSNR 后取均值；SSIM 为 pytorch-msssim 1.0.0（win11、sigma1.5、data_range1、K=(.01,.03)、nonnegative=False）；LPIPS_Alex v0.1 使用冻结官方权重和 `2*x-1`。已有 `cache/torch/checkpoints/alexnet-owt-7be5be79.pth` 必须本地保留，LPIPS 不可用时记录 NA 及原因。CUDA TF32 设置沿用原运行环境默认值（cuDNN true、matmul false），CPU 逐值等价不能视作跨设备逐值保证。

## 已冻结的结果与局限

下表取自保留的原始验证和 [四集 summary.csv](results/A0_best88/evaluation/frozen_review/metrics/summary.csv)，不是本轮重新计算的完整结果。

| Split | N | PSNR_RGB | SSIM_RGB | LPIPS_Alex |
|---|---:|---:|---:|---:|
| MILL validation | 120 | 33.16664658589749 | 0.9364022394021352 | NA（该验证表未评测） |
| MILL test | 79 | 17.5628851837581 | 0.7783799616596366 | 0.11323790759131123 |
| LOLv1 test | 15 | 16.338013063035216 | 0.7100635727246603 | 0.31309984227021537 |
| LOLv2 Real test | 100 | 20.23149322186524 | 0.7679075080156327 | 0.2760451349616051 |
| LOLv2 Synthetic test | 100 | 14.904456186727648 | 0.6492352271080017 | 0.3283244425151497 |

MILL test 输入 PSNR 为 **21.718134396878007**；增强相对输入平均**下降 4.155249213119908 dB**，约 74.68% 图像下降。不能据 validation 分数推断测试普适提升。A0 优于 A1 仅依据当前单 seed 验证对照；既有 LOL 测试结果已经被查看，不是未触碰的盲测。条件干预和 oracle 仅为诊断，不能证明物理解耦、因果独立或 SOTA。没有提供经授权的论文、作者或许可证信息，发布许可待确定。

## 本轮验收与 Git 状态

旧/新 CPU FP32 的 raw enhanced、Fs、Fd、c_hat 在 8 个输入上逐值一致，覆盖偶数、非 8 倍数、1×1、400×600、真实 MILL 和三个 LOL 子集；loss、初始化/RNG、采样及增强均一致。完整结果与本次最终自动测试见 [release_verification.json](results/A0_best88/maintenance/release_verification.json) 和 [整理报告](results/A0_best88/maintenance/cleanup_report.md)。未进行正式训练、完整四集重测或 CUDA 旧/新逐值对照。

本目录属于父仓库 `E:/结构亮度结构解耦`，Git 元数据在父级 `.git`。清理阶段曾按边界暂停 Git，后续已获得用户对共享 Git 元数据的明确授权。发布目标为现有 origin `https://github.com/942hx/sddnet-v1-no-denoise.git` 的 `新版v1无去噪` 分支，仅提交本目录审核过的文件；父级和兄弟目录的既有修改不属于本次提交。发布清单与同步核验说明见 [git_sync.md](results/A0_best88/maintenance/git_sync.md)。原清理验收 JSON/报告中的暂停状态保留为授权前历史记录。

发布清单包括源码、唯一配置、测试、依赖、文档及审核后的原始 CSV/JSON/YAML/轻量日志等文本；`.gitattributes` 对证据禁用换行归一化。原始 data、两个完整 checkpoint、LPIPS 缓存、TensorBoard 事件、所有图片以及约 10 MB 的原始 `training/A0_seed42/train.log` 只保留本地。轻量训练历史和逐轮记录仍随文本提供。本次不使用 LFS 上传权重，不提供或虚构下载地址；clone 不会自动得到本地专用材料。
