# SDDNet-v1：A0 best88 活动工程

当前只保留与 A0 best88 匹配的无去噪活动实现，`arch_revision=v1_clean`、seed42。研究扩展已结束，SG、AUX、恒等训练和 Stage6 不再是活动功能。完整事实、负结果和缺失项保存在本地 `PROJECT_HANDOFF.md` 与 `results/research_history/EXPERIMENT_HISTORY.md`；本次代码发布不包含这些实验档案。

模型输入仅为 `[B,3,H,W]` 浮点 RGB。共享编码器为 32/64/128/256 通道，结构/退化分支各 128 通道；两路平滑亮度先验只来自输入图像。瓶颈及三处 skip 使用原 bounded 条件调制，预测 `c_hat` 和 `Fd` 正常参与重建与反向。输出 `enhanced` 是未经 clamp 的残差 RGB；`return_latents=True` 另返回 `Fs/Fd/c_hat`。不存在末端去噪、小波、residual gate、standard FiLM、SG 或常数标量的活动执行分支。

## 唯一默认权重

`results/A0_best88/checkpoints/best.pth`

```text
SHA256 a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083
```

原 run 使用官方 MILL train 训练 **100 个完整 epoch**，由官方 validation 选出第 88 轮。best88 不是只训练了 88 轮，不是 G0 best37 或三折某 fold 的权重。同 run 的 `last_epoch100.pth` 保留用于溯源，不是默认权重；已完成 100 轮，不能继续第 101 轮。加载必须 strict。整个 [A0 原始记录目录](results/A0_best88/README.md) 受保护。

非 A0 checkpoint 在证据迁移和验收后删除。保留的历史指标可复核，但不能再直接用这些已删除模型推理或续训，也没有保留另一套旧实验工程。

## 运行环境与依赖

使用已有 `C:\software\anaconda3\envs\testmodel\python.exe`。本机 Python 3.9.25、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121，原设备为 RTX 4060 Laptop GPU。A0 直接运行/验收依赖见 [requirements.txt](requirements.txt)，Torch 约束见 [.core-constraints.txt](.core-constraints.txt)。Stage6 曾补装的包继续留在环境中；本次没有环境安装、卸载或版本变更。

```powershell
conda activate testmodel
Set-Location -LiteralPath 'E:\结构亮度结构解耦\V1新版无去噪'
python -m scripts.verify_release
```

本次代码同步不安装或修改环境。未来在独立环境安装时，先按 `.core-constraints.txt` 使用匹配的官方 Torch/CUDA 版本，再安装 `requirements.txt`；原完整环境锁仅作为本地溯源，不作为批量重装清单。

## 唯一训练协议与接口

[configs/a0.yaml](configs/a0.yaml) 保持原字节。官方 train/validation、100 轮、patch256、triplet batch1（低光三图）、累积2、AdamW 1e-4、cosine 到1e-6、梯度裁剪1.0、原 AMP 门禁均保持。固定三元组及同步裁剪/几何增强不变，不接受已关闭的实验协议。

原目标为 raw RGB L1 + 0.2 SSIM loss + 0.1 场景 Triplet + 0.05 照明 L1 + 空间余弦一致性；一致性权重前5轮线性升至0.02。真实等级只参与照明监督，不作为模型额外输入。Bias 和 rho 不做 AdamW weight decay。

以下训练/续训命令是接口说明，**本次没有执行**。输出必须是当前工程内部的新目录；续训只接受该未完成 run 的完整 `last.pth`：

```powershell
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new --resume outputs/A0_seed42_new/checkpoints/last.pth
```

训练只读取官方 train/validation，按 validation 平均 PSNR 选择 best，不自动运行 test/LOL。已有冻结 best/last、smoke、不同合同或中途未完成 epoch 的 checkpoint 均不能作为该 run 的合法续训状态。

## 推理与评估

```powershell
# 以下推理示例中的输入应替换为实际图片或目录；本次仅用合成 RGB 夹具验证接口。
python -m scripts.infer --input "图片或目录" --output outputs/A0_infer_new --device cpu
# 评估示例，本次未执行真实数据评估。
python -m scripts.evaluate --dataset mill --output outputs/A0_MILL_test_new
python -m scripts.evaluate --dataset lolv1 --output outputs/A0_LOLv1_test_new
python -m scripts.evaluate --dataset lolv2_real --output outputs/A0_LOLv2_real_test_new
python -m scripts.evaluate --dataset lolv2_synthetic --output outputs/A0_LOLv2_synthetic_test_new
```

`infer/evaluate` 默认严格加载 best88。评估为 FP32、batch1、原尺寸整图 RGB、外层 clamp、PNG 量化前计分，无 resize、tile、TTA 或 GT 均值对齐。SSIM 使用 pytorch-msssim 原参数；LPIPS 使用已有本地冻结权重，缺失则记 NA，不用随机权重替代。所有写路径拒绝受保护目录、越界、别名、流语法及链接/junction。

数据根为 `data/MILLs-DSLR/`、`data/LOLdataset/` 和 `data/LOLv2/`。MILL 使用完整 content_id 配对GT，末尾1–10为原等级；不根据Scene编号猜测物理背景。固定三元组的anchor/positive来自同content不同等级，negative来自其他content，低光与GT同步裁剪和增强。LOL按文件stem严格配对，仅作显式评估，不参与训练。原冻结manifest及文件指纹保存在 `results/A0_best88/` 的训练与评估记录中。

## 已有结果及限制

这些数字来自原冻结证据，本次未重测：

| 划分 | N | PSNR_RGB | SSIM_RGB |
|---|---:|---:|---:|
| MILL validation | 120 | 33.16664658589749 | 0.9364022394021352 |
| MILL test | 79 | 17.5628851837581 | 0.7783799616596366 |
| LOLv1 test | 15 | 16.338013063035216 | 0.7100635727246603 |
| LOLv2 Real test | 100 | 20.23149322186524 | 0.7679075080156327 |
| LOLv2 Synthetic test | 100 | 14.904456186727648 | 0.6492352271080017 |

来源为 [best validation](results/A0_best88/validation/A0_seed42/best/summary.json) 与 [四集 summary](results/A0_best88/evaluation/frozen_review/metrics/summary.csv)。MILL test 输入 PSNR 为 21.718134396878007，增强平均下降 **4.155249213119908 dB**。保留 A0 是保留第一版基线，不表示已解决背景、相机或跨数据集泛化，也不表示在不同协议间全局最优。

恒等监督三 seed、SG/AUX 的正负结果、Stage6 的 **3组完成、1组 AMP 门禁失败、2组未启动、heldout 全未评估** 均留在实验史。内部 selection 不是 test；照明 MAE 改善不等于增强泛化改善。

## 本次恢复验收

此前恢复使用本地 pre-Stage1 的训练/配置/CLI 和 pre-Stage3 A0 模型快照，保留必要安全校验。CPU FP32 固定合成输入 256×256、255×257、1×1 的 raw enhanced/Fs/Fd/c_hat 逐值一致；148项 state_dict、12,270,696参数、初始化/RNG及 best/last 合同均核对。54项定向测试通过，具体记录保存在本地 `results/research_history/maintenance/a0_restore/recovery_report.md`。

此前有限检查包含临时模型的局部梯度，参数更新次数为0；没有训练、续训、20步短测、真实图像评估、heldout 补测或完整四集重测。CPU 等价不证明所有 CUDA 运算逐值等价。完整历史与审阅 ZIP 保留本地。

## 代码发布范围

本次Git更新仅涉及 `V1新版无去噪/` 的活动代码、测试、使用说明及本目录忽略规则。父级与兄弟项目不随此提交更新。新增输出路径保护拒绝Windows大小写及尾点/空格、NTFS流语法绕过；配置入口明确拒绝已关闭的实验协议，A0网络及数值训练协议不变。

数据集、best88/last100、LPIPS缓存、实验档案和移交ZIP不上传。clone得到代码，不会自动得到这些本地材料；没有提供权重下载地址。运行需要自行准备对应数据与本地权重。包含冻结权重检查的测试同样需要本地文件；路径与配置拒绝测试可单独运行。
