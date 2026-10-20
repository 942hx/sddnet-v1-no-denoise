# 新版v1无去噪 · SDDNet-v1 v1_clean

这是 **SDDNetV1 的第一版内部修订**，checkpoint 标识为 `arch_revision: v1_clean`。保留共享编码器、结构分支、退化分支、全局亮度预测、条件融合和重建解码器；不包含独立末端去噪、小波/DWT/IDWT、HVI、显式Retinex分解或对抗训练。

仓库有两个独立分支：**新版v1无去噪**（当前默认分支）和 **旧版v1无去噪**。每个分支仅包含该版代码、配置、测试、依赖清单和运行说明，没有数据、checkpoint、增强图片或训练日志。Git常用操作见 [GIT_GUIDE.md](GIT_GUIDE.md)。

## 安装环境

Windows / conda 环境 `testmodel`。本机验证环境为 Python 3.9、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121；已有兼容环境直接复用，具体依赖记录见 `requirements-lock.txt`。

```bat
conda activate testmodel
scripts\setup_env.bat
```

如果尚未创建环境，先创建 `conda create -n testmodel python=3.9 -y`。setup脚本只安装缺失依赖、检查目标解释器和CUDA，不卸载或盲目升级已有兼容包。PyTorch/CUDA wheel独立按匹配组合安装，不安装整套系统CUDA Toolkit。参考 [PyTorch 官方版本组合](https://pytorch.org/get-started/previous-versions/)。

## 数据目录

```text
data/
  MILLs-DSLR/
    train/{input,gt}
    validation/{input,gt}
    test/{input,gt}
  LOLDataset/
    our485/{low,high}
    eval15/{low,high}
  LOLv2/
    Real_captured/{Train,Test}/{Low,Normal}
    Synthetic/{Train,Test}/{Low,Normal}
```

数据自行准备，不随Git上传。MILL按文件名最右侧1..10等级后缀配对，例如 `Scene1_a-3.png → Scene1_a.png`；完整content_id不合并 `_a/_b`，不能将input/GT排序zip。正样本同内容不同等级并同步几何变换，负样本异内容同等级；无明确拍摄分组时不猜测物理场景。审计不确定的配对会失败。官方损坏项 `Test13-9.png` 排除而不删除源图。

只使用MILL train训练、validation选最佳权重。MILL test和LOL仅最终测试，不用于选择权重；LOL结果是MILL训练后的跨数据集评测。

## 运行命令

在当前分支项目根目录运行：

```bat
conda activate testmodel

rem 环境、数据审计和不更新参数的验收
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --check-only

rem 独立真实数据短测：20次成功优化器更新，不是20轮正式训练
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --smoke-test

rem 从头100个完整epoch，随后用同一验证最佳权重测试四个数据集
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml

rem 续训累计到100轮，不是额外100轮
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --resume outputs/sddnet_v1_clean_mill_100e/checkpoints/last.pth

rem 只评测本版本checkpoint；权重文件自行准备
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --eval-only --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth

rem 自动测试
python -m pytest -q
```

唯一正式配置为 `configs/sddnet_v1.yaml`：100轮、patch256、triplet batch1（实际3张图）、累积2步、workers0、AMP、AdamW lr1e-4、weight decay1e-4、cosine到1e-6、梯度裁剪1.0、无提前停止；每完整epoch验证，以逐图平均PSNR_RGB保存best。bias/rho不衰减。CUDA不可用时只做CPU检查，正式训练不会悄悄改用CPU。新实验修改 `run_dir`，保留已有记录；运行锁拒绝重复进程。

## 模型接口与损失

```python
from models import SDDNetV1

model = SDDNetV1()
result = model(low_rgb)  # 只需浮点[B,3,H,W]低光输入，Dataset范围[0,1]
y_raw = result["enhanced"]
latents = model(low_rgb, return_latents=True)  # enhanced、Fs、Fd、c_hat
```

输入先计算平滑max-RGB和RGB算术均值的两通道亮度先验。图像/先验补到8的倍数；输出裁回原尺寸。编码器通道32/64/128/256，残差块数2/2/2/4；双分支均128通道，退化分支输入258通道。条件仿射分别调制瓶颈和三条skip，增益 `sigmoid(rho)` 从0.1开始学习。RGB head输出有正负的残差，`y_raw=x+delta`，模型内部不clamp；默认返回仅含enhanced，潜特征按需返回，内部计算公式一致。

普通隐藏Conv/Linear采用LeakyReLU(.1)匹配的Kaiming normal、bias0；仿射末层、RGB head和照明末层使用Normal(0,.001)、bias0。加载checkpoint后不重新初始化。

训练按cat([anchor,positive,negative])排列，重建/照明监督全部3B图，辅助特征不detach：

```text
L_rec = mean|y_raw-GT| + 0.2*(1-SSIM(clamp(y_raw,0,1),GT))
L_scene = TripletMarginLoss(margin=.2,p=2,eps=1e-6,swap=False)
          on L2-normalized GAP(Fs); distance is unsquared L2
L_illum = mean|c_hat-c_gt|, c_gt=(level-1)/9, strictly float32 [N,1]
L_cons = mean(max(1-channel_cosine(Fs_anchor,Fs_positive),0))
L_total = L_rec + .1*L_scene + .05*L_illum + lambda_cons*L_cons
```

一致性权重前5个全局epoch线性从0升到.02，之后固定；恢复不重置进度。L1直接监督未clamp输出；SSIM窗口11、sigma1.5、data_range1、RGB平均，使用pytorch-msssim 1.0.0。不同场景不做空间一致性。没有加入无界Fd分离、任意特征间隔、强制正交、IN替代监督或阻断重建梯度；这些目标不保证可识别解耦，还可能通过放大幅值作弊或冲突。

本模型不声称严格统计独立、物理反射率/照明图分解或可解释噪声估计。诊断记录特征统计、条件干预、仅重建损失到分支/亮度头的梯度；模块存在不等于条件已被有效使用。

## 评测、输出与断点

评测为FP32整图batch1、eval+inference_mode、外层clamp；无tile/TTA、GT亮度对齐、任意resize或额外裁边。PSNR_RGB为RGB MSE协议，SSIM实现如上，LPIPS为官方预训练Alex v0.1、输入2*x-1。指标在PNG量化前逐图计算并平均；输入也计算基线。缺指标写NA及原因，LOL没有MILL等级标签时不计算照明MAE。

run目录保存环境、resolved配置、数据审计、metadata、checkpoints/best.pth与last.pth、history.csv、诊断、TensorBoard、四集metrics、增强图、profile和report。性能固定单张400×600、预热20、CUDA同步计时100次，排除IO和指标耗时。所有运行输出均被Git忽略。

checkpoint严格校验架构、训练配置和数据指纹，保存optimizer/scheduler/scaler、完整epoch、global step、RNG和sampler状态。旧版checkpoint不兼容新版resume。CSV被Windows查看器占用时，完整新快照保留为 `*.pending.csv`，流程读取最新快照；每轮JSON统计另存 `epoch_records/`。无法恢复的历史训练均值明确记NA，曲线留空，不伪造损失。

第二版计划另加独立去噪块进行对比实验；本分支不包含该模块，也不包含未使用的第二版占位代码。
