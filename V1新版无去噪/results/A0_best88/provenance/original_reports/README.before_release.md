# 新版v1无去噪 · SDDNet-v1 v1_clean

这是 **SDDNetV1 的第一版内部修订**，checkpoint 标识为 `arch_revision: v1_clean`。保留共享编码器、结构分支、退化分支、全局亮度预测、条件融合和重建解码器；不包含独立末端去噪、小波/DWT/IDWT、HVI、显式Retinex分解或对抗训练。

## 条件通路受控消融（2026-10-21）

本次任务依据工作区根目录 `CODEX_V1_CONDITION_ABLATION.md`，仍为V1；一套模型和训练器，通过配置切换，不引入去噪或小波。实际验收、限制、源码恢复位置见 [审计报告](reports/condition_ablation_audit.md)，配置矩阵见 [ablation_matrix.csv](reports/ablation_matrix.csv)。旧实验目录保持原样。

`ConditionalAffine` 的 bounded 路径复现原公式 `f*(1+sigmoid(rho)*tanh(gamma_raw))+sigmoid(rho)*tanh(beta_raw)`；standard 路径为 `(1+delta_gamma)*f+beta`，没有rho/tanh/sigmoid增益。两者默认local末层std均为.001，global、RGB和照度末层仍是小幅非零初始化。`.003`只在独立的`A1_init`中改变local幅度。standard移除4个未使用的rho标量，参数量12,270,692；bounded仍为12,270,696，编码器/解码器通道深度不变。

| 配置 | 单独研究因素 | 物理图数 × 累积 | 负例候选 |
|---|---|---|---|
| `configs/ablations/A0.yaml` | 当前bounded对照 | 3×2=6 | 固定triplet的1个 |
| `A1.yaml` | A0仅改standard公式 | 3×2=6 | 同A0 |
| `A1_init.yaml` | A1仅改local std=.003 | 3×2=6 | 同A1 |
| `A2_triplet.yaml` | 原sampler内半难选择检查 | 3×2=6 | 只有1个，没有选择空间 |
| `A1_PK_random.yaml` | PK随机负例对照 | 8×2=16 | 每anchor通常3个同level异content |
| `A2.yaml` | 同PK配置仅改半难miner | 8×2=16 | 同上；先半难，再hard fallback，全easy允许0 |
| `A3.yaml` | A2仅开predicted-c输出gate | 8×2=16 | 同A2 |
| `A_batch18.yaml` | A1单独扩大累积 | 3×6=18 | 仍只有1个，累积不扩大候选池 |
| `A_batch16.yaml` | PK有效16图候选 | 8×2=16 | 与triplet同时改变sampler，不作纯batch归因 |

文件名简称均位于`configs/ablations/`。P=4,K=2的PK按图像暴露预算定义epoch（默认900图，尾批4图），不保证每张low恰好一次；实际覆盖、重复和更新写入coverage/mining日志。原triplet每轮300个anchor=900次low图像暴露。两种sampler虽可匹配900次暴露，但更新数不同（150 vs57次尝试/轮）；只将同PK随机/半难对照用于miner归因。batch研究必须同时看等暴露预算和等更新预算，详见矩阵，不能将少更新归因为更大batch的收益。

gate默认关闭。开启时所有阶段均为`g=.25+.75*(1-c_hat)`、`y_raw=x+g*delta`，不detach预测c，不喂真实level。level10标签归一化为1不等于正常GT，也不强制恒等。`L_rec+.1*L_scene+.05*L_illum+lambda_cons*L_cons`权重和未clamp的L1保留；不添加强制shuffle敏感、正交或额外分离损失。

新配置只审计/训练MILL train，在validation选权重。最终评测时才单独审计test/LOL到`final_data/`，不替换训练清单指纹。先前eval15及LOLv2成绩已经被查看，不能声称这些集合从未接触；本轮没有将它们用于开发或选择。未创建LOL-dev，也不编造其结果。

在本目录的Anaconda Prompt执行：

```bat
conda activate testmodel
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --check-only
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --smoke-test
python -m scripts.condition_ablation matrix --output reports/ablation_matrix.csv
python -m scripts.condition_ablation memory-check --config configs/ablations/A_batch18.yaml --output reports/memory_new_measurement.json
```

本次已执行检查、真实CUDA AMP 20次成功更新和固定验证诊断。没有自动执行下面的100轮命令；续训/正式测试路径要等该配置的正式训练生成后才存在：

```bat
python -m scripts.run_pipeline --config configs/ablations/A2.yaml
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --resume outputs/condition_ablation/A2_seed42/checkpoints/last.pth
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --eval-only --checkpoint outputs/condition_ablation/A2_seed42/checkpoints/best.pth
```

A0/A1等各自使用不同run_dir，从头训练；新variant不可用旧版权重续训。checkpoint严格记录architecture/config hash、variant、seed、epoch、成功update和数据指纹。诊断频率仍可通过`--grad-norm-every 0/1/100`设置，默认100。学习率仍按完成epoch在原位置step，保存时额外记录对应成功optimizer update，resume不重置；PK的epoch为明确的暴露预算。若要未来改成按update调度，应另设所有对照一致的新协议，本次未偷偷改调度。

原新版v1_clean的best只在原bounded配置下严格加载作baseline（不是并列“V1旧版无去噪”的另一架构）；以下命令只写新诊断目录，不重写旧结果。输出目录已存在时请指定新名字：

```bat
python -m scripts.condition_ablation diagnose --config configs/sddnet_v1.yaml --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth --output reports/condition_baseline_new_diagnostic
python -m scripts.condition_ablation diagnose --config configs/ablations/A2.yaml --checkpoint outputs/condition_ablation/A2_seed42/smoke/last.pth --allow-smoke --output reports/condition_smoke_new_diagnostic
```

`--allow-smoke`只允许显式短测诊断，不能进入正式`--eval-only`。固定清单取3个validation content的全部level，共30图、中心128裁剪。只缓存donor的CPU Fd/c，固定anchor输入/Fs/全部skip；同content跨level、异content同level、无固定点的随机joint分别输出，fd-only/c-only标为人为不一致条件。neutral-affine保留gate，gate-only-off和all-condition-off分开。指标基于外层clamp前后明确区分；这不是整图测试或严格因果解耦证明。

`condition_per_image.csv`/`condition_summary.csv`同时给raw/clamped输出MAE和`condition_quality_drop=PSNR(normal)-PSNR(replaced)`，不与`delta_psnr_vs_input`或`delta_psnr_vs_control`混用。尚无同预算训练完成的候选/控制正式评测，vs-control为NA。正式验证/测试还输出逐图/逐level指标、input基线、中位数/改善比例及按content（已核实group优先）聚类的bootstrap；不是将各level当独立场景。峰值显存来自真实运行，不以系统16GB内存推断。

候选与控制都完成相同协议评测后，可用同一入口的`compare`子命令生成vs-control差值；它校验图像集合、split和input基线，不接受条件干预CSV。示例（本次只以临时fixture验证接口，尚无正式候选/控制结果）：

```bat
python -m scripts.condition_ablation compare --candidate outputs/condition_ablation/A1_seed42/metrics/per_image.csv --control outputs/condition_ablation/A0_seed42/metrics/per_image.csv --output reports/A1_vs_A0_final
```

自然epoch包含尾窗，且AMP可跳过更新。矩阵中的等5000次“满窗口”暴露量只是理论口径，不能冒充实际预算；实际等更新比较要累计`updates.csv`/history的成功update与图像暴露。现阶段没有等更新正式消融结果。

本目录只加载自己的模型与工具。原bounded参数量12,270,696，standard为12,270,692；均不同于并列旧架构的12,477,156。Git常用说明见 [GIT_GUIDE.md](../GIT_GUIDE.md)。

## 整理后文件位置与入口

以下命令都在本目录运行。Anaconda Prompt 从工作区根目录使用 `cd /d "V1新版无去噪"`，PowerShell 使用 `Set-Location -LiteralPath "V1新版无去噪"`。原始正式实验已完成100轮；从头运行使用新的输出配置，现有last用于累计续训，不会追加100轮。

- `data/`：您已复制的6,137个文件原样保留；实际子目录名为 `MILLs-DSLR`、`LOLdataset`、`LOLv2`。
- `models/`、`datasets/`、`losses/`、`engine/`、`utils/`、`scripts/`、`tests/`：新版独立源码与测试，不依赖旧版私有模块。
- `configs/`：现有实验、新训练目录、独立重评测目录三个路径配置。
- `outputs/sddnet_v1_clean_mill_100e/`：已有正式实验，保留 best（epoch73）、last（epoch100）、独立smoke权重、optimizer/RNG状态、history、TensorBoard、metadata、原始评测、图片和恢复记录。
- `outputs/sddnet_v1_clean_mill_100e_new_run/`、`outputs/sddnet_v1_clean_mill_100e_reevaluation/`：相应配置将来运行时才创建，当前未启动训练或重评测。
- `cache/torch/`：独立离线LPIPS的官方AlexNet依赖；复制现有已校验依赖，不重新下载。
- `docs/`：新版任务设计、日志恢复记录及同一v1_clean修复前后审计；[历史定位说明](docs/README.md)。历史快照中的旧路径保持原样，快照脚本不作为当前入口执行。
- [跨版本比较](../V1旧版和V1新版对比/README.md)；[整理报告](../V1旧版和V1新版对比/docs/整理报告.md)。

统一训练入口是 `scripts.run_pipeline → engine.training.formal_training`，逐轮验证包含在训练中；测试/增强输出由 `--eval-only → engine.evaluation.evaluate_all` 调用。没有独立的任意单图推理CLI。模型API见后文。

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
  LOLdataset/
    our485/{low,high}
    eval15/{low,high}
  LOLv2/
    Real_captured/{Train,Test}/{Low,Normal}
    Synthetic/{Train,Test}/{Low,Normal}
```

数据自行准备，不随Git上传。MILL按文件名最右侧1..10等级后缀配对，例如 `Scene1_a-3.png → Scene1_a.png`；完整content_id不合并 `_a/_b`，不能将input/GT排序zip。正样本同内容不同等级并同步几何变换，负样本异内容同等级；无明确拍摄分组时不猜测物理场景。审计不确定的配对会失败。官方损坏项 `Test13-9.png` 排除而不删除源图。

只使用MILL train训练、validation选最佳权重。MILL test和LOL仅最终测试，不用于选择权重；LOL结果是MILL训练后的跨数据集评测。

## 运行命令

在本版本目录运行（先从工作区根目录进入 `V1新版无去噪`）：

```bat
conda activate testmodel

rem 环境、数据审计和不更新参数的验收
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml --check-only

rem 独立真实数据短测：20次成功优化器更新，不是20轮正式训练
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml --smoke-test

rem 从头100个完整epoch，随后用同一验证最佳权重测试四个数据集
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml

rem 续训累计到100轮，不是额外100轮
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --resume outputs/sddnet_v1_clean_mill_100e/checkpoints/last.pth

rem 只评测本版既有best；输出新reevaluation目录，保留原始实验
python -m scripts.run_pipeline --config configs/sddnet_v1_eval.yaml --eval-only --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth

rem 自动测试
python -m pytest -q
```

既有正式实验配置为 `configs/sddnet_v1.yaml`；`sddnet_v1_new_run.yaml` 和 `sddnet_v1_eval.yaml` 只将输出目录分别改为 `_new_run`、`_reevaluation`，其余实验参数一致。正式设置为：100轮、patch256、triplet batch1（实际3张图）、累积2步、workers0、AMP、AdamW lr1e-4、weight decay1e-4、cosine到1e-6、梯度裁剪1.0、无提前停止；每完整epoch验证，以逐图平均PSNR_RGB保存best。bias/rho不衰减。CUDA不可用时只做CPU检查，正式训练不会悄悄改用CPU。新实验修改 `run_dir`，保留已有记录；运行锁拒绝重复进程。

### 学习率与梯度日志

`history.csv` 的 `lr` 和 TensorBoard `epoch/lr` 记录该epoch真正训练使用的第0参数组学习率；`lr_group_0`、`lr_group_1`等分别记录各组。初始lr=1e-4时，第1轮记录0.0001。scheduler仍在原来的epoch末位置调用一次，checkpoint中optimizer保存的是供下一轮使用的lr，resume后按该值训练并正确记录。TensorBoard旧标量格式使用float32编码，与CSV的float64可能有末位舍入差异。

分支梯度统计可在统一入口单独配置：

```bat
rem 默认：每100次成功optimizer更新采样一次
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml --grad-norm-every 100
rem 短测时逐次检查
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml --smoke-test --grad-norm-every 1
rem 关闭这项统计（原有梯度裁剪和固定验证探针继续保留）
python -m scripts.run_pipeline --config configs/sddnet_v1_new_run.yaml --grad-norm-every 0
```

默认值来自`diagnostics.grad_norm_every: 100`，CLI优先；0关闭、1逐次、N按实际成功更新计数，忽略microbatch和AMP跳过的更新。resume使用原有`optimizer_updates`累计计数；缺失可靠计数时明确拒绝恢复，不从microbatch数猜测。诊断频率不属于数值训练contract，可以在resume时显式调整，实际值记入resolved配置和新checkpoint配置。

统计保持为unscale之后、clip之前的梯度范数。仅候选采样点读取GradScaler的预步overflow状态，并与原有old-scale读取合并为一次回传；AMP跳过时不遍历分支、不记录样本。八个分支在设备端汇总后一次回传；普通非采样步骤不进入这项统计。当前只读AMP适配已核对PyTorch 2.5.1的`_get_scale_async`和`_found_inf_per_device`，未来版本如移除它们会明确报错，可先用`--grad-norm-every 0`。backward、梯度裁剪、scaler step/update、zero_grad及scheduler调用均保持原顺序。

history新增`grad_norm_samples`和仅对实际采样值计算的`*_grad_norm_total_mean`，无采样时范数字段缺失/空值，不记0。TensorBoard `gradient/*`只记录真实采样，step为成功的optimizer更新数；原有`grad_norm_preclip`仍是梯度裁剪返回的全网范数，不受此频率控制。固定验证探针中的重建梯度检查也保持原来的低频调用。

旧实验的CSV和event文件不会自动修正。已知原始cosine配置、各参数组base_lr、eta_min、T_max，且核实每个完整epoch只step一次时，epoch N的训练lr可用`eta_min+(base_lr-eta_min)*(1+cos(pi*(N-1)/T_max))/2`重建，并与checkpoint的scheduler.last_epoch/base_lrs和optimizer的下一轮lr交叉核对。只能另存`history_lr_reconstructed.csv`等审计副本并注明来源；若中途改过调度或信息不全，应保留NA，不能覆盖原始history/event证据。恢复缺失历史行时也不把checkpoint内的下一轮lr冒充本轮训练lr。

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

正式`--eval-only`、正式resume及直接正式评测入口还要求`run_kind='formal'`。`smoke`、缺失或未知run_kind均拒绝，不按epoch编号猜测，也不自动改写标签；拦截在创建正式输出、准备数据和模型推理之前。既有本版正式checkpoint已保存formal标记，继续兼容；缺标记文件如需迁移，必须先独立核实来源并另存可审计副本，本次不提供自动绕过选项。显式标为短测的架构/profile诊断仍可读取smoke checkpoint，但不能生成正式测试summary。

第二版计划另加独立去噪块进行对比实验；本分支不包含该模块，也不包含未使用的第二版占位代码。
# A0/A1 正式配对（复核后阶段）

本阶段仅串行从新初始化训练 A0、A1 各100轮，用完整 MILL validation 选 best。
统一入口新增 `--train-only`（可与 `--resume` 同用，与三种检查/短测/评测模式互斥）。
`data.scope=mill_development` 的裸训练也默认隔离最终测试。该模式只审计 train/validation，
不访问 test/LOL、不创建 final_data、不加载或下载 LPIPS、不做 profile。
训练结束会 strict 加载同一 run 的 best，导出 `validation/best/per_image.csv`、`per_level.csv`、
`summary.json` 与含 checkpoint SHA256、best epoch、配置/数据指纹的 `provenance.json`。
`profile.enabled=false` 对其他入口和直接 profile 调用也有效。

```bat
conda activate testmodel
cd /d "E:\结构亮度结构解耦\V1新版无去噪"
python -m scripts.run_pipeline --config configs/ablations/A0.yaml --check-only
python -m scripts.run_pipeline --config configs/ablations/A0.yaml --train-only
python -m scripts.run_pipeline --config configs/ablations/A1.yaml --train-only
rem 中断后，只恢复同一 variant 的完整 epoch formal checkpoint：
python -m scripts.run_pipeline --config configs/ablations/A0.yaml --train-only --resume outputs/condition_ablation/A0_seed42/checkpoints/last.pth
```

已验收的两组首次串行启动可运行 `scripts\run_A0_A1_train_only.bat`。
它在前台运行，逐步检查退出码及真实100轮状态，A0失败不会启动A1；已有正式结果时拒绝重新开始。
不要在已有进程运行时重复启动。两组均完成后会自动比较各自 best 的完整 validation。
本阶段不启动 A2/A3/PK/额外batch/初始化/多seed，也不运行四集测试。

A0/A1 共同：seed42、patch256、triplet1=3图、accumulation2=6图/更新、gate=false、local std=.001。
除 variant、run_dir、FiLM公式外配置一致，共有初始化逐值一致，bounded仅多4个rho标量。
每轮900图像暴露、150更新尝试；成功次数与AMP跳步按实际日志记录。
照明校准差不构成修改loss的理由，本轮不改模型或损失。

```bat
rem 历史best只读诊断；输出必须是新的空目录：
python -m scripts.condition_ablation calibration --config configs/sddnet_v1.yaml --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth --output reports/best73_calibration_preformal
rem 仅在两组best完整验证导出后：
python -m scripts.condition_ablation compare --candidate outputs/condition_ablation/A1_seed42/validation/best/per_image.csv --control outputs/condition_ablation/A0_seed42/validation/best/per_image.csv --output reports/A1_vs_A0_validation_formal
python -m scripts.formal_pair report
```

照明诊断只读300张train和120张validation，分别计算整图、同步中心256；不更新权重、不把GT/level输入模型。
两种视野分别报告，中心crop不等同训练随机crop。比较输出按dataset/split、MILL level/content隔离，
少于两个独立场景时bootstrap CI为NA；不生成混合四集的单一“总体提升”。
缺少dataset/split字段的validation CSV明确默认为mill/validation。
本轮真实状态和结果见 `reports/formal_pair_review_and_results.md`，旧记录和smoke均保留。

