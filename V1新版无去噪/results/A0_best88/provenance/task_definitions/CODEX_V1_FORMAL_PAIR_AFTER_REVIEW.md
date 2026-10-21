# SDDNet-v1：压缩包复核后的正式对照指令

## 0. 任务范围与已知事实

工作目录：`E:\结构亮度结构解耦\V1新版无去噪`。
环境：conda `testmodel`；已有 RTX 4060 Laptop CUDA 环境已通过真实短测。
本任务接续已有 condition-ablation 工程，不重写网络、不重复导出材料、不安装新模型、不升级能正常工作的依赖。

归档 `condition_ablation_materials_20261021_105941.zip` 的187份清单文件通过大小与SHA256核验。归档提供了代码、配置、历史指标和smoke日志，没有原始图像或权重本体。外部复核仅做了静态检查与小尺寸CPU前向，没有在本机替你完成CUDA训练。

已确认：
- 历史 `outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth` 为 epoch 73，SHA256：
  `c2d34cb31961f86eeeef23bbc3920704a7bff4eee180c1bb2f8a3ce27da0869d`。
- 历史metrics/evaluation_status.json、per_image.csv、checkpoint_identifiers.json一致指向此best。
- history中epoch73 validation PSNR_RGB=32.69872242054443；同best历史MILL test PSNR_RGB约17.20956，input约21.71813。两者是不同split，不可写成新版提升。
- 历史test Level10平均c_hat约0.346742，目标为1；平均illumination MAE约0.653258。该结果是真实已有记录，不是此轮新模型的成绩。
- 当前模型的标签形状、L1照明监督、无GT/level前向、合法Triplet与predicted-c gate已实现。未发现必须再次重做模型或损失的依据。
- A2 smoke只完成20次更新，不能作为正式resume或最终测试权重。

本轮目标：修正开发/测试的入口隔离和比较汇总；对旧best做一次只读照明校准诊断；若数据与工程检查通过，从头串行训练A0与A1各100轮，只用MILL validation选择和比较。暂不启动A2/A3、额外batch、初始化或多seed组。

## 1. 安全边界

1. 先检查Git状态、既有运行锁、相关进程、解释器、CUDA及输出目录；不终止正在运行的任务，不并行占用同一GPU。
2. 修改前只做小型源码/配置快照与改动清单；不复制数据集、不删除历史版本、权重、CSV、日志、审计报告或smoke。
3. 若A0/A1正式history或checkpoint已存在，不覆盖、不自行从头再跑，先报告现状并给明确续训命令。
4. 不修改数据内容、划分或level命名。正式前发现可证实的数据配对/标签/泄漏错误，立即停止并报告；不能默默修正后继续训练。
5. 保留旧环境，禁止为了此次任务重装conda/CUDA、盲目升级依赖。

## 2. 必须补的流程修正——不改变网络与训练目标

### 2.1 新增真正的 `--train-only`

当前 `scripts/run_pipeline.py` 的正式分支在 `formal_training` 后无条件调用 `evaluate_all` 和 `profile_model`；`engine/evaluation.py` 对ablation会显式调用 `prepare_data(scope='final')`。因此单有 `data.scope=mill_development` 不能隔离正式训练结束后的测试集访问。

在原统一入口增加 `--train-only`：
- 与 `--check-only/--smoke-test/--eval-only` 互斥；可与 `--resume` 同用。
- 仍执行100个完整epoch，每轮MILL validation，保存best/last。
- 不调用evaluate_all，不扫描/解码test或LOL，不创建final_data，不下载LPIPS，不隐式执行性能profile。
- 训练后strict加载同一run的best，完整验证MILL validation 120张，保存正式best验证逐图与逐等级结果到 `validation/best/`。
- 保存best对应epoch、文件SHA256、配置指纹、data指纹、完整验证count及选模规则；不要把最后epoch的验证结果当作best结果。
- 保留显式 `--eval-only --checkpoint ...` 作为以后冻结方案后的四数据集测试入口。smoke仍必须被拒绝。
- 开发scope的裸训练命令也默认不自动测试，或者明确拒绝并提示使用--train-only；禁止仅凭“训练结束”就把正在做配置选择的run视作已经完成最终冻结。
- `profile.enabled: false`必须真实生效；不能在run_pipeline里无条件调用后忽略它。
- 新增单元/入口回归测试：即使test/LOL目录不存在，train-only仍可在合成fixture中完成；用会抛异常的spy证明evaluate_all、final audit和LPIPS不被调用。测试不需要真的训练100轮。
- 保留resume契约、run_kind=formal、AMP、累积、scheduler、每轮验证/诊断原来的数值时序。不要因新增入口改变loss、初始化或优化器。

### 2.2 比较脚本必须按dataset与split分开汇总

当前 `utils/ablation_metrics.compare_control` 对所有匹配行只输出一个汇总，`paired_summary`的cluster key也没有dataset命名空间；当传入四数据集per_image.csv时，会按图像数混合不同数据集，并可能混合同名scene。

修订现有比较实现，不另外写重复统计脚本：
- 按 `(dataset, split)` 独立输出均值、median、改善比例、样本数、场景数和场景聚类bootstrap区间。
- MILL另输出Level1–10，以及各content的paired差值。
- cluster key显式包含dataset与split；不把不同数据集的同名scene合并。
- 不默认输出跨四数据集的单一“总体提升”。如需要以后加宏平均，必须另命名和事先声明；本次不做。
- 保留candidate/control全图集合相同、input基线指标相同、禁止intervention CSV、非有限值不造数的检查。
- 对同一dataset少于两个独立cluster时，CI标NA并说明，不宣称单场景区间具有泛化统计意义。
- 对本轮单个MILL validation的比较保持原本逐图差值定义：candidate minus control。
- 兼容缺少dataset字段的validation CSV，明确默认为mill；固定split=validation，不猜测test。
- 增加混合dataset测试与同名scene测试。例如A集1图+10 dB、B集3图-1 dB，必须分别报告+10/-1，不能把+1.75 dB作为两集共同提升。

## 3. 旧best的只读照明/视野诊断

在现有 `scripts.condition_ablation` 增加子命令 `calibration`，不要改变正式model.forward签名。

建议接口：
```
python -m scripts.condition_ablation calibration --config configs/sddnet_v1.yaml --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth --output reports/best73_calibration_preformal
```

要求：
1. 只读取MILL train与validation；不重新评测test或LOL，不用它们调超参数。仅引用已有test CSV作历史问题描述。
2. 复用并核对现有metadata的实际图像哈希与配对，加载旧best时strict=True，记录真实checkpoint SHA256与epoch。不要触发旧run的prepare_data全数据审计，不覆盖旧config_resolved/status/metrics。
3. 模型eval+FP32+inference_mode，无optimizer，无权重修改，无新训练；GT与level只计算指标，不进入模型。
4. 对全部300个train low与120个validation low分别计算：
   - 原始完整分辨率；
   - 固定中心256×256裁剪，low与GT完全同步，不resize、不随机增强。
   不混合两种视野计算一个总指标，也不把固定中心crop等同于训练随机crop分布。
5. 每行记录split/content/filename/raw level/c_gt/c_hat/绝对误差/视野/输入GT路径与哈希、RGB均值(输入和GT分开)、Max-RGB均值、图像shape、raw输出越界率、PSNR_RGB/SSIM_RGB和vs-input差值。
6. 输出每split×view×level、每content的MAE、c_hat均值/标准差、标签计数、PSNR/SSIM；同一图像比较full与crop预测差。
7. 用训练标签的真实中位数构造常量预测基线，在相同split/view样本上比较MAE；只作诊断，不用常量替换正式c_hat。
8. 核对GT复用规则、内容ID保留后缀、[0,1]和(level-1)/9、标签[N,1]、全图评测与crop指标的区别。
9. 校准表现差不能单凭数值判定代码bug，也不能据此自动增大loss权重或添加新gate/归一化/损失。记录“未证明原因”；数据与实现正确时允许继续受控实验。
10. 输出 `calibration_per_image.csv`、`calibration_per_level.csv`、`calibration_per_content.csv`、`view_comparison.csv`和报告。输出目录若已存在且非空则要求新名字。

这一步的目的是分清训练/验证分布与视野效应，不承诺从这些数据就能解释test失效，也不新做oracle/GT条件输入实验。

## 4. 正式实验前验收与锁定

- 运行现有测试和新增测试；明确真实PASS/FAIL/SKIP，不要求固定总数仍为82。
- 核对A0与A1合并继承后的配置，除variant/run_dir和affine_mode外，数值训练设置必须相同。
- 使用同一个seed=42，两者共有参数初始化应逐值一致；bounded多出的4个rho标量单独记录。此检查不得通过strict=False加载旧模型实现。
- 两组均：patch256，triplet batch1=3图，accumulation2=名义6图/更新，每轮900次图像暴露、150次更新尝试，100轮、同一loss/学习率/调度/选模规则。
- 本轮故意不改为16图或PK，否则无法隔离FiLM公式。A2的16图短测证明显存可行，不是A0/A1应该同时改采样的理由。
- gate=false、local_init_std=.001，不开启A1_init，不改loss权重、不添加去噪或小波。
- 保存配对配置diff、代码版本/文件哈希、环境、数据指纹与计划；更新数/AMP跳步/图像暴露按实际统计。
- 在Windows本机做必要的CUDA入口短测，用独立smoke目录，不与正式权重混用；不要删除/覆盖已有20步记录。

## 5. 通过检查后，串行运行A0、A1各100轮

若有真实标签/配对/finite/梯度/入口测试失败，先停止报告，不擅自修改网络后训练。
若检查通过，仅质量指标不够好并不是工程失败，不为了“先有好结果”偷偷改方案。

执行：
```
conda activate testmodel
cd /d "E:\结构亮度结构解耦\V1新版无去噪"

python -m scripts.run_pipeline --config configs/ablations/A0.yaml --train-only
python -m scripts.run_pipeline --config configs/ablations/A1.yaml --train-only
```

要求：
- 从新初始化开始；不传smoke、旧best或其他variant作为resume。
- A0完整结束且状态验证成功后才启动A1，不并发。
- 不覆盖原 `outputs/sddnet_v1_clean_mill_100e`，本轮仍使用各自 `outputs/condition_ablation/A0_seed42` 与 `A1_seed42`。
- 每组累计100轮，无提前停止；best按MILL validation平均PSNR_RGB选择，保存SSIM及各level辅助结果。
- 不主动终止训练。若会话不能持续到完成，真实报告进程、日志、已完成轮数及准确的独立终端命令，不谎称跑完。
- 可提供 `scripts/run_A0_A1_train_only.bat` 供Anaconda Prompt调用：顺序执行、每一步检查退出码和run状态，A0失败不执行A1；不要自动新建隐蔽后台进程。
- 断点继续示例：
```
python -m scripts.run_pipeline --config configs/ablations/A0.yaml --train-only --resume outputs/condition_ablation/A0_seed42/checkpoints/last.pth
```
  A1同理。验证resume仍是同variant/同data/同完整epoch的formal检查点。

## 6. 本轮完成后的结果与边界

两组都完成后，加载各自best做完整MILL validation，使用修正后的compare比较：
```
python -m scripts.condition_ablation compare --candidate outputs/condition_ablation/A1_seed42/validation/best/per_image.csv --control outputs/condition_ablation/A0_seed42/validation/best/per_image.csv --output reports/A1_vs_A0_validation_formal
```

以上best输出路径应由本次实现提供并验证，不是旧代码已有能力的假设。

交付：
- A0/A1 completed_epoch、attempted/successful更新数、AMP跳步、总暴露、best_epoch与SHA256；
- 完整validation PSNR_RGB、SSIM_RGB、Level1–10结果、输入参照、逐图vs-input与A1-vs-A0；
- 照明MAE与full/crop区别；条件诊断仍明确只是依赖/敏感性检查；
- 各组训练/验证耗时、显存、日志与实际配置；
- 单seed开发结果，不宣称统计普适优势，不把validation选择后的结果当独立test成绩。

暂不运行A1_PK_random/A2/A3/A_batch/A1_init，不跑全部组合，不跑四个test集。下一阶段若要研究半难负例，应成对跑A1_PK_random与A2；不能拿A2直接对A0而把差异全部归因于FiLM或miner。

本轮新增最终报告 `reports/formal_pair_review_and_results.md`。保留旧材料；不再重复“清理整项目”。
