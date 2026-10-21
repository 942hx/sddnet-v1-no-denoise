# SDDNet-v1 条件通路受控消融：真实验收报告

本次只在 **V1新版无去噪/** 修改，仍为第一版 v1_clean；没有去噪、小波或平行模型工程。
本文原baseline指本目录既有v1_clean/bounded best(epoch73)，不是并列“V1旧版无去噪”的另一架构/权重；其他版本未修改。
已完成审计、增量实现、自动化测试、真实CUDA AMP 20次成功更新和固定MILL validation诊断。**没有启动任何100轮正式消融，也没有执行本修订的完整四数据集测试。**

## 1. 先审计所得事实

- 旧正式last：完成100轮、30000个microbatch、15000次更新尝试、14995次成功更新、5次AMP跳步。源码正式调用未传max_updates，sampler/count检查完整遍历300个训练anchor/轮。
- 32秒口径可核实：checkpoint累计训练循环3228.305134秒/100=32.283051秒/轮。它不含验证、条件诊断、保存/报告。旧代码无显式CUDA synchronize，但每步多处.item会同步；不存在可重建的逐epoch完整墙钟列，后者标为NA。
- 每轮300个anchor、30个完整content，原三元组共900次low图像暴露；这不是900张独立图。原100轮共有90000次暴露（包含AMP跳过窗口），不等于90000次参数更新。
- history共100行，scene均值仅epoch70恰为0；epoch100为0.00033564051。TensorBoard有99个scene点，仅[70]为0。epoch76的CSV中断/恢复导致日志缺点，不能据99条日志否认已完成100轮。
- 逐UPDATE日志确实经常为0：14532/15000；最后150条为148条。这打印的是更新边界的最后一个microbatch loss，不是epoch均值，也不能重建所有microbatch的active比例。hinge满足margin后为0并不自动失效。
- 旧诊断置换[1,0,2]留有固定点，只有3个128裁剪；未分别输出clamp-MAE、joint及两种donor分层。其epoch100 c-shuffle raw-MAE=0.2213662、Fd-shuffle=0.04611329，不支持笼统“条件完全没用/极弱”。新诊断见第5节。
- 历史MILL test分等级CSV中的delta_PSNR_RGB定义为output-input，不是候选-control，也不是条件替换质量差。level3–10均值为负；只作为已有结果审计，本轮不重新读test图调参。

| 历史MILL test level | N | delta_psnr_vs_input | 照度MAE |
|---|---:|---:|---:|
| 1 | 8 | 11.79197 | 0.0009253306 |
| 2 | 8 | 2.545691 | 0.1016908 |
| 3 | 8 | -0.9140227 | 0.1934974 |
| 4 | 8 | -2.676464 | 0.2744678 |
| 5 | 8 | -4.263817 | 0.3447294 |
| 6 | 8 | -5.905344 | 0.4151389 |
| 7 | 8 | -7.787878 | 0.4782337 |
| 8 | 8 | -9.937115 | 0.5209629 |
| 9 | 7 | -12.21638 | 0.586403 |
| 10 | 8 | -16.68586 | 0.6532578 |

尚不能核实/不作结论：32秒包含验证的说法；所有历史microbatch的距离分布；旧验证集triplet loss（原日志没有）；Fs/Fd的统计独立或物理照明分解；“Triplet为0导致精度差”；新公式/gate/增大batch会提高PSNR。capture group_id没有经核实的映射，保持空白，不从Scene编号猜相机/系列。

## 2. 环境、数据与保护

- 实际解释器：`C:\software\anaconda3\envs\testmodel\python.exe`；Python 3.9.25，Torch 2.5.1+cu121，torchvision 0.20.1+cu121，CUDA 12.1，cuDNN 90100，驱动 596.08。
- GPU NVIDIA GeForce RTX 4060 Laptop GPU，实际总显存8585216000 bytes（8188 MiB）；不是系统16GB内存。CUDA矩阵/卷积反向以及真实AMP训练已通过。无升级、卸载或重装CUDA。
- 新阶段只审计MILL train300对/30 content和validation120对/12 content；各10个level。420条记录的input/GT哈希、完整content/level/group与旧审计相同，无train/validation内容或GT像素重复。test/LOL图像本轮不参加开发审计/训练/选择。
- 已核对旧outputs下65个权重、CSV、日志、event、JSON/YAML文件，SHA256及长度均未变；其他版本目录、原图、用户文档和Git记录未删除。
- 修改前快照：`E:\结构亮度结构解耦\V1新版无去噪\reports\condition_ablation_before\20261021T012615Z/source.zip`，39个源码/配置文件，附source_manifest.json、原未提交diff和protected.json。
- 恢复方法：先把source.zip解压到空目录，按source_manifest逐文件比对恢复源码；结合condition_implementation.diff处理新增文件/测试。不需恢复/删除data或outputs，不能用reset/clean清理科学记录。根仓库中先前移动目录造成的删除/未跟踪状态不是本任务删除文件。

## 3. 实际修改与模型/损失

以下清单由快照和当前文件自动比较生成，删除文件为0：

| 文件 | 操作 |
|---|---|
| `models/sddnet_v1.py` | modified |
| `losses/mining.py` | added |
| `losses/sdd_loss.py` | modified |
| `datasets/mill_triplet_dataset.py` | modified |
| `datasets/pk_dataset.py` | added |
| `engine/condition_diagnostics.py` | added |
| `engine/diagnostics.py` | modified |
| `engine/evaluation.py` | modified |
| `engine/profiling.py` | modified |
| `engine/reporting.py` | modified |
| `engine/smoke.py` | modified |
| `engine/training.py` | modified |
| `utils/ablation_metrics.py` | added |
| `utils/experiment.py` | modified |
| `utils/runtime.py` | modified |
| `scripts/condition_ablation.py` | added |
| `scripts/prepare_data.py` | modified |
| `scripts/run_pipeline.py` | modified |
| `configs/ablations/A0.yaml` | added |
| `configs/ablations/A1.yaml` | added |
| `configs/ablations/A1_init.yaml` | added |
| `configs/ablations/A1_PK_random.yaml` | added |
| `configs/ablations/A2.yaml` | added |
| `configs/ablations/A2_triplet.yaml` | added |
| `configs/ablations/A3.yaml` | added |
| `configs/ablations/A_batch16.yaml` | added |
| `configs/ablations/A_batch18.yaml` | added |
| `configs/ablations/base.yaml` | added |
| `tests/test_condition_ablation.py` | added |
| `tests/test_condition_pipeline.py` | added |
| `tests/test_training_regressions.py` | modified |
| `README.md` | modified |
| `.gitignore` | added |

职责：models中唯一ConditionalAffine和SDDNetV1实现模式/gate/私有encode-decode；losses/mining只负责合法索引选择；datasets/pk复用原MILL解析与同步变换；training复用唯一优化器/循环，记录暴露/距离/覆盖；condition_diagnostics执行固定条件替换；condition_ablation提供matrix/memory-check/diagnose/compare。原setup_env.bat和简洁requirements继续可用，未引入新依赖。

bounded精确保留：`f*(1+sigmoid(rho)*tanh(gamma_raw))+sigmoid(rho)*tanh(beta_raw)`。standard：`(1+delta_gamma)*f+beta`，没有rho/tanh/sigmoid gain、LN或硬clamp。c预测头自己的Sigmoid保留。
A0/A1的local std均.001；只有A1_init改为.003；global末层、RGB残差头、照度末层维持Normal(0,.001)、bias0。standard gamma近1、beta近0，非所有末层全零。通道/残差深度不变；bounded参数12270696，standard参数12270692，差4个原rho标量。旧best在bounded下strict加载，和保存的旧源码输出Fs/Fd/c/enhanced逐值完全一致。
gate默认关闭；开启统一用预测c，`g=.25+.75*(1-c_hat)`，`y=x+g*delta`；不detach，不用真实c，不强制level10恒等，也不限制delta只能为正。
损失仍只有：`L_rec=L1(y_raw,GT)+.2*(1-SSIM(clamp(y_raw),GT))`；`L_total=L_rec+.1*L_scene+.05*L_illum+lambda_cons*L_cons`。Triplet是normalized GAP(Fs)的未平方L2，margin.2、eps1e-6；illum L1使用float32[N,1]、(level-1)/9；空间cos一致性仅同content同步几何，5个exposure epoch线性升至.02。没有强制敏感、正交、对抗或Fd无界分离loss。
选择距离detach，但最终triplet用原始embedding反传。半难优先最近合法d_ap<d_an<d_ap+.2；无半难有hard则最近hard；全easy允许0；无合法负例跳过并计数，绝不把同content错误标负。已知group也排除，同level优先。每anchor只有1个候选时记录该事实。
PK P=4/K=2使用相同level集合，各content内部crop/flip/rot同步，low/GT始终同步。原triplet与PK的Fs anchor覆盖/候选池不同，不能把A1→PK的变化归为FiLM收益；miner仅比较同PK的random vs semihard。
原scheduler仍在完整epoch验证/诊断后调用一次；新日志记录其对应optimizer update，resume不重置。没有悄悄改成按microbatch或每次update重启cosine。PK epoch定义为900次暴露；若未来改为update-based调度，应对全部候选另立统一协议。

## 4. 实测显存、短测与测试

最终全套：**82 PASS / 0 FAIL / 0 SKIP**，详见condition_release_tests.xml。覆盖尺寸/pad/crop、公式/初始化、gate条件源、合法/不足/easy负例、固定Fs/skip、derangement、梯度、尾窗、checkpoint hash、PK续训CLI、日志lr/AMP/频率、数据泄漏和vs-control匹配。仅2条既有torchvision预训练参数弃用警告。
首次75/76通过，失败是旧测试夹具截断所有range，已改为仅截断epoch循环；之后全套通过。实际check-only曾因GBK打印UTF-8子进程结果失败，修复编码后真实入口64 PASS/15 deselected（当时测试集79例）；故障文件保存在check_cli_attempt1。最终新增用例包含于82例验收。没有隐藏失败记录。

| 最终真实CUDA AMP显存探针 | patch | 物理图×累积 | 有效图/完整更新 | peak allocated bytes | peak reserved bytes | 状态 |
|---|---:|---|---:|---:|---:|---|
| A2 | 256 | 8×2 | 16 | 2007971840 | 2298478592 | PASS |
| A_batch18 | 256 | 3×6 | 18 | 848896000 | 941621248 | PASS |

探针均为可丢弃模型的一次成功更新，原始/最终测量分别保留，和主smoke预算分开。两种配置均未OOM，未降patch、未瘦身。16/18不是独立图数或triplet候选池大小；nominal PK候选3、尾批P=2时为1，累积不共享负例。

主smoke：A2/seed42/standard/local.001/PK4×2/semihard/gate关闭，**20次成功更新，20次尝试，0次AMP跳步**；40个microbatch，320次图像暴露，覆盖30个content、195个独立low文件，重复125次。
训练循环显式CUDA同步计时11.079094秒，不包含前后诊断、6张整图验证和写文件；峰值allocated=2119248896 bytes（1.9737 GiB）。这不是20轮，更不是100轮。

平均训练loss（40个microbatch按图像数加权；计数另用逐triplet总表）：

| 项 | 实测 |
|---|---:|
| total | 0.1751752 |
| rec | 0.1562176 |
| l1 | 0.1186678 |
| ssim_loss | 0.1877492 |
| scene | 0.04389214 |
| illumination | 0.2910411 |
| cons | 0.0233835 |
| weighted_scene | 0.004389214 |
| weighted_illum | 0.01455205 |
| weighted_cons | 1.632914e-05 |
| lambda_cons | 0.0006902655 |

主smoke mining总数：`{'triplet_anchor_count': 320, 'triplet_valid_count': 320, 'triplet_skipped': 0, 'semihard_selected': 129, 'hard_fallback': 6, 'active_triplet_ratio': 0.421875, 'valid_negative_count': 3.0, 'd_ap': 0.04564090081657923, 'd_ap_p10': 0.005260116280987859, 'd_ap_median': 0.020937385968863964, 'd_ap_p90': 0.12383233979344371, 'd_an': 0.3252008073264733, 'd_an_p10': 0.08285568729043007, 'd_an_median': 0.2894698530435562, 'd_an_p90': 0.6844495654106147}`。active非零或gate小不作为成功标准。
limited validation：首6张整图，PSNR_RGB=18.98411、SSIM_RGB=0.7095657。逐图input基线和逐level在smoke/validation_per_*.csv；未覆盖level写count0/NA。不据这6张或crop panel挑正式权重。
新权重smoke/last.pth：arch_revision=v1_clean、run_kind=smoke、completed_epoch=0、optimizer_updates20，保存architecture/config hash、variant、seed、数据指纹及原有optimizer/scheduler/scaler/RNG/sampler。真实CUDA只读恢复一致，额外更新0；真实--eval-only拒绝smoke，未创建summary.csv、未改status。旧版权重只作原bounded baseline，不能strict=False续训到新variant。

## 5. 条件诊断真实结果

所有诊断为相同30张MILL validation（3个content×10级）中心128裁剪；normal与正式forward逐值一致，固定原输入/Fs/全部skip，仅替换缓存的validation donor Fd/c。random是seed42无固定点置换，batch1也真正跨图替换。fd_only/c_only标为人为不一致；joint两条件来自同一donor。neutral-affine显式gamma1/beta0并保留gate；gate-only-off/all-off分开。无权重或模型状态改写。
下表为 **correct PSNR minus replaced-condition PSNR** 和两种输出MAE；不是vs-input或vs-control。baseline73和20步pilot训练预算不同，禁止用此表比较模型优劣。

| 权重/模式 | donor分层 | raw MAE | clamped MAE | condition_quality_drop dB |
|---|---|---:|---:|---:|
| 旧正式best73 bounded/neutral_affine | none | 0.105613 | 0.1050801 | 11.58871 |
| 旧正式best73 bounded/gate_only_off | none | 0 | 0 | 0 |
| 旧正式best73 bounded/all_condition_off | none | 0.105613 | 0.1050801 | 11.58871 |
| 旧正式best73 bounded/joint | same_content_different_level | 0.06146287 | 0.06132123 | 8.31873 |
| 旧正式best73 bounded/c_only | different_content_same_level | 0.03788011 | 0.03780437 | -2.781826 |
| 旧正式best73 bounded/joint | different_content_same_level | 0.02171788 | 0.02167552 | 1.37497 |
| 旧正式best73 bounded/joint | random_derangement | 0.04587417 | 0.04578694 | 5.380689 |
| A2 20步pilot/neutral_affine | none | 0.006558043 | 0.006544979 | 0.1638438 |
| A2 20步pilot/gate_only_off | none | 0 | 0 | 0 |
| A2 20步pilot/all_condition_off | none | 0.006558043 | 0.006544979 | 0.1638438 |
| A2 20步pilot/joint | same_content_different_level | 0.0009750651 | 0.0009694493 | 0.06437634 |
| A2 20步pilot/c_only | different_content_same_level | 6.93096e-06 | 6.913815e-06 | -3.487579e-05 |
| A2 20步pilot/joint | different_content_same_level | 0.0004688428 | 0.0004657052 | 0.008392314 |
| A2 20步pilot/joint | random_derangement | 0.0006358855 | 0.0006319049 | 0.03531962 |

baseline中同level异content的c-only平均quality_drop为负，说明错误条件可能提高质量；没有通过挑样本隐藏。强敏感不等于条件正确/退化纯净。pilot c-only MAE很小是20步阶段的真实结果，不能据此判定长期训练失败，更不增加强制敏感loss。gate关闭时gate-only-off差值0完全正常。
affine_statistics.csv给每图每层有效gamma=1+offset、beta的mean/std/p99/max/abs_p99/abs_max和调制增量/feature RMS（分母下限1e-6）；legacy diagnostics.csv中旧gamma字段仍是offset，不能与有效gamma混读。下面仅摘初始化首图bottleneck层，完整统计以CSV为准：

| 阶段 | gamma mean/std/p99/max | beta mean/std/p99/max | 调制增量/feature RMS |
|---|---|---|
| A2初始化 | 0.9996477/0.003935797/1.007823/1.013915 | -9.884051e-05/0.003998092/0.008877988/0.01968811 | 0.04335327 |
| A2训练20步 | 0.9908566/0.02026241/1.041905/1.072746 | 0.001250431/0.01953003/0.03332646/0.04418594 | 0.5730994 |

重建单项FP32梯度来自独立autograd.grad图，未混入triplet/illum辅助梯度；JSON在各诊断目录reconstruction_gradients.json。正常训练总梯度按原--grad-norm-every采样，此20步默认100没有样本，不伪记0。
validation的triplet_probe.csv是固定30图panel、每anchor通常2个同level合法负例；它不冒充训练P4池的3个候选。输入/Fs/skip不变、诊断后恢复、GT/level不进forward均有自动化覆盖。

## 6. 可复现配置与预算

配置矩阵：reports/ablation_matrix.csv，完整resolved配置：outputs/condition_ablation/A2_seed42/config_resolved.yaml。A0→A1只改公式；A1_init只改local初始化；A2_triplet明确只有1候选；A1_PK_random→A2只改miner；A2→A3只开预测gate；A_batch另列。
A0/A1 triplet：900暴露/epoch、150次尝试；PK8×2：900暴露、113 microbatch、57次尝试（尾窗/尾批按真实样本数加权）；triplet3×6：900暴露、50次尝试。100轮名义暴露均90000，尝试更新分别15000/5700/5000，实际成功数须扣AMP跳步。
等5000次满窗口更新的理论暴露：6图批=30000、16图批=80000、18图批=90000；这是满窗算术，不能当自然epoch尾窗或AMP跳步后的实际预算。精确等更新比较必须累计实际日志，现阶段未做该正式实验，不把更新数差异归因为batch或FiLM。
100轮默认、验证频率和原cosine时机保留，无提前停止。仅一个seed42验收；三seed复验、A1_init/A3长期训练、等更新消融、离线线性probe和oracle全部NA（未执行）。无预期PSNR提升承诺。

## 7. 已实跑/已回归与未执行命令

在Anaconda Prompt：

```bat
conda activate testmodel
cd /d "E:\结构亮度结构解耦\V1新版无去噪"
python -m pytest -q
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --check-only
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --smoke-test
python -m scripts.condition_ablation memory-check --config configs/ablations/A2.yaml --output reports/memory_new_pk.json
python -m scripts.condition_ablation memory-check --config configs/ablations/A_batch18.yaml --output reports/memory_new_triplet.json
python -m scripts.condition_ablation diagnose --config configs/sddnet_v1.yaml --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth --output reports/condition_baseline_new
python -m scripts.condition_ablation diagnose --config configs/ablations/A2.yaml --checkpoint outputs/condition_ablation/A2_seed42/smoke/last.pth --allow-smoke --output reports/condition_smoke_new
```

上述模式均已实跑；memory/diagnostic输出需用新名字。smoke重跑写独立attempt子目录，不是formal resume。本次main smoke仅一轮20-update验收，未重复训练正式模型。
下面正式命令只完成入口/临时CPU短程resume等回归验证，**没有执行100轮和完整四数据集测试**；必须在对应正式checkpoint存在之后续训/评测，不能换成smoke文件：

```bat
python -m scripts.run_pipeline --config configs/ablations/A0.yaml
python -m scripts.run_pipeline --config configs/ablations/A1.yaml
python -m scripts.run_pipeline --config configs/ablations/A2.yaml
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --resume outputs/condition_ablation/A2_seed42/checkpoints/last.pth
python -m scripts.run_pipeline --config configs/ablations/A2.yaml --eval-only --checkpoint outputs/condition_ablation/A2_seed42/checkpoints/best.pth
python -m scripts.condition_ablation compare --candidate outputs/condition_ablation/A1_seed42/metrics/per_image.csv --control outputs/condition_ablation/A0_seed42/metrics/per_image.csv --output reports/A1_vs_A0_final
```

compare已用临时fixture实际调用验证；真实vs-control为NA，因为没有同预算训练完成的新候选/控制评测。该入口校验完整图像集合/split/input指标，给独立delta_psnr_vs_control和场景聚类bootstrap，不读取模型调参。
最终指标协议保留FP32整图batch1、外层clamp、PNG量化前计算、PSNR_RGB、SSIM窗11/sigma1.5/data_range1、LPIPS_Alex正确范围；没有GT对齐/隐式resize/tiles。单scene或缺指标明确标记；完美匹配PSNR按原协议inf，非有限差值的CI不伪造。
LOL-dev未建立/未执行；eval15及LOLv2以往结果已经被查看，无法证明从未参与人工选择，不能称未触碰测试集。本轮仅MILL train/validation开发，不反复用这些测试集调参。

## 8. 交付路径与未完成项

- 报告：reports/condition_ablation_audit.md；实现diff：reports/condition_implementation.diff；源码快照：reports/condition_ablation_before/。
- 真实环境/数据审计/配置：outputs/condition_ablation/A2_seed42/{environment.json,environment_report.txt,config_resolved.yaml,data_audit.json,metadata/mill.csv}。
- 短测权重：outputs/condition_ablation/A2_seed42/smoke/last.pth（仅smoke）。
- 真实训练日志：outputs/condition_ablation/A2_seed42/smoke.log；smoke/{updates.csv,triplets.csv,coverage.json,smoke_results.json,validation_per_image.csv,validation_per_level.csv}。
- 初始/训练后条件CSV：smoke/{initial_conditions,trained_conditions}/{condition_per_image.csv,condition_summary.csv,affine_statistics.csv,triplet_probe.csv,fixed_validation_manifest.json,reconstruction_gradients.json}。
- baseline条件CSV：reports/condition_baseline_best73/；从新权重重新加载复验CSV：reports/condition_smoke20_recheck/。
- 单元测试：reports/condition_release_tests.xml；只读状态/保护验证：reports/condition_delivery_verified.json；正式评测拒绝记录：reports/real_smoke_formal_rejection.log。
- 显存与实际配置：reports/memory_pk8_acc2_final{.json,.config.json}、memory_triplet3_acc6_final{.json,.config.json}；初始测量没有删除。
- 未执行：全部100轮消融、三seed正式复验、正式best的新四数据集指标/LPIPS和vs-control质量结论；正式best.pth目前NA，理由是只完成20步验收。没有将初始随机网络或短测数值当成正式成绩。
- 无需用户补充即可运行已给出的正式命令，但本次遵守边界停止在验收。没有终止其他训练；验收结束无本任务留置的后台训练。
