# 下一阶段：冻结 A0、核查 A1 照明预测、评测已冻结 A0

## 0. 目的与边界

项目：`E:\结构亮度结构解耦\V1新版无去噪`
环境：现有 conda `testmodel`。
依据：用户的 `formal_pair_review_and_results.md`。

本任务是报告分析后的建议任务，不代表这些诊断和最终测试已经完成。
本次不训练、不续训、不改网络、损失、初始化、采样器或优化器，不增加去噪、小波、输出 gate、stop-gradient 或额外损失。
不要重复运行 A0/A1 100轮，也不要启动 A2/A3 或其他消融。
不要重装环境、删除旧结果、终止已有训练。若有冲突进程，报告并停止冲突操作。

决策先于测试：在 seed42、本轮训练协议和 MILL validation 下，暂选 A0 bounded 作为第一版主候选；A1 standard 保留作负向消融。选择依据是已完成的验证，而不是下面的 test 分数。

已报告：
- A0：100轮；best epoch88；validation PSNR 33.16664658589749；SSIM 0.9364022394021352；illumination MAE 0.04591539830251819。
- A1：100轮；best epoch53；validation PSNR 30.652833454756074；SSIM 0.9312802081306776；illumination MAE 0.30167775195505914。
- 两组均14995次成功更新、15000次尝试、5次AMP跳过、90000次图像暴露。
这些是需要与已有文件核对的记录，不能用作缺失指标的填充值。

## 1. 只读核对与冻结

1. 检查当前工作区及真实文件路径，不假设脚本新增了不存在的参数。
2. 读取 A0/A1 已有 resolved 配置、history、状态、best checkpoint 标识和完整 validation CSV。
3. 检查 strict 加载、run_kind=formal、variant、arch_revision、seed、epoch、数据与配置指纹。
4. 期望权重路径：
   - `outputs/condition_ablation/A0_seed42/checkpoints/best.pth`
   - `outputs/condition_ablation/A1_seed42/checkpoints/best.pth`
5. 报告中的 best SHA256：
   - A0 `a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083`
   - A1 `98c309140d1a292e5807ff8364319d1df790601345c891395177990416e3f825`
   不一致时先调查，不改写 hash，不悄悄替换权重。
6. 创建带时间戳的独立输出目录，在其中先写 `frozen_selection.json`：固定 A0、checkpoint/hash、配置、选型依据、时间、各测试集用途。输出目录不覆盖旧报告、验证CSV、状态或配置。
7. 本轮 A0/A1 未访问最终测试集，不代表这些数据历史上从未被查看。报告准确标记为“冻结本轮选型后的评估”，不能称全项目从未触碰的盲测。

## 2. 先整理现有日志，不改变训练

输出 A0/A1 对齐的100轮记录：train total/rec/illum/scene/cons（仅已有字段）、val PSNR/SSIM/illum MAE、LR、AMP跳步、best与last。
- 保留原始项与加权项的区别，缺失字段写NA。
- best53不能单独证明A1在53轮之后过拟合；结合整条history描述。
- 不根据摘要重建不存在的每步梯度或loss。
- 对A0/A1完整validation逐图CSV重新分组汇总，检查与报告一致；输出逐等级和逐content差值。
- 允许使用现有compare工具，必须分dataset/split，不混合各数据集。

## 3. A1照明预测的有限只读诊断

目的是判别“亮度回归在训练域也失准，还是主要在未见场景失准”，不是修复模型。

### 3.1 全图预测

对两组各自 best，在 MILL train 和 validation 上做 eval/FP32/no-grad 全图推理；优先复用有足够字段且checkpoint、配置、视图一致的已有结果，不重复推理。
- 不读取test来找A1的错误原因。
- 预测来自普通模型的原始 c_hat，不以真实标签或GT替代前向条件。
- 保存 `variant, checkpoint_hash, split, content_id, level, c_gt, c_hat, abs_error`；正常增强PSNR/SSIM若已有则复用。
- 按level、content输出mean/std/min/max/count/MAE。
- 计算常数0.5回归器在相同样本上的MAE；计算Pearson/Spearman（方差为0或有效样本不足写NA）。
- 不把MAE当成中间等级的预测均值。只有目标0和1且c_hat在[0,1]时，可用端点MAE推算均值。
- 保存逐图范围检查、标签解析结果以及有必要的散点/按等级均值图；不伪造线性关系。

### 3.2 一次固定面板的条件核查

复用经过测试的条件诊断实现，事先固定validation各content的level1/5/10，正常增强所用 Fs/Fd/skip 必须保持不变；固定全图或同一视图并记入manifest。
比较normal、c固定为0.5、c由对应真实等级替代三种情形；最后一种严格标记 `oracle_DIAGNOSTIC_ONLY`。
- 只有私有只读诊断接口允许替换c；普通模型输入接口、训练与正式评测始终只接收低光图。
- 先验证 normal 与普通forward一致。
- 同时报 raw/clamped output-MAE，以及 normal PSNR minus replacement PSNR；正负号说明清楚。
- oracle改善或不改善都如实报告，它不是可部署方案，不能混入正式指标，也不能单独证明因果机制。
- 不把shuffle-MAE变成训练loss，不重新优化任何参数。
- 若现有接口不能可靠冻结Fs/Fd/skip，跳过并标记原因，不做大范围重构来勉强完成。

### 3.3 可选局部梯度核查：仅在现有诊断容易复用时做

固定少量预声明train/validation样本及同一视图，在eval模式下启用autograd、FP32；不更新参数、不调用optimizer.step。
分别对 L_rec 与配置实际权重乘以 L_illum 计算 illumination_head 和 degradation_branch 的梯度范数及cosine。
- 用autograd.grad，不将辅助loss混入重建梯度；不使用AMP缩放梯度做比较。
- 零范数/未连接参数明确标记，cosine为NA，不伪填0。
- 负cosine只说明这些样本/检查点上的局部方向冲突，不据此宣称是全部性能下降的原因。
- 如果需要新增大量代码，本轮不做，注明未执行；不影响下面冻结A0的评估。

## 4. 已冻结A0的最终数据集评估

仅当权重、数据配对、指标实现和验证复算检查通过，再评测已经冻结的A0。
诊断结果难看本身不是改变选型或停止报告坏结果的理由；实现/非有限/标签错误才是停止条件。

使用同一个 A0 best88，分别评测：
- MILL test
- LOLv1 eval15
- LOLv2 Real_captured/Test
- LOLv2 Synthetic/Test

复用现有 --eval-only 入口或独立只读包装；先检查执行路径，防止覆盖历史配置/状态、启动训练、选择新checkpoint或意外运行profile。
现有入口预期形式（执行前核查实际支持）：
`python -m scripts.run_pipeline --config configs/ablations/A0.yaml --eval-only --checkpoint outputs/condition_ablation/A0_seed42/checkpoints/best.pth`

要求：
1. 严格保持现有FP32整图batch1和指标协议；记录外层clamp、颜色范围、量化前计算与GT配对方式。
2. 不做GT均值/曝光对齐，不偷偷resize，不改用tile，不用目标图调整输出。
3. 同时计算未增强输入和增强输出的PSNR_RGB/SSIM_RGB，分别报vs-input。
4. LPIPS使用已有实现和可用的正确权重；权重不可用/网络受限写NA及原因，不重装环境、不悄悄换版本或假造数值。
5. MILL另报按level/content的均值、逐图下降比例、照明MAE；LOL没有真实等级，照明MAE为NA，不制造伪标签。
6. 分别输出各数据集summary与per_image，不输出样本数加权的跨数据集“总体提升”。
7. 不运行A1/A2/A3测试来重新挑方案，不修改A0损失/配置来适配test。若A0跨场景或跨域下降，如实报告。
8. 已有历史test曾用于人工查看，这次不能宣传为绝对未触碰测试集；本轮选型没有使用这些新增结果。
9. 只有实际完成的项目标完成。出现OOM、缺数据、权重失败等先写清原因，不自动换评测协议。

## 5. 最终交付与停止

输出到本次独立目录：
- frozen_selection.json
- A0/A1 history对齐CSV及曲线（已有字段）
- 全图illumination逐图/逐level/逐content及constant baseline报告
- 有实际执行的固定条件核查结果
- 四数据集分开的最终metrics和MILL分等级结果
- final_review.md，分开写：事实、推导、待验证假设、缺失项目
- 修改清单、真实运行命令、开始/结束状态、输出位置

核对旧权重和原CSV未改写。只在新目录保存结果。
结论只允许：“当前seed42和协议下A0优于A1的validation结果”“冻结A0的跨域表现”等有数据支持的陈述。
不能声称已证明严格解耦、bounded普遍优于standard、原测试问题已解决或超过未同协议复现的CIDNet/MILL方法。
任务完成即停止。本次不启动任何训练。
