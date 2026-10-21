# A0 seed42 · best88 原始证据索引

该目录集中本次保留的 1,906 份 A0 原始文件。文件内容逐份以大小和 SHA256 校验，不改写历史 CSV 数值、日志、配置快照或 checkpoint 元数据。以下目录是现位置；原文中的旧路径通过维护清单解释。

| 位置 | 内容 |
|---|---|
| `checkpoints/best.pth` | 唯一默认 best88，SHA256 `a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083` |
| `checkpoints/last_epoch100.pth` | 同 run 完整 epoch100 状态，含续训状态；保留但不继续第101轮 |
| `training/A0_seed42/` | 原100轮历史、train.log、TensorBoard、epoch记录、采样/coverage、resolved配置、开发集manifest和环境 |
| `validation/` | 同run best验证及冻结复核、逐图/分等级、选模证据 |
| `evaluation/frozen_review/` | 同一冻结权重四集原评测，含 metrics、原图/增强/拼接预览和 final_data 审计 |
| `diagnostics/` | 训练/冻结条件诊断与照明诊断、预览和统计 |
| `provenance/frozen_review/` | 冻结身份、SHA、只读保护及环境等原记录 |
| `provenance/original_reports/` | 原始分析、曲线、历史测试结果、A0/A1混合选型/测试证据 |
| `provenance/test_evidence/formal_pair_smoke_A0/` | A0正式前smoke的文本/配置/预览证据；该smoke权重不是best88或正式last |
| `provenance/task_definitions/` | 原始任务指令，非当前运行入口 |
| `provenance/environment/` | 原环境锁定记录，非全量必装依赖 |
| `maintenance/` | 本次迁移、删除、数值与自动化验收、数据和Git边界说明 |

`provenance/original_reports/A1_best53_LOL_compare_20261021_153445/`、`A1_vs_A0_validation_formal/` 等含 A0 选型/测试证据，完整原样保留。其名称及 A1 对照行不是活动 A1 实现；历史报告中的旧命令不代表当前受支持的 CLI。当前命令以项目 [README](../../README.md) 为准。

维护入口：[原路径→新路径](maintenance/path_map.csv)、[原始证据双向哈希清单](maintenance/preserved_original_manifest.csv)、[实际删除清单](maintenance/deleted_files.csv)、[操作计划](maintenance/operation_plan.csv)、[验收 JSON](maintenance/release_verification.json)、[整理报告](maintenance/cleanup_report.md)。没有原始记录去重或覆盖；预览和TensorBoard只迁移。

冻结结果与测试局限见项目 README；MILL test 增强相对输入平均下降 4.155249213119908 dB 的失败结果完整保留。后续已获得共享 Git 元数据写入授权，审核后的文本进入发布清单，详见 [Git 同步说明](maintenance/git_sync.md)。checkpoint、所有图片、TensorBoard、约 10 MB 的原始 train.log 继续只保留本地，data 和 LPIPS 缓存仍在项目原位置。原清理报告中的 Git 暂停状态是授权前记录。
