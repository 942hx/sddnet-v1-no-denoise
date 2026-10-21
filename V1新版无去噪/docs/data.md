# 数据、配对与边界

所有原始数据保持原样，当前本地 `data/` 共 6,137 个文件。整理后的最终全量 SHA256/大小/mtime 核对见 [验收记录](../results/A0_best88/maintenance/release_verification.json)。数据没有重新复制、合并、重命名、去重或清理。

```text
data/
├── MILLs-DSLR/       train、validation、test；各含 input/gt
├── LOLdataset/      our485、eval15；各含 low/high
└── LOLv2/           Real_captured、Synthetic；各含 Train/Test、Low/Normal
```

MILL 按文件名末尾 `-1` 至 `-10` 提取照明等级，前缀完整字符串为 content_id，GT 与 content_id 配对。不能把数字前缀擅自合并为物理场景；没有已验证的 capture-series 映射时 group_id 为空。训练只读 train，验证选模只读 validation，沿用现有划分。Triplet 的 anchor/positive 来自同一 content 不同等级，negative 来自另一 content，低光/GT 同步 crop 和增强。

LOL 按低光和 GT 的文件 stem（不区分大小写）严格配对，拒绝重复 stem、缺配对或尺寸不一致，不能按排序位置凑对。LOLv1 使用 eval15，LOLv2 Real/Synthetic 分开统计 Test；本项目不使用 LOL 做训练。

唯一既有官方排除项是 MILL `test/Test13-9.png`，依据及当时是否存在记录于原始 [data_audit.json](../results/A0_best88/evaluation/frozen_review/final_data/data_audit.json) 的 `official_exclusions`（沿用 `datasets/common.py` 的既有规则）。排除评测不等于删除原始文件；本轮没有新增排除项。

开发集原审计/文件名 manifest 位于 `results/A0_best88/training/A0_seed42/` 和其中的 `metadata/mill.csv`；冻结四集审计/manifest 位于 `results/A0_best88/evaluation/frozen_review/final_data/`。原始相对路径基于配置中的对应数据根，历史绝对路径保持原字节，通过 [迁移表](../results/A0_best88/maintenance/path_map.csv) 解释。

评测 CLI 使用原冻结 test manifest，逐个校验本次读取的 input/GT SHA；`--max-images` 会标记 `limited_smoke`，不能当完整测试。训练准备仅审计 MILL train/validation，不自动读取 test/LOL。数据和输出独立；所有运行输出必须通过项目边界及受保护目录检查。

独立审计入口为 `python -m scripts.prepare_data --config configs/a0.yaml --output outputs/data_audit_new --scope development`，输出目录必须为新目录。`--scope final` 仅在用户显式要求时进行全数据审计，不训练、不评测；本轮仅验证该 CLI 的帮助、路径保护和合成数据单元测试，没有重新运行完整审计。
