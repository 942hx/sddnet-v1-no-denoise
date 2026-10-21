# Codex 执行任务：仅整理「V1新版无去噪」，保留 A0 best88 并同步现有 Git 仓库

## 0. 任务性质与唯一修改范围

这是一项**工程整理、等价性验收和 Git 同步**任务，不是模型研究任务。

唯一允许创建、修改、移动、删除项目文件的位置：

```text
E:\结构亮度结构解耦\V1新版无去噪\
```

以下称其为 ROOT。开始时读取实际目录并确认规范化绝对路径与 ROOT 相同，不以当前终端目录或文件夹显示名称代替校验。

- 不操作 ROOT 的兄弟目录，例如「V1旧版无去噪」「V1旧版和V1新版对比」及其他版本。
- 不修改父目录中的源码、配置、README、数据、压缩包或其他用户文件；不把清理对象搬到父目录或其他版本中。
- 不移动、删除或改写任何原始数据集。ROOT/data 若存在，原地保护；配置引用 ROOT 外部共享数据时，只读，不移动或复制它们。
- 不重装、升级或修改 testmodel 环境；仅使用其现有 Python。不要修改全局 Git 配置、系统环境变量或外部缓存。需要临时文件、测试输出、缓存时，显式定向到 ROOT 内。
- 不启动正式训练、续训、重新选权重或模型调参；不自动重新跑完整四集评测。
- 不新增去噪块、小波、归一化、gate、FiLM 变体、损失项或数据增强。
- 可以删除已经确认无关的文件，不要保留一堆 archive/old/backup/final2。不要为被删除的 A1/A2 等实验再制造全量备份包。A0 原始证据、权重和可复现材料不属于垃圾。
- 如果发现本目录有训练/评估/写日志进程正在运行，暂停有冲突的整理，不杀进程。
- Git 目录、worktree 或 submodule 的范围另按第9节处理，不能假设“在 ROOT 执行 git”就一定只操作 ROOT。

所有批量文件操作先形成明确清单，再逐路径校验。不得只用字符串前缀判断包含关系；拒绝通过 `..`、符号链接、Windows junction/reparse point 越界。不要沿链接递归删除；删除工具不得跟随链接。ROOT 本身和 data、.git、凭据文件等保护项永远不进入删除列表。

## 1. 唯一保留的发布候选

用户所说的「A088」统一解释为：

```text
SDDNet-v1 / A0 / seed42 / best epoch 88
arch_revision = v1_clean
FiLM = bounded
residual_gate = false
无独立末端去噪、无小波
```

不是 epoch83、不是旧 best73、不是 A1 best53、不是 A2 smoke，也不是只训练了88轮。A0正式训练完成100轮，最佳验证权重在第88轮。

冻结的 best checkpoint 文件 SHA256：

```text
a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083
```

优先从现有冻结记录、正式 A0 run 和报告交叉定位。已知历史线索：

```text
outputs/condition_ablation/A0_seed42/
outputs/A0_frozen_review_20261021_150830/
reports/formal_pair_preflight/
```

这些是查找线索，不是可以直接整体删/移的目录清单。现场存在的 frozen_selection、checkpoint metadata 和哈希是最终依据。

如果找不到与上述 SHA256 完全一致的 checkpoint，或存在有冲突的冻结记录：停止破坏性清理和 Git 提交，列出问题。不得挑一个指标接近的 best.pth 替代。

最终发布别名可以是 `A0_best88`，但不要把 checkpoint 内部的 `variant=A0`、arch_revision、epoch、optimizer/RNG 等元数据改写为新名称。

## 2. 先识别保留内容，后删除

建立临时清单：相对路径、类型、文件大小、SHA256、操作（保留/移动/修改/删除）、新位置、理由。文件扫描只在 ROOT 内，不扫描数据目录之外的无关工作区。

### 必须保留

1. 上述原始 best88 checkpoint，保持文件字节不变。移动可用原子rename或复制-校验-删除源文件，不允许 torch.load 后再 torch.save 来“整理”它。
2. 同一 A0 正式100轮的训练记录：history、train log、已有逐更新统计、TensorBoard事件、成功更新/AMP跳步记录、原始 resolved config、环境记录、数据划分/配对manifest、seed和数据指纹。
3. 同一 A0 run 的最终 last checkpoint 若保存完整续训状态，保留一份并明确命名为 `last_epoch100.pth`，仅作为训练历史/恢复状态，不作为默认推理权重。不要把它当成另一个需要发布的模型。
4. A0 best88 的完整 validation 逐图、逐等级、逐content、summary、曲线与选模证据。
5. 冻结后四数据集的全部 A0 测试记录：逐图CSV、summary、MILL逐level/content、input baseline、LPIPS状态与协议、冻结记录、SHA、评估配置、执行命令、状态与失败清单、增强/对照预览、测试报告。
6. A0 的照明诊断、固定条件面板、正常/常数/oracle诊断记录，以及该模型相关的测试与保护验证记录。oracle结果只能标为诊断，不混入正式指标。
7. A0 所依赖的许可、第三方归属信息、数据说明和必要的预训练指标缓存。LPIPS AlexNet 的缓存是指标依赖，不应当误认为别的模型实验。
8. 整理后支持 A0 的训练、评估、推理、数据校验、checkpoint读取、指标计算和必要自动化测试。

### 混合报告的处理

A0/A1对照报告、合并history、final_review.md等可能同时含 A0 与 A1：

- 只要其中包含 A0 选模或评估证据，原文件就属于 A0 相关记录，完整保留到 provenance/original_reports 或相应证据目录。
- 不要因为文件名带 A1 就删除整个文件；不要截掉不利结果或重写原CSV。
- 可以另生成标注来源的 A0-only 视图，明确是派生文件，不代替原始记录。
- 不因此保留完整 A1 模型实现、A1 权重、A2 smoke和其他独立实验目录。

### 可删除

在完成依赖检查、A0材料迁移和验收后，删除：

- A1、A1_init、A1_PK_random、A2、A2_triplet、A3、A_batch等不再使用的活动配置、批处理入口、专用源码分支与独立权重/实验输出；其中包含A0原始证据的混合文件先保留。
- 旧best73的独立运行产物、与本次A0无关的smoke、重复压缩包、临时导出、过时任务提示词、纯历史备份、调试垃圾和缓存文件；先确认不是唯一来源或A0依赖。
- A0中间周期checkpoint可以删除，但必须先确认不是best88、不是唯一last完整训练状态，也没有不可替代的关联测试；不要删除其训练/测试记录。
- 已失去调用方的PK/半难挖掘/standard/gate等实验代码与测试；不能只凭文本搜索无命中就判定无用，还要检查配置、动态导入、CLI和checkpoint兼容性。
- 源码中的注释掉旧方案、无效分支、重复工具和空占位。

用途不明、可能为用户研究笔记、许可证、外部来源或唯一A0证据的文件，不盲删；说明不确定性。不要把必要的测试、数据审计、断点读取和错误处理当成“无关代码”。

## 3. 将 A0 的结果集中到一个目录

采用以下布局，允许现有数据的内部子路径保持原样，以少改原记录为优先：

```text
V1新版无去噪/
├── README.md
├── requirements.txt
├── requirements-dev.txt             # 仅在确实需要区分测试依赖时存在
├── pyproject.toml                   # 简洁的测试/格式/包元数据配置，按实际需要
├── .gitignore
├── .gitattributes
├── configs/
│   └── a0.yaml                     # 唯一活动配置，自包含或仅必要公共项
├── models/                         # 唯一 A0 模型实现
├── datasets/                       # MILL triplet与LOL配对读取
├── losses/                         # 唯一A0实际训练目标
├── engine/                         # 共享训练、评估、推理逻辑
├── utils/                          # 真正复用的少量工具
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── verify_release.py
├── tests/                          # A0相关回归测试
├── docs/
│   ├── data.md
│   └── reproduction.md
├── results/
│   └── A0_best88/
│       ├── README.md               # 权重/记录索引与历史路径映射说明
│       ├── checkpoints/
│       │   ├── best.pth            # 原始冻结文件，SHA必须相同
│       │   └── last_epoch100.pth   # 如原来存在，保留完整训练状态
│       ├── training/               # A0原始训练日志、history、event、配置
│       ├── validation/             # 完整验证记录
│       ├── evaluation/             # 四集原始结果、逐图指标与预览
│       ├── diagnostics/            # A0照明/条件诊断
│       ├── provenance/             # 冻结记录、环境、hash、混合原报告等
│       └── maintenance/            # 本次迁移清单、等价性测试、验收说明
├── data/                           # 如原来存在：原样保护，不上传数据
├── cache/                          # 仅必要的现有指标缓存，本地保留不上传
└── outputs/                        # 仅供将来的新运行；本次不制造新训练
```

不要为了凑目录而创建空文件/多层框架。保留现有 models/datasets/losses/engine 分层，避免无必要改成全新框架导致checkpoint兼容问题。

根目录不保留「new、old、final2、backup」等并行代码工程。原始历史路径可以继续出现在不可改写的证据文件中；使用新索引与 path_map.csv 映射，不全局替换原始CSV/日志中的路径。

文件去重只允许按完整SHA256确认字节相同、且映射中记录所有原路径；禁止仅凭同名、大小或“看起来重复”删除。迁移期间对A0原文件先验证新位置哈希，再删除旧位置。对于大量预览/日志，所有原有A0记录都要保留，不能只保留README中的几个汇总数字。

必要时使用 ROOT/.cleanup_work 作短期等价性参考和迁移中间区；成功验收后清空，不留下整个旧工程备份。失败时保留未完成证据并报告，不能假装成功清理。

## 4. 清理代码，但严格保持 A0 数值行为

这次是行为等价重构，不是设计新的“A088网络”。

必须保持：

- SDDNetV1及其参数名称、state_dict key、张量尺寸、计算顺序和参数量与原A0兼容。
- `arch_revision=v1_clean`、bounded公式、所有原有rho，不能将“删除standard支持”误操作为删除A0的调制参数。
- 两通道先验、共享编码器、结构/退化分支、照明头、瓶颈与多尺度skip调制、残差RGB重建均不改。
- 不改padding位置、插值设置、算子顺序、初始化随机数消耗顺序、训练增强或loss细节。
- 不把模型输出提前clamp；保持评估外层clamp和现有浮点指标协议。
- 只接收低光RGB；GT和真实等级不进入正常forward。
- 不改变原训练100轮、seed42、patch256、triplet microbatch1=3图、accumulation2=6图/完整更新及优化器/scheduler/AMP/尾窗行为。
- 损失以实际A0 resolved配置为准：保留原重建、场景Triplet、照明L1和已使用的空间一致性及warmup；不要恢复成过时的三项损失版本。

只移除实际不使用的standard/gate/PK/miner选择。必要的A0-only私有诊断接口可以保留，不得暴露真实照明条件到普通推理接口。

可以保留很薄的兼容解析层用于读取原 A0 checkpoint/config，比如校验 `affine_mode=bounded`、`residual_gate=false`，但不能悄悄忽略错误的variant或以 `strict=False` 掩盖差异。

原checkpoint文件本体不改。活动配置中的输出路径改为项目相对路径；原resolved配置保持原字节。路径迁移记录到sidecar，不能直接修改checkpoint中的历史路径或哈希。

如果历史续训校验将路径哈希与架构哈希混在一起，保留显式的路径迁移映射并检查训练语义不变，不能关掉全部兼容校验。只从完整last恢复，不把best或smoke当成已完成100轮的last。

## 5. 简洁、可运行的入口

不要复制多套训练循环。scripts只是薄CLI，调用共享engine。

实现并实际验证帮助与参数检查：

```text
python -m scripts.train --config configs/a0.yaml --output outputs/<NEW_RUN>
python -m scripts.train --config configs/a0.yaml --output outputs/<RUN> --resume <FULL_LAST_CHECKPOINT>
python -m scripts.evaluate --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth --dataset mill --output outputs/<NEW_EVAL>
python -m scripts.evaluate --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth --dataset all --output outputs/<NEW_EVAL_ALL>
python -m scripts.infer --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth --input <IMAGE_OR_DIR> --output outputs/<NEW_INFER>
python -m scripts.verify_release --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth
```

以上是目标接口，本轮只验证和小规模推理，不运行正式train，不因检查命令触发完整四集测试。

- 训练默认仅MILL train/validation，不自动执行test或LOL。
- evaluate显式选数据集，all也是显式操作；各数据集分别统计。
- 输出目录不能指向被冻结的 `results/A0_best88`，不覆盖历史结果；新目录已经有结果则拒绝，除非明确的同run resume。
- CLI错误清晰，Windows中文/空格路径可用；保留 `if __name__ == '__main__'` 等必要入口保护。
- 真实数据根路径从配置或显式覆盖读取，不能要求修改源码中的硬编码路径。
- 旧run_pipeline、formal_pair、condition_ablation、freeze_review等入口在其功能被A0-only入口/诊断覆盖后删除；必要兼容保留不超过薄适配层，不能重复实现。
- 不新增只有说明没有实现的命令，不将本轮没有运行的命令写成“已实测”。

## 6. 文档应具有研究代码发布的基本质量

README写清：

1. 项目是 `SDDNet-v1 (A0, best epoch 88)`，现阶段研究版本，无独立末端去噪。
2. 真实环境与安装要求：据现有环境报告填写，不猜版本、不重装环境。给出未来新环境的安装说明，但本轮不执行安装。
3. 数据目录、MILL按场景/等级配对、LOL配对规则、已排除样本的原始依据；不要更改既有划分。
4. 默认权重路径、完整SHA256、验证选模依据、训练100轮与best88的区别。
5. 训练、继续同run、单图/文件夹推理、MILL和LOL评测命令；严格区分训练/验证/测试。
6. 指标定义与精度：沿原FP32整图batch1、外层clamp、量化前计算、PSNR_RGB逐图均值、SSIM和LPIPS既有实现；不能混写PSNR_L、GT均值对齐或暗中resize/tile/TTA。
7. 同一冻结权重的真实结果与局限，不称“顶刊官方代码”、不声称SOTA/物理解耦已证明。
8. 全部记录位于results/A0_best88；数据、权重、记录哪些随Git提供、哪些只保留本地，准确列出。

以下已冻结指标可作核对，正式展示仍从原始记录读取，不手工修改CSV来匹配：

| Split | N | PSNR_RGB | SSIM_RGB | LPIPS_Alex |
|---|---:|---:|---:|---:|
| MILL validation | 120 | 33.16664658589749 | 0.9364022394021352 | 未在该表评测则NA |
| MILL test | 79 | 17.5628851837581 | 0.7783799616596366 | 0.11323790759131123 |
| LOLv1 test | 15 | 16.338013063035216 | 0.7100635727246603 | 0.31309984227021537 |
| LOLv2 Real test | 100 | 20.23149322186524 | 0.7679075080156327 | 0.2760451349616051 |
| LOLv2 Synthetic test | 100 | 14.904456186727648 | 0.6492352271080017 | 0.3283244425151497 |

必须保留局限：MILL test输入PSNR=21.718134396878007，增强相对输入平均下降4.155249213119908 dB；不能隐藏该失败结果。A0优于A1的依据是当前单seed验证对照，不是全局或普适优势。

文档描述保持专业、可复现；不伪造作者、论文录用、DOI、下载地址、badge或新许可证。已有LICENSE与第三方版权声明必须保留；没有许可授权时说明待确定，不自动选MIT。

依赖文件只保留实际需要的包。环境锁定记录可放provenance，不将整个现有conda环境的所有无关包作为安装必需。不要加入与A0无关的联网服务、tracker或大框架。

## 7. 删除前必须完成的等价性验收

在删除唯一旧实现前，用旧代码+冻结best88导出小型golden结果到ROOT内临时目录。不要保存整个旧工程为长期垃圾备份。

至少检查：

1. best88的完整文件SHA256整理前后不变。
2. state_dict严格加载；key集合、shape、dtype、参数量与旧A0一致。
3. 同一设备、精度、种子、输入下，比较raw enhanced、Fs、Fd、c_hat；覆盖256×256、非8倍数尺寸、最小合法尺寸、400×600和至少一张真实MILL图。CPU FP32确定性参考优先要求逐值一致；实际CUDA差异如存在必须给出绝对/相对误差、原因与验收依据，不能默默放宽阈值。
4. 对每个实际可访问的LOL测试子集选至少一张已有配对输入，只验证整理前后等价；不据其结果调参，不覆盖正式CSV。
5. 同一张量批次的每项loss与total一致；场景/等级采样和同步增强规则不变。
6. 临时内存模型的梯度路径测试、已有A0恢复逻辑测试；不对best88执行optimizer.step，不改变原run状态。不得把CPU测试称为CUDA通过。
7. 评估协议和指标测试、越界输出梯度测试、非法输入和路径保护测试。
8. 原有A0记录的大小和SHA逐文件核对。移动不等于允许重写；formatter、换行转换、路径替换不得扫入原始证据。
9. import/CLI/help、测试集合、README相对链接和缺文件检查。没有本地数据的测试明确标记skip，不能构造假的PASS。
10. 输出路径保护：任何入口不能默默写ROOT外部、修改数据或覆盖冻结记录。

先将A0相关文件归位并通过校验，再删除确认无关的源文件/旧产物。代码替换需要的临时引用在验收后删除。失败则停止删除和push，报告真实差异，不通过重训恢复接近的指标。

## 8. 维护清单和记录完整性

`results/A0_best88/maintenance/` 只保留必要的小型说明和清单：

- `path_map.csv`：旧路径、新路径、size、SHA256、去重关系。
- `deleted_files.csv`：本次真正删除的路径、类型、大小和理由，不夸大删除数量。
- `preserved_original_manifest.csv`：全部不可改写A0原始证据的迁移前后哈希；不包含清单自身或持续更新的汇总。
- `release_verification.json`：真实测试结果、数值等价结果、数据保护与边界检查。
- `cleanup_report.md`：修改/删除/保留摘要、仍需确认的问题、最终目录和已验证命令。

不能将旧82/95/9项验收数量直接冒充整理后测试数量；报告本次实际执行的测试及对应版本。

不保留全量无关源码snapshot或A1产物压缩包。原始报告中保留的A1比较行仅作为A0选型证据，说明这一点，不能称“所有含A1字样文件全部删除”。

## 9. Git 同步：先确认仓库范围和目的地

用户希望同步到已有「V1新版无去噪」Git仓库，但本任务没有给出远程URL。必须从现有配置确认，不能把中文目录名直接拼成远程地址。

### 9.1 初始只读检查

记录并核查：

- Git工作树顶层、实际git-dir、git-common-dir（处理worktree/.git文件/submodule）。
- 当前branch、upstream、remote URL和push URL（输出中屏蔽凭据）、远程是否与本项目对应。
- 已跟踪/未跟踪/已暂存/未提交内容，以及已有但尚未push的提交。

可用 `git rev-parse --show-toplevel`、`--absolute-git-dir`、`--git-common-dir`、`git status`、`git remote -v` 等只读检查；必要时设置 `GIT_OPTIONAL_LOCKS=0` 防止初查刷新共享索引。

### 9.2 严格边界

- 若工作树根就是ROOT，且实际.git与common-dir也在ROOT内，可继续本节的提交/推送。
- 若ROOT只是父仓库子目录，或.git/common-dir在ROOT外：先完成本地目录内整理，但暂停Git写操作，报告实际Git布局并请求针对共享Git元数据/仓库的授权。不能自作主张提交父仓库、改兄弟版本、调整submodule或在ROOT中新建嵌套仓库。用户的“不要动其他文件夹”优先。
- 若没有可确认的remote/branch、存在多个可能目标、凭据失败或仓库并非用户指定项目：本地整理可完成，Git同步标记BLOCKED并准确说明所需信息，不猜地址，不新建公开仓库，不更改仓库可见性。
- 不修改全局user.name/email/credential；缺少身份就报告，不编造。

### 9.3 Git包含哪些文件

默认提交：

- A0源码、唯一活动配置、必要测试、依赖声明、README/docs、已有许可。
- A0冻结记录、文本报告、原始CSV/JSON/YAML、数据划分/文件名manifest及轻量曲线；不能用 `/results/` 一条ignore把所有测试证据隐藏掉。

默认只保留本地，不提交普通Git：

- 原始数据、完整大型checkpoint、LPIPS预训练缓存、TensorBoard大事件文件、大批原图/增强预览、临时缓存和凭据。

“不上传”不等于“删除”。这些A0相关材料仍在统一的results/A0_best88内本地保留，README和交付清单准确说明。

对权重：如果本项目已经使用并验证了可用的LFS/既有发布机制，可沿用实际适用的机制同步best88，必须确认上传可用和SHA不变。不临时改写仓库历史，不创建未经确认的新存储服务。没有可用大文件机制时，本地保留权重、Git提供SHA/路径说明，在交付中明确“checkpoint未同步”，不能谎称clone即可得到全部材料或编造下载地址。

如果原来data/cache/大文件已被Git跟踪，`.gitignore`不能代替索引清理。只有在清单批准并确认范围后，才能对ROOT内明确列出的非发布文件使用 `git rm --cached -- <paths>`；这只解除跟踪，工作树文件必须保留。禁止 `git rm -r .`、`git rm --cached -r .` 全局重置索引。

对原始A0证据，配置精确的 `.gitattributes` 禁用自动文本换行归一化（如该证据路径的 `-text`），避免Windows Git提交/检出改变原字节；这不意味着对整个仓库运行renormalize。LFS若已用，应使用兼容的局部规则。

提交前检查无token、密码、私有密钥、凭据URL或未授权的原始数据。不能为了“漂亮”修改原始证据来掩盖敏感信息；若确有敏感证据文件，本地保护、排除上传、提供明确脱敏派生版本与说明。

### 9.4 安全提交与推送

1. 只暂存经审核的ROOT范围内文件与删除。不要在父目录 `git add .` / `git add -A`。即使仓库独立，也使用明确清单并检查staged diff。
2. 记录原有未提交修改；不要通过reset/stash/clean丢掉它们。此任务已授权的整理改动可以提交；混有无法识别的用户工作时先报告。
3. 检查已有未push提交，不能无意间把不属于本任务的历史一并发布；有范围/凭据风险则暂停。
4. 确认远程仅该目标、显式branch/refspec，正常fast-forward push；先dry-run，再实际push。
5. 禁止force/force-with-lease、`+refspec`、`--mirror`、`--all`、批量tags、历史重写、filter-repo、删除远程branch/tag、自动pull-rebase和reset --hard。
6. 不为这次清理改远程名称、visibility或创建新仓库。非fast-forward、保护分支或权限错误则停止，不自动“解决”冲突覆盖远程。
7. 提交消息可用 `refactor: publish clean A0 best88 baseline`；根据实际结果描述，不写声称性能提高的commit message。
8. 提交成功不等于同步成功。push后只读检查远程目标branch的tip是否与提交一致；仅一致才报告SYNCED。失败则给本地commit、目标branch、错误与未同步项。
9. 不为了让git status表面干净删除被ignore的本地科学记录。最终报告区分已跟踪干净、预期本地保留、意外未提交三类状态。

## 10. 交付与停止条件

完成后给用户：

1. 整理后的目录树（不展开海量数据/图片）。
2. 唯一默认模型A0 best88的路径、完整SHA256、严格加载及前后输出等价结果。
3. 保留记录的位置、数量、迁移映射；删除对象类别、数量、节省空间。
4. 本次真实测试结果与尚未验证项。
5. 已验证的环境使用、训练、评估、单图推理命令；本次没有正式训练。
6. README中如实披露的测试局限，并指向原始结果，不增加性能承诺。
7. Git根目录、目标remote（不含凭据）、branch、本次commit、实际push状态；哪些大型材料仅保留本地。
8. 确认没有修改兄弟文件夹/父级工作树文件、原始data、conda环境或全局配置；不能仅凭“从ROOT启动”就声称已确认。

如果Git因父仓库边界或目标不明确而BLOCKED，要明确说“本地整理完成，Git未同步”，不要绕过限制。

任务完成即停止。不要继续训练、评估全部测试集、添加第二版模块或启动长期后台任务。
