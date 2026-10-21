# A0 best88 整理与验收报告

本报告记录 Git 共享元数据授权之前的清理验收，以下 Git 暂停状态为当时的真实情况。用户后续已授权继续同步；发布范围及后续核验说明见 [git_sync.md](git_sync.md)。

本地整理完成，Git未同步。唯一修改范围为 `E:\结构亮度结构解耦\V1新版无去噪`。没有正式训练、实际续训更新、完整评测、网络设计、环境安装或升级。

## 身份与不可变证据

- 唯一默认模型 SDDNet-v1 / A0 / seed42 / best88 / v1_clean / bounded / residual_gate=false，无独立末端去噪、无小波。
- 原始 best：`results/A0_best88/checkpoints/best.pth`，SHA256 `a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083`。
- 同 run 完整 last：`results/A0_best88/checkpoints/last_epoch100.pth`，SHA256 `e988c1051c7922b8b4b752191734018ef7799b71cbca6e50deaed4703e29f1a0`；训练完成100轮，best按MILL validation选为88轮。拒绝对它继续第101轮。
- 原始 best/last 均未重新序列化、转换或改元数据。严格加载148项state_dict，12,270,696参数。
- 原始记录1,906份，共532,276,407字节，迁移前后逐文件大小和SHA256相同；不对日志/CSV/图片/历史配置做重写，也未按同名覆盖或去重。

## 最终目录

```text
V1新版无去噪/
├── README.md
├── .gitignore / .gitattributes / .core-constraints.txt
├── requirements.txt / pytest.ini
├── configs/a0.yaml
├── models/                 唯一 A0 实现
├── datasets/ / losses/ / utils/
├── engine/                 共享训练、评估、推理、诊断与发布校验
├── scripts/                train、evaluate、infer、verify_release、prepare_data
├── tests/                  当前自动测试
├── docs/                   data.md、reproduction.md
├── data/                   原始数据，不展开
├── cache/                  已有官方 LPIPS/AlexNet 缓存
└── results/A0_best88/
    ├── checkpoints/        best.pth、last_epoch100.pth
    ├── training/ / validation/
    ├── evaluation/frozen_review/
    ├── diagnostics/
    ├── provenance/         原始混合报告、冻结/配置/环境/任务证据
    ├── maintenance/        操作、迁移、删除、哈希与验收清单
    └── README.md
```

`outputs/` 当前没有正式新run；未来CLI按显式新目录创建。本轮成功后 `.cleanup_work/` 必须删除，不保留全量旧工程备份。

## 主要迁移（相对本目录）

| 原位置 | 新位置 |
|---|---|
| outputs/condition_ablation/A0_seed42/checkpoints/best.pth | results/A0_best88/checkpoints/best.pth |
| 同run checkpoints/last.pth | results/A0_best88/checkpoints/last_epoch100.pth |
| 同run history/log/config/environment/metadata等 | results/A0_best88/training/A0_seed42/ |
| 同run validation/、conditions/ | results/A0_best88/validation/A0_seed42/、diagnostics/training_conditions/ |
| outputs/A0_frozen_review_20261021_150830/evaluation/ | results/A0_best88/evaluation/frozen_review/ |
| 冻结validation_review/、illumination/、conditions/ | validation/frozen_review/、diagnostics/illumination/、diagnostics/frozen_conditions/ |
| 冻结身份/保护记录和原始报告 | provenance/frozen_review/、provenance/original_reports/ |
| outputs/A1_best53_LOL_compare_20261021_153445/ | provenance/original_reports/同名目录（完整混合A0/A1证据） |
| reports/A1_vs_A0_validation_formal/、formal_pair_preflight/ | provenance/original_reports/同名目录 |
| formal_pair_smoke/A0的非权重记录 | provenance/test_evidence/formal_pair_smoke_A0/ |
| 原根README、CODEX任务文件、完整环境锁定 | provenance/original_reports/README.before_release.md、task_definitions/、environment/ |

逐文件原路径、新路径、大小和SHA见 [path_map.csv](path_map.csv)；双向复核见 [preserved_original_manifest.csv](preserved_original_manifest.csv)。表中短路径均在 `results/A0_best88/` 下。包含A0选择/测试的混合A1报告完整保留，其文件名/历史配置不作为活动模型。

## 修改与运行边界

保留 models/datasets/losses/engine/utils 模块结构和 checkpoint 模块名。仅移除不使用的 A1/A2/A3、standard、PK、random/semihard miner 和 residual gate 算法分支。原A0固定triplet统计中的零值兼容字段、只接收合法A0值的解析和必要私有条件诊断保留；没有第二套模型或训练循环。

配置唯一为 `configs/a0.yaml`，从真实A0 resolved参数建立自包含配置，保留数值contract/hash；修正数据审计默认配置。薄CLI共享engine：新训练只MILL开发集，evaluate必须显式选择数据集，infer只推理，verify_release只读核验。训练/评估/推理/独立数据审计均拒绝保护目录、ROOT外部、`..`、link/junction和已有新输出；同run续训还检查完整last身份与完成轮次。

当前说明及所有可用CLI路径已更新；生成报告不再引用已删除的smoke/profile入口。原历史路径不批量替换，由path_map解释。依赖仍使用testmodel，必要LPIPS缓存原地保留。`.gitignore`仅在本目录修正未来可提交文本的选择，`.gitattributes`对原证据禁用换行归一化；没有改共享索引。实际新建/修改/保留源码列表见 [source_changes.csv](source_changes.csv)。

## 删除清单与依据

验收通过后，已删除原项目5,302个核实无关文件，共5,300,025,737字节（约5.30GB），移除迁移/删除后1,842个空目录。`docs/` 曾作为旧空目录删除，随后重新用于两份当前发布说明，不能把它描述为最终不存在。原始数据、已有LPIPS缓存、A0 best/正式last和1,906份原始证据均未列入删除。

| 原文件判定依据 | 实际文件数 |
|---|---:|
| 可再生Python/pytest缓存 | 50 |
| 唯一活动配置已由冻结resolved配置自包含生成configs/a0.yaml，完整训练契约已逐值核对相同 | 1 |
| 已沿调用链核实的A1/A2/PK/standard/gate专用配置、入口、测试或被A0薄CLI替代的旧管线 | 29 |
| 旧best73及前一轮整理/日志修复独立历史资料，与本次A0正式run无关 | 545 |
| 仅旧best73校准入口依赖，A0固定条件及重建梯度诊断已有共享实现 | 1 |
| 历史单元测试合成夹具；测试XML保留，不是正式实验数据 | 255 |
| 独立A1/A2、旧best73或正式前smoke产物；与冻结A0 run不同，A0混合证据已另列保留 | 1676 |
| 历史合成测试工作区、内存PK实验、纯源码快照或已由原始正式报告保留信息的生成工具 | 2745 |

具体包括独立A1/A2与PK实验、旧best73、非正式smoke权重、历史合成夹具、纯源码快照、已被薄CLI替代的旧入口、无关旧说明和再生缓存。A0正式前smoke文本/图片/环境证据已保留；其权重不是冻结best88或正式last。不存在无法归属而擅删的项目；当前无用途未确认的遗留文件。

精确实际删除路径、类型、大小、SHA及依据见 [deleted_files.csv](deleted_files.csv)。本轮临时golden/测试工作区另行逐项列在同表，原因前缀“本轮临时”；其数量/字节不混入上述5,302个原文件。原始操作计划见 [operation_plan.csv](operation_plan.csv)。

## 本次验收

- 最终自动测试：**75通过、0失败、0跳过、0收集错误**，15.67秒；2条torchvision旧参数弃用警告，不是数值/导入失败。真实Windows junction测试执行通过。
- 旧/新CPU FP32：8例（256×256、255×257、1×1、400×600、真实MILL及LOLv1/LOLv2Real/LOLv2Synthetic各1图）的raw enhanced/Fs/Fd/c_hat逐值一致，max_abs/max_relative均0；严格state_dict、初始化/RNG、每项loss、固定采样/同步增强一致。旧实现被替换之前已保存golden，成功后临时引用删除。
- CUDA：已有AMP/梯度测试通过；真实last_epoch100通过CUDA Trainer恢复，模型/optimizer/scheduler/scaler/sampler/loader generator状态核对一致，optimizer update数0。没有将CPU逐值等价称为CUDA逐值等价。
- 38份活动Python源码语法检查通过；实际模块导入均来自本目录，唯一活动配置和2个原始正式checkpoint定位通过。
- 9个CLI检查通过：4个主入口help、数据审计help、verify_release、MILL1图CPU评测、真实单图及两张合成图文件夹推理；中文/空格路径通过。有限评测显式标记limited_smoke，不当作四集完整成绩。
- 原始data全量6,137文件、1,609,115,786字节的大小/SHA/mtime及文件集合不变；原始记录1,906份大小/SHA不变，缓存SHA/mtime不变。全部写操作仅发生于本目录，经逐项真实路径校验；父级/兄弟目录写操作0，环境安装/升级0。
- 最初一次测试的13项失败为旧测试微型模型不接收A0参数及junction权限限制，已修正/实测解决；随后cmd输出GBK解码线程警告改为捕获原始bytes后消失。最终未屏蔽模型异常、未放宽strict、未跳过保护用例。
- **未验证**：100轮复训、真实续训更新、完整validation/四集重测、CUDA旧/新逐值对照、新环境安装、远程push/tip。前四项按用户小规模验收边界不执行，环境受保护，Git受父仓库边界限制。

详细输入、误差、测试nodeid、命令/退出码、数据和Git检查见 [release_verification.json](release_verification.json)。历史82/95/9等测试数量均未作为本轮结果。

## 可用命令与实测范围

从本目录运行，使用既有 `conda activate testmodel`。新的输出目录必须不存在。

```powershell
python -m scripts.verify_release --config configs/a0.yaml --checkpoint results/A0_best88/checkpoints/best.pth
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new
python -m scripts.train --config configs/a0.yaml --output outputs/A0_seed42_new --resume outputs/A0_seed42_new/checkpoints/last.pth
python -m scripts.evaluate --config configs/a0.yaml --dataset mill --device cpu --max-images 1 --output outputs/MILL_one_image_new
python -m scripts.evaluate --config configs/a0.yaml --dataset all --output outputs/four_tests_new
python -m scripts.infer --config configs/a0.yaml --input "data/LOLdataset/eval15/low/1.png" --output "outputs/单图 推理_new" --device cpu
python -m scripts.infer --config configs/a0.yaml --input "data/LOLdataset/eval15/low" --output "outputs/文件夹 推理_new" --device cpu
```

verify、MILL1图和单图实际执行；文件夹接口用两张临时合成图实际执行（未全量处理LOL目录）；训练/续训及完整all仅help、参数/保护、相关单元和状态恢复验证，未实际执行这些正式命令。新入口默认读best88，配置和checkpoint覆盖仍做同一A0兼容校验。

原始真实指标及局限见 [项目README](../../../README.md)：MILL test输入PSNR=21.718134396878007，增强均值下降4.155249213119908dB，约74.68%图像下降；单seed验证优势不代表普适/SOTA。无伪造论文、作者、下载、许可证或新实验结果。

## Git真实状态与仅本地材料

- 工作树根 `E:/结构亮度结构解耦`；git-dir/common-dir均为父级 `.git`，超出本次唯一修改范围。
- 现有 origin fetch/push：`https://github.com/942hx/sddnet-v1-no-denoise.git`；branch `新版v1无去噪`，upstream `origin/新版v1无去噪`。
- 既有HEAD `45e0a698ea9deafb8dd669442a2e11334cb99fd7`；相对本地缓存upstream ahead/behind为0/0，不代表本轮验证过实时远端tip。
- 本次commit：无；暂存：无；push：未尝试；远程tip：未验证。没有force/mirror/改历史、创建嵌套仓库或修改外部.git。既有父目录未提交修改保留，不能把旧的整理改动归为本轮提交。
- **本地整理完成，Git未同步**。根据用户范围及原任务§9.2，同步需要另行授权对父仓库共享Git元数据的写操作。
- 本轮所有材料都未上传。默认仅本地：原始data、best88/last100、LPIPS缓存、TensorBoard大事件文件、大量预览/增强/拼接图片。没有可声称已验证的大文件上传机制，checkpoint未同步。
- 局部ignore已允许将来审核源码/配置/测试/文档/原CSV/JSON/YAML/log及一个轻量历史曲线；候选文本未发现凭据模式。当前没有提交，所以不能声称这些文本已在远端。

## 临时清理终检

完成：本轮临时 420 个文件（804,588,686 字节）和 235 个空目录已按真实路径逐项删除；`.cleanup_work/` 已不存在，旧模型参考/golden/测试夹具均未留下；活动源码无 __pycache__ 或 pytest 缓存。每个实际删除项已追加至 deleted_files.csv，并与5,302个原始无关文件分别计数。
