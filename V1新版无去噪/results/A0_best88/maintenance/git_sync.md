# A0 best88 Git 发布范围与核验

用户已明确授权写入父仓库共享 Git 元数据，仅暂存/提交 `V1新版无去噪/` 内本次审核过的源码、配置、测试、文档和结果文本。原清理验收的 Git BLOCKED 状态属于授权前历史，本次不改写它为当时已同步。

目标为现有 `origin`：`https://github.com/942hx/sddnet-v1-no-denoise.git`，分支 `新版v1无去噪`，upstream 为 `origin/新版v1无去噪`。同步前本地 HEAD 与实时远程 tip 均为 `45e0a698ea9deafb8dd669442a2e11334cb99fd7`，没有额外未推送提交。Git worktree 为父目录，git-dir/common-dir 为父级 `.git`。

精确发布清单见 [git_sync_manifest.csv](git_sync_manifest.csv)，记录每份文件的大小、工作区 SHA256 与 Git blob OID。为避免自引用，清单自身不纳入其内部哈希行，但仍经独立审核、随本次提交提供。清单之外的文件不得被此次暂存。原始 A0 文本必须与不可变迁移清单相符，暂存后的 blob 必须与本地原字节相同。

默认仅保留本地：全部原始 `data/`、`cache/`、两个完整 checkpoint、TensorBoard、所有原图/增强/预览/拼接图片，以及 `results/A0_best88/training/A0_seed42/train.log`（10,042,419 字节）。日志原文不裁剪或改写；history、epoch_records、配置、环境及其他轻量文本正常发布。单个候选文本须小于 5 MiB，不提交任何二进制、大型产物或凭据。

本次正常提交并仅推送 `HEAD:refs/heads/新版v1无去噪`；禁止 force、mirror、改历史、改远端或捎带父级/兄弟目录已有修改。推送前先 dry-run，推送后用 `ls-remote` 核对目标 tip 必须等于本次 commit。精确 commit 和 push/tip 结果在同步结束时向用户交付；本说明不在验证之前宣称推送成功。

当前 Git 的本地代理 `127.0.0.1:7890` 未运行。同步调用使用临时空代理参数直接连接，不修改全局/仓库网络配置。

模型、训练、损失、数据和指标均未在同步阶段改动；沿用清理阶段的 75 项自动测试和数值等价证据，不因 Git 发布重跑训练或完整评测。
