# Git 入门：保存、上传和查看两个版本

仓库：<https://github.com/942hx/sddnet-v1-no-denoise>（私有）。

## 这次如何组织

Git 在本机保存代码历史，GitHub 保存上传到网上的副本。`commit` 是保存到本机，`push` 才是上传。分支可以理解为同一项目的两个版本视图。

| 分支名称 | 内容 |
|---|---|
| 新版v1无去噪 | 默认分支；`SDDNetV1`、`arch_revision=v1_clean` |
| 旧版v1无去噪 | 旧实现；`model_revision=sddnet_v1_fixed_100e_1` |

每个分支只有本版的源代码、配置、依赖清单、测试和说明。切换分支时，项目目录会显示该版的代码；不是同时在目录里堆两套模型。

`data/`、`outputs/`、权重、日志、实验CSV、缓存、源码备份和个人任务文档没有提交或上传，仍保留本机。本仓库没有提供训练好的权重。两版checkpoint格式不兼容，请分别使用对应版本。

## 平时改完代码如何保存和上传

在项目根目录打开 Anaconda Prompt，依次执行：

```bat
git status
git diff
git add .
git diff --cached --stat
git commit -m "说明这次修改了什么"
git push
```

`git status` 查看当前分支和改了哪些文件，`git diff` 查看具体改动，`git add` 选择本次要保存的改动，`git commit` 保存一个本地版本，`git push` 上传到GitHub。没有代码改动时，不需要commit。

`.gitignore` 已限制只接纳源码目录和必要文档，所以普通 `git add .` 不会加入数据和训练输出。不要用 `git add -f` 强行加入这些文件。新代码优先放在现有的 models/datasets/losses/engine/utils/scripts/configs/tests 目录；需要新增根目录文件时，先在 `.gitignore` 中允许该文件。

## 查看和切换版本

```bat
git branch
git status
git switch "旧版v1无去噪"
git switch "新版v1无去噪"
git log --oneline --all --graph
```

切换前先保存并提交当前修改；如果Git提示本地修改会被覆盖，先处理这些修改，不要强行丢弃。切换会更换代码文件，忽略的数据和权重仍留在本机。两版的训练命令不同，切换后阅读该分支的 README。训练进程运行期间不要切换版本。

只想查看旧版时，可以在GitHub网页左上角分支下拉框选择“旧版v1无去噪”，不必切换本机代码。点击绿色 Code → Download ZIP 下载当前选中分支；ZIP不包含数据和权重。

## 在另一台电脑获取代码

```bat
git clone https://github.com/942hx/sddnet-v1-no-denoise.git
cd sddnet-v1-no-denoise
git switch "旧版v1无去噪"
```

默认clone新版，切换命令可选。私有仓库需要登录有访问权限的GitHub账号。数据与权重需自行放入对应目录，然后按README配置环境。

在另一台电脑推送过代码后，本机开始编辑前可以先执行：

```bat
git pull --ff-only
```

如果提示无法快进，先查看 `git status` 和提交历史，不要直接强制覆盖。

## 本次设置

- 默认分支：新版v1无去噪；两个分支均配置了上游，切换后可直接 `git push`。
- 本机仓库级提交作者使用 GitHub 的 noreply 地址，未改动全局 Git 身份设置。
- GitHub令牌由系统凭据管理器管理，没有写进代码或仓库。
- 创建仓库并不等于给代码添加开源许可证；本次未擅自选择许可证。

官方中文入门：[记录每次更新](https://git-scm.com/book/zh/v2/Git-基础-记录每次更新到仓库)、[分支的新建与切换](https://git-scm.com/book/zh/v2/Git-分支-分支的新建与合并)。
