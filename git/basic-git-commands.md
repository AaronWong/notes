# Git 基本命令

参考 ali-code https://code.aliyun.com/help/code-basics/basic-git-commands.md

## 1 获取master分支的最新更改

`git checkout master`

## 2 下载项目的最新更改

该命令用于拉取某分支的最新副本（建议工作时每次都输入这个命令）。

`git pull 远端 分支名称 -u`

(远端: origin) (分支名称: 可以是"master"或者是一个已经存在的分支)

## 3 创建一个分支

由于空格不会被识别，所以请使用"-"或者"_"。

`git checkout -b 分支名称`

## 4 在某分支上进行开发

`git checkout 分支名称`

## 5 浏览更改

`git status`

## 6 将更改加入到本次提交

当输入"git status"时，您的更改会显示为红色。

`git add 红色的修改`

`git commit -m "提交的描述"`

## 7 代码提交您的更改到服务器

`git push 远端 分支名称`

## 8 删除代码库的所有更改（不包含提交到暂存区的变更）

`git checkout .`

## 9 删除代码库的所有更改（包含未跟踪的文件）

`git clean -f`

## 10 将某分支合并到master分支

`git checkout 分支名称`

`git merge master`