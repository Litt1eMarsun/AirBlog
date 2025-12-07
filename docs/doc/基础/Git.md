---
title: Git
urlname: uqrs0ofaad68zbf4
date: '2025-11-10 16:35:18'
updated: '2025-11-17 22:43:26'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1763389814249-ba95b1b6-8e5a-42a5-b334-e11cd0963fe8.png'
description: 'git常用命令：https://deepinout.com/git/git-questions/360_git_whats_the_difference_between_git_switch_and_git_checkout_branch.htmlswitch和checkout的区别PR流程背...'
---
## git常用命令：
[https://deepinout.com/git/git-questions/360_git_whats_the_difference_between_git_switch_and_git_checkout_branch.html](https://deepinout.com/git/git-questions/360_git_whats_the_difference_between_git_switch_and_git_checkout_branch.html)

switch和checkout的区别

## PR流程
1. 背景：

远程可以有Master/Feat分支，本地同样也可以有对应分支，分支之间可以互相进行merge。

git树图上只会显示当前分支可达的历史

[https://blog.csdn.net/arvinrong/article/details/135876836](https://blog.csdn.net/arvinrong/article/details/135876836)

2. 规范：

在本地feat分支中开发，等到一次开发完成之后，可以将更新推送到远程的feat当中，并在远程的github/gitlab上进行一次PR申请，由项目管理人同意后并入远程的master分支当中。

3. 实际步骤

### 场景1：
单人开发一条feat分支的时候参考流程如下：

1. 经常pull远程的master分支到本地的当前feat分支当中，从而保证在最新代码上进行开发（至少在push之间必须经过一次pull）
2. 如果开发完成，pull远程的到本地，同时处理并更改，更改完成之后commit到本地，之后push到远程的feat分支当中

单人分支开发的时候有些细节：

1. pull默认=fetch+merge，但是单人频繁的merge可能会导致提交记录很乱

merge会进行一次merge commit，（先提出一个commit，并将远程master的head指向当前最新的commit。如图

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1763389814249-ba95b1b6-8e5a-42a5-b334-e11cd0963fe8.png)

如果这样的操作很频繁，则这样会导致当前feat分支当中可能会出现很多个交汇的线，分支一多很头疼。

2. 所以可以使用git pull --rebase（如果进行单人开发的话）

rebase的意思是变基，也就是将当前的feat分支的basenode 直接从之前开发的basenode转到目前最新的master分支的base node上

也就是直接将当前分支的basenode变成最新提交的commit head





### 场景2：
多人开发一条分支：

有空再写

