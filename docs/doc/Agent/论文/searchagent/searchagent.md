---
title: searchagent
urlname: bqkn82pma3gvzqrm
date: '2025-11-26 14:19:15'
updated: '2025-11-26 15:01:35'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1764138356934-1943a95a-fbff-4dc8-90a5-a3f7221d09b7.png'
description: search r11. 方法简述rollout部分具体rollout是在searcho1的基础上去做的，在rollout上，只在推理部分算成是rollout，并且分为三个action：think，search，还有一个answer。触发到search就会触发搜索，之后把搜索的内容放到searc...
---
## search r1
### 方法简述
1. rollout部分

具体rollout是在searcho1的基础上去做的，在rollout上，只在推理部分算成是rollout，并且分为三个action：think，search，还有一个answer。触发到search就会触发搜索，之后把搜索的内容放到search里面。

2. 对于算法

在search里面的搜索到的部分R会进行mask，并不会计算到整个句子的奖励当中。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764139321373-b6f2cfe9-deb1-4e24-8f02-733b36ed972a.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764139369342-17cf846c-b487-4f78-b113-04e82a816e51.png)



### 实验部分
用的数据集是HotpotQA和<font style="color:#000000;background-color:#FFFFFF;">General QA </font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764138356934-1943a95a-fbff-4dc8-90a5-a3f7221d09b7.png)

数据集相比于O1和search涨点很多。

比了ppo和grpo，grpo收敛块，但是ppo效果好

比了base和instruct，差不多。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764138668889-2f23d4fa-c506-46d2-8ed4-517fb0e3434c.png)

比了mask是否有效，结论有效，提了至少五六个点

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764138749108-67f04c51-f986-45d6-ab72-d649a6ec433f.png)

turns是先下降后上升，（报告里看到的 ，但是不知道为什么这里没写）

类似于zerosearch这种

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764138852498-66e70608-343e-4d25-aa9c-0a4ee32be68a.png)



## zerosearch
### 简述
search部分使用sft的LLM来代替。

有多种代替的方法，可以是prompt，可以是3bsft，可以是14b..

具体训练的方法是先rollout很多轨迹，之后用这些轨迹去训LLM，标注噪声含量。

具体代替是课程式学习方法，逐渐提升噪声层次，作为search到的结果，让学习逐渐从清晰的search结果过渡到模糊的search结果中提取信息。

所以总结一下就是大模型替代search工具+课程式学习

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764139387737-81448fc3-84be-4e26-b6ca-2d67170f8ae2.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764139535901-a5129cd7-d4c7-40f6-9a97-28b6c6d621e5.png)

模型上Zerosearch vs searchr1 放发生PPOVSGRPOVSreinforce

联系之前那片base可以知道inst强化学习效果更好。

同时能看出来zero效果更好.

