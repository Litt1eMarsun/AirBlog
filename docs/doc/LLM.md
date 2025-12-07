---
title: LLM
urlname: bu9rzg5qkt3vqp9q
date: '2025-07-17 15:24:10'
updated: '2025-11-21 10:55:55'
description: 1. 学下LLM的架构种类1.1. 为什么主要用的还是以Decoder为主的架构？1.1.1. Trans中encoder-decoder的主要区别结构区别mask attentionmutihead attention中输入了encoder的memory推理/训练时的使用区别推理/训练的时候...
---
## <font style="color:rgb(25, 27, 31);">学下LLM的架构种类</font>
### 为什么主要用的还是以Decoder为主的架构？
#### Trans中encoder-decoder的主要区别
1. 结构区别
    1. mask attention
    2. mutihead attention中输入了encoder的memory
2. 推理/训练时的使用区别
    1. 推理/训练的时候使用了很多次的decoder做推理，但encoder只使用一次

剩下的看推理/训练数据流动过程就ok了

##### transforemr中推理的过程中数据流动的过程是什么样的？
对于transforemr，文中并没有提到文本生成任务。对于翻译任务而言，推理过程如下

> **<font style="color:rgb(0, 0, 0) !important;">Step 1：Encoder 预处理源语言（仅执行一次）</font>**
>
> + **<font style="color:rgb(0, 0, 0) !important;">输入英文句子</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">："I love machine learning"</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 1.1：Tokenize（分词）</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">  
</font><font style="color:rgba(0, 0, 0, 0.85) !important;">将英文句子转换为 token 序列：</font>`<font style="color:rgb(0, 0, 0);">[I, love, machine, learning]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（假设无需子词拆分），形状为 </font>`<font style="color:rgb(0, 0, 0);">[1, 4]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（batch_size=1，seq_len=4）。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 1.2：嵌入（Embedding）+ 位置编码</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">每个 token 通过嵌入层映射到高维向量（维度</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">d_model=512</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">），形状变为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1, 4, 512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">叠加位置编码（捕获词序信息），输出仍为 </font>`<font style="color:rgb(0, 0, 0);">[1, 4, 512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 1.3：Encoder 层计算</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">  
</font><font style="color:rgba(0, 0, 0, 0.85) !important;">输入经过 N 个 Encoder 层（含自注意力 + 前馈网络），最终输出</font>**<font style="color:rgb(0, 0, 0) !important;">源语言上下文特征</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">（记为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">memory</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">），形状为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1, 4, 512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">自注意力层：捕获英文单词间的依赖（如 "love" 与 "I" 的关系）。</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">前馈网络：对每个位置的特征独立进行非线性变换。</font>
>
> #### **<font style="color:rgb(0, 0, 0) !important;">Step 2：Decoder 自回归生成目标语言（逐 token 生成）</font>**
> <font style="color:rgba(0, 0, 0, 0.85);">Decoder 的任务是基于 </font>`<font style="color:rgba(0, 0, 0, 0.85);">memory</font>`<font style="color:rgba(0, 0, 0, 0.85);">（Encoder 输出）和</font>**<font style="color:rgb(0, 0, 0) !important;">已生成的中文前缀</font>**<font style="color:rgba(0, 0, 0, 0.85);">，逐个生成中文 token，直到生成 </font>`<font style="color:rgba(0, 0, 0, 0.85);"></s></font>`<font style="color:rgba(0, 0, 0, 0.85);">。</font>
>
> ##### **<font style="color:rgb(0, 0, 0) !important;">初始状态：生成第一个 token</font>**
> + **<font style="color:rgb(0, 0, 0) !important;">初始输入</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：目标序列从起始标记 </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> 开始，输入为 </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">[<s>]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">，形状 </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">[1, 1]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（batch_size=1，seq_len=1）。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 2.1：目标序列预处理</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">Token 嵌入：</font>`<font style="color:rgb(0, 0, 0);"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> </font><font style="color:rgba(0, 0, 0, 0.85) !important;">映射为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1, 1, 512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">位置编码：叠加位置信息（此时序列长度为 1，位置编码对应第 1 个位置），输出 </font>`<font style="color:rgb(0, 0, 0);">[1, 1, 512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 2.2：Decoder 第一层自注意力（带 mask） （在这里可以看出mask attention的设计原因）</font>**
>     - **<font style="color:rgb(0, 0, 0) !important;">输入</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：目标序列嵌入（</font>`<font style="color:rgb(0, 0, 0);">[1,1,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">）。</font>
>     - **<font style="color:rgb(0, 0, 0) !important;">掩码（mask）</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：使用 1×1 的下三角矩阵（仅允许当前 token 看到自身），形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,1,1,1]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（batch_size=1，头数 = 1 时）。</font>
>     - **<font style="color:rgb(0, 0, 0) !important;">计算</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：通过自注意力层，输出 </font>`<font style="color:rgb(0, 0, 0);">[1,1,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（仅基于 </font>`<font style="color:rgb(0, 0, 0);"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> 自身的特征）。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 2.3：Encoder-Decoder 注意力（跨注意力）</font>**
>     - **<font style="color:rgb(0, 0, 0) !important;">输入</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：自注意力的输出（</font>`<font style="color:rgb(0, 0, 0);">[1,1,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">）和 Encoder 的</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">memory</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（</font>`<font style="color:rgb(0, 0, 0);">[1,4,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">）。</font>
>     - **<font style="color:rgb(0, 0, 0) !important;">作用</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：让当前生成的 token 关注源语言中相关的词（如</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> </font><font style="color:rgba(0, 0, 0, 0.85) !important;">可能先关注 "I"）。</font>
>     - **<font style="color:rgb(0, 0, 0) !important;">输出</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：融合源语言信息的特征，形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,1,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 2.4：前馈网络（FFN）</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">对跨注意力输出进行非线性变换（如</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">Linear → ReLU → Linear</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">），输出仍为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,1,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">步骤 2.5：输出层与 token 选择</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">经过线性层（映射到词汇表维度），输出 logits（未归一化的概率），形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,1,V]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（V 为词汇表大小）。</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">对 logits 做 softmax，取概率最高的 token：假设生成第一个中文 token</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>**<font style="color:rgb(0, 0, 0) !important;">"我"</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
>
> #### **<font style="color:rgb(0, 0, 0) !important;">Step 3：生成第二个 token</font>**
> + **<font style="color:rgb(0, 0, 0) !important;">更新输入</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：目标序列变为</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">[<s>, 我]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">，形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">[1,2]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">位置编码</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：新增位置 2 的编码，嵌入后形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">[1,2,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">自注意力层（mask 更新）</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">掩码变为 2×2 的下三角矩阵（确保 "我" 只能看到 </font>`<font style="color:rgb(0, 0, 0);"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> 和自身，无法看到未来 token）</font>
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">自注意力输出：融合</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);"><s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> </font><font style="color:rgba(0, 0, 0, 0.85) !important;">和 "我" 的特征，形状</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,2,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">跨注意力层</font>**
>     - <font style="color:rgba(0, 0, 0, 0.85) !important;">关注源语言："我" 可能重点关联英文的 "I"，输出融合后特征</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[1,2,512]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">输出层</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：取最后一个位置（"我" 之后）的 logits，生成第二个 token</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>**<font style="color:rgb(0, 0, 0) !important;">"爱"</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
>
> #### **<font style="color:rgb(0, 0, 0) !important;">Step 4：重复生成，直到结束</font>**
> + **<font style="color:rgb(0, 0, 0) !important;">第 3 轮</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：输入</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[<s>, 我, 爱]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">，seq_len=3，mask 为 3×3 下三角矩阵，生成 token</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>**<font style="color:rgb(0, 0, 0) !important;">"机器"</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">第 4 轮</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：输入</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[<s>, 我, 爱, 机器]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">，生成 token</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>**<font style="color:rgb(0, 0, 0) !important;">"学习"</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">。</font>
> + **<font style="color:rgb(0, 0, 0) !important;">第 5 轮</font>**<font style="color:rgba(0, 0, 0, 0.85) !important;">：输入</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`<font style="color:rgb(0, 0, 0);">[<s>, 我, 爱, 机器, 学习]</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">，生成 token</font><font style="color:rgba(0, 0, 0, 0.85) !important;"> </font>`**<font style="color:rgb(0, 0, 0);"></s></font>**`<font style="color:rgba(0, 0, 0, 0.85) !important;">（结束标记）。</font>
>
> #### **<font style="color:rgb(0, 0, 0) !important;">Step 5：终止条件</font>**
> <font style="color:rgba(0, 0, 0, 0.85) !important;">当生成 </font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"></s></font>`<font style="color:rgba(0, 0, 0, 0.85) !important;"> 时，推理结束，最终输出序列为：</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">我 爱 机器学习</font>`<font style="color:rgba(0, 0, 0, 0.85) !important;">（去除起始和结束标记）。</font>
>

<font style="color:rgba(0, 0, 0, 0.85) !important;">  
</font>

    - 

<font style="color:rgba(0, 0, 0, 0.85) !important;"></font>

想下attention里面的kqv?



value矩阵太大了怎么办？——对value进行LR分解



#### encoder为主的NLU结构是什么样的？
#### Decoder为主的结构是什么样的？
#### E-D?——和Trans
#### 对比？


## 利用LLaMA factory 对 LoRA 进行 SFT
<details class="lake-collapse"><summary id="u5c7306b8"><span class="ne-text">学习llama factory</span></summary><ol class="ne-ol"><li id="ub996726b" data-lake-index-type="0"><a href="https://www.bilibili.com/video/BV1oTEwzcEeZ/?spm_id_from=333.1387.upload.video_card.click&amp;vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d" data-href="https://www.bilibili.com/video/BV1oTEwzcEeZ/?spm_id_from=333.1387.upload.video_card.click&amp;vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d" target="_blank" class="ne-link"><span class="ne-text">https://www.bilibili.com/video/BV1oTEwzcEeZ/?spm_id_from=333.1387.upload.video_card.click&amp;vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d</span></a></li><li id="uabaace83" data-lake-index-type="0"><span class="ne-text">学习下上述链接</span></li></ol></details>


1. 需求分析，数据处理，技术选型，评估指标搭建好
2. 参照已有的项目文档，把基础的功能代码熟悉，技术流程全部熟悉
    1. 整理数据处理流程
    - 花园code的合成数据流程整理
    - weclone的敏感数据/block关键词处理等流程，清洗数据等流程
    - 银行项目看看
    - 小红书小灯大模型基本流程
+ 技术选型一系列问题
3. 根据补充需求，和假设的评估结果，对技术流程提出优化方案
4. 整理微调调参知识
+ 小红书文章
+ 花园code教程
5. 整理模型lora基础知识
+ 小红书
6. 整理评估指标，迭代标准
+ llama facory
+ 小红书
+ 先搞清楚详细的项目背景和需求，让大模型帮忙生成一系列指标（想法来源于小红书小灯如何包装项目）
7. 整理常见问题极其迭代
+ 小灯的文章有很多bad case的例子，可以整理一下

## 


