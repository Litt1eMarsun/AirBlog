---
title: 手搭基模_pretrain部分：
urlname: viwnaxmwh5pzpnhe
date: '2025-08-06 10:17:58'
updated: '2025-11-11 11:22:10'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1754446737066-04b7bd38-3009-4a10-83d2-7e6faf7be854.png'
description: '1. tokenizer中文模型对中文的分词会有单独优化，与英文模型不同。所以不能确定一句话具体对应着多少token 1.1. Word2vec(tokenizer搭建)参考内容https://zhuanlan.zhihu.com/p/55983009实现了通过神经网络高维的句子向低维的vec...'
---


## tokenizer
中文模型对中文的分词会有单独优化，与英文模型不同。所以不能确定一句话具体对应着多少token 

### Word2vec(tokenizer搭建)
参考内容[https://zhuanlan.zhihu.com/p/55983009](https://zhuanlan.zhihu.com/p/55983009)

实现了通过神经网络高维的句子向低维的vec转化的过程。vec要有至少以下几个能力：

1. **<font style="color:rgb(25, 27, 31);">携带上下文信息</font>**
2. **<font style="color:rgb(25, 27, 31);">词的表示是稠密的</font>**

模型示意图如下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1752737432814-795ebaa8-1d6b-4ebc-af9a-81b085526adc.png)

输入是词汇库中**<font style="color:rgb(25, 27, 31);">V个单词</font>**单词数量，Hidden Layer**<font style="color:rgb(25, 27, 31);">为单词向量的维度，</font>**是自己设的超参也就是N，输出层大小和输入层数量一致。

举例，假设<font style="color:rgb(25, 27, 31);">语料库词汇有八个单词['a','cat','chasing','climbed','dog','saw','the','tree']，隐藏层大小为3，则WI为8*3，WO为3*8。</font>

<font style="color:rgb(25, 27, 31);">具体的训练过程如下：</font>

<font style="color:rgb(25, 27, 31);">训练数据是多个句子（当然句子也可以排列组合形成更多句子），句子中肯定每个词语都携带了上下文的信息，但是模型的设计之初，输入和输出都是整个词汇表的大小，所以我们可以把一个句子标志为一个独热编码，去让模型学习词语之间的关系，也就是满足了之前提到的第一个要求：</font>**<font style="color:rgb(25, 27, 31);">携带上下文信息</font>**

<font style="color:rgb(25, 27, 31);">举个例子：输入的句子如果是“Dog chasing cat”，我们希望网络学习单词“cat”和“climbed”之间的关系。</font>

<font style="color:rgb(25, 27, 31);">则按理说当“cat”输入到网络时，网络应该显示“climbed”的</font>**<font style="color:rgb(25, 27, 31);">高概率</font>**<font style="color:rgb(25, 27, 31);">。在单词嵌入术语中，单词“cat”被称为</font>**<font style="color:rgb(25, 27, 31);">context word</font>**<font style="color:rgb(25, 27, 31);">，单词“climbed”被称为</font>**<font style="color:rgb(25, 27, 31);">target word</font>**<font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(25, 27, 31);"></font>

### 其他
实际工程上的特殊字符存在tokneizer_config.json当中

## 旋转位置编码
### 角度设定
先定义如下函数：也就是给不同embedding的不同维度的特定旋转角度

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754446737066-04b7bd38-3009-4a10-83d2-7e6faf7be854.png)

```plain
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

```

### 广播角度  

之后常规操作，对freq进行广播方便与x做乘积

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)
```

  
 对于倒数第二行，操作过程如下：

```python
x.shape = (2, 128, 8, 64)  # batch=2, seq_len=128, n_head=8, head_dim=64
freqs_cis.shape = (128, 64)
```

+ reshape 出来的 shape 会是 `[1, 128, 1, 64]`，可以广播到 `(2, 128, 8, 64)`
+ 这样可以直接做：

x_rotated = x * reshape_for_broadcast(freqs_cis, x)

### 旋转过程如下：
```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

原理如下：  
实际操作是把每两个数看成一个二维向量坐标并进行旋转。拆分成的32,2中的2分别进行

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754451824295-d45e01d4-1493-42f0-a4c0-4d247a23a349.png)

之后再拼接到一起形成[x',y;]这样就对所有embedding进行旋转了。只是有一个复数表达方式罢了，纯花活

把之前堪称的二维向量坐标现在看成复数，则旋转可以如下表达：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754451967516-ef3e5060-36bd-4b70-b64e-849f3cf7b81b.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754451989250-f196b0a4-b5c7-4cd4-9355-1e8e8559559e.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754452034637-11921ff6-204c-496f-b1a3-eae3456ede5a.png)

这样被视为复数的两个坐标就被变幻成了如下形式，之后再拼接。

代码逐行解释

1. xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)

把query的最后一个维度拆分成两个，并分别赋给sq_r,xq_i

即(2, 128, 8, 64) → (2, 128, 8, 32, 2),并分别赋给sq_r,xq_i

这个reshape的原理如下：

`**reshape()**`** 不改变数据的顺序，只是重新解释它的形状（按行主序 row-major 顺序）**。

也就是说：

    - 原始的 `(2, 128, 8, 64)` 中的所有数据，在内存中是一个一维数组，共有 `2×128×8×64 = 131072` 个元素。
    - `reshape` 后，**不会打乱这些元素的顺序**，只是按新的维度解释它。

比如说

x = torch.arange(128, dtype=torch.float32)  # 这是一个 1D 向量

x → (64, 2)，其中 x[i, 0] 是实部，x[i, 1] 是虚部

tensor([[ 0.,  1.],   # 复数1：实部0，虚部1

        [ 2.,  3.],   # 复数2：实部2，虚部3

        [ 4.,  5.],   # ...

        [ 6.,  7.],

        [ 8.,  9.]])

先在行上排，再排右。

2. freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)

广播freq_cos和sin，方便和后面的乘积

3. xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin     xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

实部，虚部旋转

理解了旋转矩阵怎么来的，现在看看为什么能凸显出相对位置

<font style="color:rgb(25, 27, 31);">设旋转矩阵如下图所示</font>

```plain
R(pos) = [cos(pos·θ)  -sin(pos·θ)]
         [sin(pos·θ)   cos(pos·θ)]
```

现在对n位置的q矩阵和m位置的k矩阵应用旋转矩阵，结果如下图所示

```plain
q_n = R(n) · Q = [cos(n·θ)  -sin(n·θ)] · Q
                 [sin(n·θ)   cos(n·θ)]
k_m = R(m) · K = [cos(m·θ)  -sin(m·θ)] · K
                 [sin(m·θ)   cos(m·θ)]
```

最终要对q和k算注意力乘积，乘积计算的过程如下：

```plain
Attention(n,m) = q_n · k_m = (R(n)·Q) · (R(m)·K)
```

实际是Q的转置QT乘K

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759495527857-3e4789b5-d127-4682-885c-ec19e21e40d5.png)

因为![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759495546445-ece28c80-bbdc-48b2-aed9-9efb1c0a5570.png)



所以重新把矩阵用欧拉公式转换乘θ形式之后再乘积就是如下结果，Qt和K之间的乘积是两个角度之差，能够完成位置编码的使命

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759495684372-7ce10aec-80d7-4081-9462-b653a1a0ddfa.png)





### NTK-aware及YARN等后续的context扩展工作
[https://zhuanlan.zhihu.com/p/20328774059](https://zhuanlan.zhihu.com/p/20328774059)

## RLHF
### PPO
####  数据获取： 第一步：人类评审两个回答，形成“偏好对”  
比如你输入一个提示（prompt）：

"如何处理职场中的人际关系？"

语言模型生成了两个回答：

+ 回答 A：要学会沟通，尊重他人意见。
+ 回答 B：直接无视那些不喜欢你的人，不用太在意。

人类评审者觉得 A 更得体，于是选了 A。

于是形成一条训练数据：

```plain
(Prompt, Answer A, Answer B, preference: A)
```

📌 **这叫偏好对（preference pair）**。

🔍 现实中会让语言模型生成多个候选回答，然后让人类对其中两个或多个打分或排序。

####  训练一个“奖励模型”（Reward Model）  
##### 📘 什么是奖励模型？
你可以把它想象成一个“打分老师”，专门根据回答质量给出分数。

它的目标是：输入一个 prompt 和一个回答，输出一个分数，代表“这个回答有多好”。

##### 🧠 怎么训练奖励模型？
这时就用上面收集的人类偏好对！

比如：你告诉模型，"回答 A 比 回答 B 更好"，那么它要学会给 A 的分数比 B 高。

训练方式一般是：

+ 输入 (Prompt, Answer A)，模型预测分数 s_A
+ 输入 (Prompt, Answer B)，模型预测分数 s_B
+ 用一个 loss 函数惩罚 `s_A < s_B` 的情况

通常使用一个叫 **pairwise loss** 的函数（比如 logistic loss）：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754651230689-4a0c1250-a44c-4642-9670-ae850a826a92.png)

👉 **也就是让模型的分数朝着“人类更喜欢的那个”靠拢。**

#### 🧪 第三步：用强化学习优化语言模型（PPO等）
好，现在我们有了一个“打分老师”（奖励模型），下一步就是：

用这个老师来“指导语言模型”：让它输出分数更高的回答。

这一步使用的是强化学习（比如 PPO：Proximal Policy Optimization）。

##### 🤖 为什么需要强化学习？
因为语言模型本身输出的是“词概率”，但奖励模型不是概率，它是一个“外部打分”系统。

我们不能直接用“梯度反向传播”来优化语言模型，让它的回答更高分。因为：

+ **模型输出是“离散的词”**

具体来说，语言模型输出的是一个个词，比如：

```plain
"你好，今天的天气..."
```

在模型内部，它其实不是直接输出这个句子，而是：

在每一个位置，预测一个词的 **概率分布**，然后从中**采样（抽签）**出一个词。

比如：

+ 模型预测当前位置的词是：
    - “你好” 的概率是 0.6
    - “早上好” 的概率是 0.3
    - “嘿” 的概率是 0.1
+ 然后模型**随机抽样**一个词出来，比如这次抽到了“你好”

👉 这个采样（sampling）操作，是**非可导（non-differentiable）**的。

也就是说，你不能用梯度告诉模型：“你刚才抽出的是个低分词，下次别抽了。”

因为“抽签”这个过程没法微调。

所以需要强化学习框架：语言模型成为“智能体”（Agent），尝试不同的回答（动作），根据奖励模型给的分数（奖励）来调整策略（即生成回答的方式）。



### DPO


## 搭建LLaMA	
### RMSNorm
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754725850218-e9317af6-a04f-4091-a310-6c5d51a69113.png)

把普通LayerNorm除的方差变成了平方和的均值，并且减少了减去均值这一步，同时减少了一个偏执，对比下NormLayer

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754904784418-dbbaa652-bcc8-4b14-b782-86b90c4dfa0f.png)

RMSNorm有什么优点呢？  
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754904843913-d36ef87a-38dd-49c8-b234-06c829e62da6.png)

RMSnorm代替了所有的LayerNorm，因为论文中**<font style="color:rgb(25, 27, 31);">这样处理之后，整体计算更高效，模型的训练会更快。</font>**<font style="color:rgb(25, 27, 31);">并且效果没有跌下来多少</font>

正好在这里对比一下LayerNorm和BatchNorm

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754902761599-72a3af2d-3f5c-437c-b099-8d86e31868ce.png)

在transformer里面，一个batch的不同seq的length不一样，要保持batch一致得对seq_length较小的进行seqlength尺度下的padding，这种类似下图

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1754901952380-5aca9ca5-de9e-47b8-942c-ea0af080a5da.png)

batchNorm会包含了padding，效果会不好，LayerNorm对embedding尺度下做norm的话是不包含paddidng的





### _init_weights(self, module):
对线性层按照正态分布进行初始化

 以一个 `nn.Linear(in_features=4, out_features=3)` 为例，它的 `weight` 是一个形状为 `[3, 4]` 的张量。  

执行这行代码后：

```python
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

会对这个 3×4 的矩阵的 **每个元素**，**独立地**从 `N(0, 0.02^2)` 中采样。  
 也就是每个initial时候的线性层都会在一个正太分布里采样，不过应该先效果是一样的

同时对embedding也这样，对线性层的bias全赋值为0









### attention
<font style="color:rgb(25, 27, 31);">attention深度解析：  
</font>[https://zhuanlan.zhihu.com/p/626820422](https://zhuanlan.zhihu.com/p/626820422)

这里的attention计算是GQA

![](https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/5-images/llama2-attention.png)

1. 输入的数据大小为batch_size, seq_len, dim
2. K,Q,V三个矩阵大小为

self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

注意三个矩阵的大小不一致，kv是一个大小，并且比较小，通过矩阵之后直接把embedding dim大小变成head*dim大小，后面会拆分成多个头

3. 之后给KQ矩阵利用posembedding进行旋转，具体代码看之前的RoPE
4. 再对K和V两个矩阵进行repeat，从而匹配query,之后就可以进行注意力计算了，repeat代码如下：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )
```

5. 计算attentionmap，详见下面的attention中的forward  
 (batch,head,seq_l,dim)* (batch,head,dim,seq_l)获得注意力图谱(batch,head,seq_l,seq_l)。
6. 之后是比较难理解的mask，为什么要做mask，为什么有效？

**首先明确以下为什么要做mask，因为希望在训练的时候希望一次性训练一整句话(数据层面的 并行操作，并不是一次性输出所有的语句）**

比如训练的时候输入的x,y是 

X: [BOS, T1, T2, T3, PAD] # 输入

Y: [T1, T2, T3, PAD, PAD] # 目标 讨论下x在过attention模块的过程。

我们希望经过mask之后输出的attention map和value乘积是[batch,head,seq_length,dim]

其中[:,:,0,:]这一列数据继续处理之后和T1做loss,把[:,:,1,:]这一列数据处理之后是和T2做loss的，依次类推，从而做到了每次输入模型能够在每个batch训练一整句话,  而不是每输入一个字和groundtruth做loss，之后再添加再输入，这是推理的时候干的，效率会比较低

**接下来说明下 mask前后的步骤大概如下**

    1. **计算 Attention Scores  之后 引入 Mask（关键防偷窥环节）  **

Mask 的本质：

        * 创建一个与 `<font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">scores</font>` 同形状的矩阵 `<font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">mask</font>`
        * 对于不允许访问的 (t,s)(t, s)(t,s) 位置，填充一个极大负数（例如 -1e9）
        * 对允许访问的位置填 0

例如 Causal Mask（防止看未来 token）：

    2. **Softmax 转为概率分布**

<font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">attn_weights</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">=</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">softmax</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">(</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">scores</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">masked</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">)</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">\text{attn\_weights} = \text{softmax}(\text{scores}_{\text{masked}})</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">attn_weights</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">=</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">softmax</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">(</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">scores</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">masked</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">)</font>

        * <font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">每一行（每个 token 对所有 token 的注意力分布）只会在允许的位置有概率值</font>
        * <font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">禁止的位置概率严格是 0</font>
        * <font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">从数学上切断了获取未来 token / padding token 信息的路径</font>
    3. **加权求和（信息流的物理屏蔽）**

最后用注意力权重加权 V：

outputt=∑s=1Lattn_weights[t,s]⋅Vs\text{output}_t = \sum_{s=1}^{L} \text{attn\_weights}[t, s] \cdot V_soutputt=s=1∑Lattn_weights[t,s]⋅Vs

        * 因为禁止的 s 位置概率是 0，所以它们的 V_s 根本不会贡献到 output_t
        * **在数据流上彻底阻断了偷窥通道**

**总的来说从seq这个维度看，每个v向量的权重都只和当前和前面所有的权重有关，做乘积的时候都只乘了前面的注意力图谱上的权重，这样输出就实现了物理屏蔽，最终的目的是为了并行一次用一整句话完成训练**

7. 投影回之前的维度

```python
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
```

(batch,head,seq,head_dim)-(batch,seq,head*head_dim)

(batch,seq,head*head_dim)-(batch,seq,dim)_dropout

8. 完整的代码如下：

```python
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率。
        self.dropout = args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询（Q）、键（K）、值（V）。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键和值进行扩展以适应重复次数。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)# 转换下维度
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention。
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 使用手动实现的注意力机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

<font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);">  
</font><font style="color:rgb(82, 82, 82);background-color:rgb(248, 248, 248);"> </font>

### MLP
输入输出的维度一致，

(batch,seq,dim)-(batch,seq,dim)

扩大再缩小罢了

```python
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

这里解释一下MLP中的细节：：

首先MLP中使用的激活函数是SiLU函数，可以理解成beta=1的Swish函数，也就是没有门控矩阵的swish函数

[https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.SiLU.html](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.SiLU.html)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759486914877-13cd71fa-df5f-4507-809e-97e409372cd3.png)  
 ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759486924537-301fae1a-f975-4edc-8817-1a5fcdd301dd.png)

这里解释以下Swish函数

[https://zhuanlan.zhihu.com/p/364620596](https://zhuanlan.zhihu.com/p/364620596)（包含了激活函数的发展历程，暗含了为什么用swish函数比较多）

<font style="color:rgb(25, 27, 31);">Swish激活函数又叫作自门控激活函数，它由谷歌的研究者发布，数学表达式为：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759486982039-6dd12c4c-ffad-4e7d-bcb5-9c8685dd6ac6.png)

<font style="color:rgb(25, 27, 31);">β为可学习的参数或一个固定超参数，  可以看做是一种软性的门控机制。</font>

<font style="color:rgb(25, 27, 31);">当 σ(x)接近于1时，门处于“</font>**<font style="color:rgb(25, 27, 31);">开</font>**<font style="color:rgb(25, 27, 31);">”状态，激活函数的输出近似于x本身；</font>

<font style="color:rgb(25, 27, 31);">当 σ(x)接近于0时，门处于“</font>**<font style="color:rgb(25, 27, 31);">关</font>**<font style="color:rgb(25, 27, 31);">”状态，激活函数的输出近似于0；</font>

<font style="color:rgb(25, 27, 31);">因此，</font>**<font style="color:rgb(25, 27, 31);">Swish 函数可以看作线性函数和ReLU 函数之间的非线性插值函数</font>**<font style="color:rgb(25, 27, 31);">，</font>**<font style="color:rgb(25, 27, 31);">其程度由参数  控制。</font>**

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759487045556-71c3dbd7-57cf-4d8b-80f0-ef5f36c2d4ac.png)

这就是为什么使用Swish最重要的原因，门控函数可以根据情况自己选择使用什么样的激活函数，  
可以是β很大，也就是ReLU，选择这个的时候可能不会碰到什么x小于0 的情况，

可能是β=0...



实际上的FFN中，输入进SiLU的是x*w_gate，也就是说门控矩阵是手动输入的，也就是说手动从SiLU函数实现了Swish函数

同时与swish不同的是，门控矩阵还对<font style="color:rgb(25, 27, 31);">σ(x)外也有作用，也就是βxσ(βx)，这里可以理解成一个完整的激活函数，对应的代码是：</font>

<font style="color:rgb(25, 27, 31);">F.silu(self.w1(x)）</font>

<font style="color:rgb(25, 27, 31);">之后再对整体做个点乘W1，这些都是升高维度的过程</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759487337098-9dfc36ab-b7a8-4909-93e5-75fb47425f43.png)

这个就是整体激活的过程

### LLaMA总结构
根据之前提到的搭建decoder

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

  
 再搭建llama

```python
class Transformer(PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight 

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else: 
            # 只对最后一个位置的输出进行前向传播，推理过程走这边
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

```

### 推理过程：
#### 处理输入数据
    1. 截断

#### 获得词表概率
```python
logits = self(idx_cond).logits # 这里self就是前向传播，输出的logits形式为(batch_size, seq_len, vocab_size)
logits = logits[:, -1, :] # 只保留最后一个时间步的输出，用来获得下一个token
```

#### 挑选输出token 
```python
if temperature == 0.0:
     # 选择最有可能的索引
    _, idx_next = torch.topk(logits, k=1, dim=-1)#贪心策略，直接选概率最高的 token id
else:
    # 缩放 logits 并应用 softmax
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
```

##### 贪心策略
直接选概率最高的，随机性低，表达力弱

##### 温度缩放采样策略（Temperature Scaling Sampling）  
1. 背景：为什么要温度缩放？

在生成模型里，比如 GPT，你会得到一个预测概率分布：

```plain
rust


复制编辑
logits -> softmax -> 概率分布 P
```

如果我们直接选概率最大的 token（贪心策略），生成结果会：

    - 非常确定，但**缺乏多样性**，容易重复
    - 对低概率选项几乎没有探索

如果我们直接按原始概率分布随机采样：

    - 多样性高
    - 但容易出现低质量或语义跳跃的结果

**温度缩放** 就是用一个系数 **T（Temperature）** 来调节这个随机性。

2. 公式

假设原始 logits 为 zi，softmax 公式是：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755098137734-2f72f992-e166-471e-b8d5-36e68e66cbda.png)

加入温度 T：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755098153001-af3ea48c-2d67-472e-a2f5-c3fa3c353a96.png)

    - **T < 1**：分布变得更尖锐 → 更确定 → 接近贪心
    - **T > 1**：分布变得更平滑 → 更随机 → 多样性提升
    - **T → 0**：极端贪心（直接取最大值）
    - **T → ∞**：几乎均匀随机
3. 效果示意

假设 logits = `[5.0, 2.0, 1.0]`  
softmax(T=1) → 概率分布：

```plain
csharp


复制编辑
[0.84, 0.11, 0.05]
```

    - **T = 0.5**（降低温度）：

```plain
nginx


复制编辑
logits / 0.5 = [10, 4, 2]
softmax → [0.97, 0.02, 0.01]
```

分布更尖锐，几乎必选第一个 token。

    - **T = 2.0**（升高温度）：

```plain
nginx


复制编辑
logits / 2 = [2.5, 1.0, 0.5]
softmax → [0.62, 0.23, 0.15]
```

分布更平滑，第二、第三个 token 也有更大概率被选到。

4. 在采样里的作用

温度缩放通常与**随机采样**（multinomial）或**Top-k/Top-p 策略**结合使用：

    1. 模型输出 logits
    2. **除以温度 T**
    3. 进行 softmax 得到概率
    4. 再随机抽样一个 token 作为下一个生成的 token

这样：

    - T 小 → 模型更保守，生成结果更稳定、重复性高
    - T 大 → 模型更开放，生成多样性更高

详见[https://zhuanlan.zhihu.com/p/1899617450024235966](https://zhuanlan.zhihu.com/p/1899617450024235966)

#### 返回token 
```python
     # 将采样的索引添加到序列中并继续
        idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token
```

index是输入的问题长度，generate在max_new_tokens下循环生成的所有token的长度就是index:之后的部分，返回这些部分



#### 总代码
```python
@torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits # 这里self就是前向传播，输出的logits形式为(batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出，用来获得下一个token
            
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)#贪心策略，直接选概率最高的 token id
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token
```

  
 

### 预训练部分 
```python
input_id:  [BOS, T1, T2, T3, PAD, PAD]
X:         [BOS, T1, T2, T3, PAD]   # 输入
Y:         [T1,  T2, T3, PAD, PAD]  # 目标
mask:      [1,    1,  1,   0,   0]
```

输入的数据就是X，

从LLaMA输出[1,1,vocab_size]就是词表各个单词出现的概率

而Y就是给定的groundtruth

输入T1和Bos(gt)做loss，t2和t1做loss之后会获得类似以下结果

```python
loss:
[
 [0.2, 0.1, 0.3, 0.05, 0.02],
]

```

得到每个计算loss的过程就是利用交叉熵MSE

和groundtruth可以计算交叉熵，过程大概如下

假设词表大小 V=5，  
真实目标 token 是 `T1`（索引 1），真实分布是：

q=[0,1,0,0,0]

模型预测的概率分布：

p=[0.05,0.70,0.20,0.03,0.02]

交叉熵公式：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755186298975-060ed82f-4299-4573-bb8c-c9197d0c1690.png)

因为q 是 one-hot，只有目标 token 位置是 1，所以公式简化为：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755186348570-ffc9b284-4452-4774-ae5b-74b5c689fc57.png)

在例子中：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755186365700-af703ad5-fdb7-4d6b-bdc5-8d8762818402.png)

最终再对loss做mask求和，再对batch求和，就是一整个batch的loss

在代码中体现如下：

```python
       if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target
```

### SFT
感觉只有mask不一样，SFT的mask过程如下：  
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755188116513-b7a16a2b-b519-4849-900b-1e228bbe0922.png)

也是输入一整段文本计算loss，不过只把标记为 Assistant  部分的文本之外的其他部分全部mask掉，指计算该部分的loss

### 使用Transformer搭建模型并用Deepspeed并行训练
#### 下载配置文件和权重模型
```python
import os
# 设置环境变量，此处使用 HuggingFace 镜像网站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir your_local_dir')
```

#### 加载配置文件和模型
```python
# 加载定义好的模型参数-此处以 Qwen-2.5-1.5B 为例
# 使用 transforemrs 的 Config 类进行加载
from transformers import AutoConfig

# 下载参数的本地路径
model_path = "qwen-1.5b"
config = AutoConfig.from_pretrained(model_name_or_path)
```

其中，AutoConfig作用如下：  
`AutoConfig.from_pretrained(...)`

+ **作用**：只加载**模型的配置文件**（config.json 里的信息），例如：
    - 模型的层数、隐藏维度、注意力头数、词表大小等超参数。
    - 还有 tokenizer 相关的一些配置（但不加载 tokenizer 本身）。
+ **不会加载模型的参数（权重）**，只是一个“结构蓝图”。
+ 典型用法：
+ 具体加载config的方式：
    - **如果是本地目录**
        * 它就会直接在这个目录下找 `config.json`。
        * 如果没找到，就会报错：`OSError: Can't load config for ...`。
+ **如果不是本地目录**（比如拼错路径，或者目录不存在）
    - 🤗 Transformers 会把你传的字符串当作 **Hugging Face Hub 上的仓库 ID**。
    - 用 **正确的 repo 名字**，比如：

```plain
config = AutoConfig.from_pretrained("Qwen/Qwen-1.5B")
```

如果模型是 gated/private，还得先：

```plain
huggingface-cli login
```

然后加载时 `transformers` 会自动带上你的 token。

#### 创建模型并加载模型
之后利用config创建模型（但是并没有加载参数）

```python
# 使用该配置生成一个定义好的模型
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
```

`AutoModelForCausalLM.from_config(config, trust_remote_code=True)`

+ **作用**：根据传入的 **配置对象 (**`**config**`**)****创建模型结构**，但是：
    - **不会加载预训练好的参数**（此时权重是随机初始化的）。
    - 只是按照 config 搭建一个“空白模型”。

横向对比一下：`AutoModelForCausalLM` 其实是 🤗 Transformers 提供的 **“Auto 类”** 之一，用来根据 `config.model_type` 自动匹配合适的模型实现。  不同的Auto模型任务头不一样

| 类名 | 作用 | 典型应用场景 |
| --- | --- | --- |
| **AutoModel** | 只加载“裸”模型（没有任务头），即基础 Transformer 编码器/解码器。 | 获取 hidden states，做 embedding，fine-tune 自定义任务。 |
| **AutoModelForCausalLM** | 自回归语言建模（左到右预测），典型 GPT 系列。 | ChatGPT、Qwen、LLaMA 这种生成任务。 |
| **AutoModelForMaskedLM** | 掩码语言建模（BERT 类）。 | MLM 预训练，填空。 |
| **AutoModelForSeq2SeqLM** | 序列到序列生成（编码器-解码器）。 | 翻译、摘要、T5/BART 等。 |
| **AutoModelForSequenceClassification** | 文本分类（句子级别）。 | 情感分析、意图分类。 |
| **AutoModelForTokenClassification** | 序列标注。 | NER（命名实体识别）、POS 标注。 |
| **AutoModelForQuestionAnswering** | 抽取式问答。 | SQuAD、阅读理解。 |
| **AutoModelForMultipleChoice** | 多选题任务。 | RACE 数据集。 |
| **AutoModelForVision2Seq** | 图像到文本生成。 | Image Captioning（BLIP-2）。 |
| **AutoModelForImageClassification** | 图像分类。 | ViT, ResNet。 |
| **AutoModelForObjectDetection** | 目标检测。 | DETR, YOLOS。 |
| **AutoModelForSemanticSegmentation** | 语义分割。 | SegFormer。 |
| **AutoModelForSpeechSeq2Seq** | 语音到文本。 | Whisper。 |
| **AutoModelForAudioClassification** | 音频分类。 | Wav2Vec2, Hubert。 |


 任务头指的是在 **基础模型（backbone / encoder-decoder）** 的最后，再接一个 **额外的小层（一般是线性层 + softmax / sigmoid 等）**，专门针对某个任务输出结果。  

```python
  # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else: 
            # 只对最后一个位置的输出进行前向传播，推理过程走这边
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

```

加载模型，直接from_pretrained就可以了

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
```

之后加载tokenizer

```python
# 加载一个预训练好的 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

  
 

## 搭建minimind
### 环境搭建
### 模型结构
### 模型训练：
使用框架训一下

### 
## <font style="color:rgb(31, 35, 40);">MoE</font>
MoE的好处是：在相同的预训练<font style="color:rgb(25, 27, 31);">在相同的计算预算条件下，您可以显著扩大模型或数据集的规模。特别是在预训练阶段，与稠密模型相比，混合专家模型通常能够更快地达到相同的质量水平。  
</font><font style="color:rgb(25, 27, 31);">翻译成人话就是，由于scaling law，模型肯定越大越好，MoE可以在计算资源一定的情况下扩大模型的大小，从而扩大模型的表现能力。</font>

<font style="color:rgb(25, 27, 31);">MoE基础结构：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759235701623-eb26b7f8-46d3-47b8-bbcf-157cf3deded8.png)

多了个MoE层，又路由和多个ffn来组成，路由来决定使用什么ffn

因为每次调用的是其中几个专家模型（ffn）作为输出用，所以我们也称这样的模型是稀疏的，这样会有很多问题：

1. 路由分配不均问题

<font style="color:rgb(25, 27, 31);">比如说输入10 个token， </font>**<font style="color:rgb(25, 27, 31);">可能会有五个令牌被路由到同一个专家，而剩下的五个令牌分别被路由到不同的专家。这导致了批量大小的不均匀分配和资源利用效率不高的问题</font>**<font style="color:rgb(25, 27, 31);">。</font>

很烂的最最最一开始的门控网络是这样的：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759236495391-c8844bb7-0b14-4ae1-a400-edce10f8b132.png)

对所有专家E输入了x之后，再经过门控网络去取出n个专家的计算结果，这样为啥要对所有都计算呢？

一般门控网络是这样的：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759236570140-7e14036d-845d-4b60-bfd4-cd8e16cd9670.png)

对x乘wg之后做softmax，之后挑选概率最高的作为选择的专家

稀疏性引入了一些有趣的选择，万一我不选择最高的一个专家呢？

比如说这个引入了噪声再选取出topk个专家的算法：

**如果完全用确定性的 Top-K**，容易出现两个问题：

1. **负载不均衡**：少数专家被频繁选择，其他专家几乎没被用到，导致训练不稳定、浪费参数。
2. **探索性不足**：门控层太“贪心”，只盯着高分的专家，不给其他专家学习机会。

为了解决上述问题，Google 在 **Switch Transformer / GShard** 里引入了 **Noisy Top-K Gating**。  
做法是：

在计算专家分数时，**往每个分数里加一点噪声（通常是高斯噪声）**：

1. s~i=si+ϵi,ϵi∼N(0,σ2)\tilde{s}_i = s_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)s~i=si+ϵi,ϵi∼N(0,σ2)

其中 sis_isi 是专家 i 的原始打分，ϵi\epsilon_iϵi 是噪声。

2. 用带噪声的分数 s~i\tilde{s}_is~i 来做 Top-K 选择。

这样，每次选择专家时会有一些随机性，不是总挑分数最高的专家。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1759236762439-62aca437-a18f-4364-8745-b1c4441b5578.png)

加噪的原因是

这是比较soft的操作，先取出k个专家，之后过了边softmax，获得了概率分布之后求加权和来获得最终的MoE融合结果

上面提到的都是shazzer在LSTM中提到的。

负载均衡也可以从loss的角度解决

还使用了辅助损失，本质上是个正则，让各个专家处理尽可能一致的batch数量。

进一步提升：

<font style="color:rgb(25, 27, 31);">谷歌使用 </font>[**GShard**](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2006.16668)<font style="color:rgb(25, 27, 31);"> 尝试将 Transformer 模型的参数量扩展到超过 6000 亿，除了引入了上一节中讨论的类似辅助损失外，还引入了一些关键变化:</font>

<font style="color:rgb(25, 27, 31);">结构层面：</font>

<font style="color:rgb(25, 27, 31);">每个batch的路由：topk中的top1是根据排名选取的，但是第二个是根据权重比例随机选择的</font>

**<font style="color:rgb(25, 27, 31);">分配batch的上限</font>**<font style="color:rgb(25, 27, 31);">: 我们可以设定一个阈值，定义一个专家能处理多少令牌。如果两个专家的容量都达到上限，令牌就会溢出，并通过残差连接传递到下一层，或在某些情况下被完全丢弃。专家容量是 MoE 中最重要的概念之一。为什么需要专家容量呢？因为所有张量的形状在编译时是静态确定的，我们无法提前知道多少令牌会分配给每个专家，因此需要一个固定的容量因子。</font>

后面23年的swin transformer，微调MoE的技术都可以看看

[https://zhuanlan.zhihu.com/p/674698482](https://zhuanlan.zhihu.com/p/674698482) hugging face给出的这个报告

再到后面就自己搜一搜

## Qwen2.5
一句话总结：

<font style="color:rgb(0, 0, 0);background-color:rgb(249, 250, 251);">Qwen2.5 是 Qwen 团队推出的全面大型语言模型（LLM）系列，在</font>**<font style="color:rgb(0, 0, 0) !important;background-color:rgb(249, 250, 251);">预训练</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(249, 250, 251);">和</font>**<font style="color:rgb(0, 0, 0) !important;background-color:rgb(249, 250, 251);">后训练</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(249, 250, 251);">阶段均有显著提升：预训练阶段将高质量数据集从 7 万亿 token 扩展至</font>**<font style="color:rgb(0, 0, 0) !important;background-color:rgb(249, 250, 251);">18 万亿 token</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(249, 250, 251);">，为常识、专业知识和推理能力奠定基础；后训练阶段通过超 100 万样本的精细监督微调（SFT）及多阶段强化学习（含离线 DPO 和在线 GRPO），大幅优化人类偏好对齐、长文本生成等能力。该系列提供丰富配置，开源模型涵盖</font>**<font style="color:rgb(0, 0, 0) !important;background-color:rgb(249, 250, 251);">0.5B 至 72B 参数</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(249, 250, 251);">的基础模型与指令微调模型（含量化版本），专有模型含 Qwen2.5Turbo 和 Qwen2.5-Plus 两款 MoE 变体；在基准测试中表现顶尖，如开源旗舰模型 Qwen2.5-72B-Instruct 性能媲美参数约为其 5 倍的 Llama-3-405B-Instruct，且可作为基础模型支撑 Qwen2.5-Math 等专业模型的训练，适用于学术与工业场景。</font>

<details class="lake-collapse"><summary id="u39ba1423"><span class="ne-text" style="font-size: 16px">解释：数据集大小</span></summary><p id="u5ad43971" class="ne-p"><br></p><h2 id="mxJdt"><span class="ne-text"><br /></span><span class="ne-text"> </span></h2></details>
<details class="lake-collapse"><summary id="u9b24341a"><span class="ne-text" style="font-size: 16px">解释：强化学习的在线离线</span></summary><h3 id="KDztG"><span class="ne-text">1. 离线强化学习（Offline RL）</span></h3><ul class="ne-ul"><li id="uc7d5cda9" data-lake-index-type="0"><strong><span class="ne-text">数据来源</span></strong><span class="ne-text">：完全依赖 </span><strong><span class="ne-text">已有的人类反馈数据集</span></strong><span class="ne-text">（比如，人类对模型回答的排序/偏好、标注好的评分）。</span></li><li id="u9cf6b046" data-lake-index-type="0"><strong><span class="ne-text">训练方式</span></strong><span class="ne-text">：模型不会和环境实时交互，而是从这些静态数据中学习。</span></li><li id="u76051bf1" data-lake-index-type="0"><strong><span class="ne-text">优势</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ua937afd5" data-lake-index-type="0"><span class="ne-text">不需要实时采样，成本低。</span></li><li id="ue39daa63" data-lake-index-type="0"><span class="ne-text">数据可控，标注质量高。</span></li></ul></ul><ul class="ne-ul"><li id="uda7af251" data-lake-index-type="0"><strong><span class="ne-text">典型应用</span></strong><span class="ne-text">：像 </span><strong><span class="ne-text">DPO（Direct Preference Optimization）</span></strong><span class="ne-text"> 就是典型的离线 RL 算法，它直接利用离线偏好数据进行优化，让模型回答更符合人类偏好。</span></li></ul><p id="ucd49a519" class="ne-p"><span class="ne-text">👉</span><span class="ne-text"> 类比：就像学生通过看往年的标准答案和评分规则，自学怎么答题更受老师喜欢。</span></p><hr id="dh7Yo" class="ne-hr"><h3 id="d260b6c1"><span class="ne-text">2. 在线强化学习（Online RL）</span></h3><ul class="ne-ul"><li id="uaaa47875" data-lake-index-type="0"><strong><span class="ne-text">数据来源</span></strong><span class="ne-text">：模型在训练时会 </span><strong><span class="ne-text">实时生成回答</span></strong><span class="ne-text">，然后通过人类反馈或自动奖励模型来打分，再据此更新策略。</span></li><li id="ue23a716d" data-lake-index-type="0"><strong><span class="ne-text">训练方式</span></strong><span class="ne-text">：模型边生成、边被评估、边更新，形成“交互—反馈—优化”的循环。</span></li><li id="ubce80608" data-lake-index-type="0"><strong><span class="ne-text">优势</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u15d1947e" data-lake-index-type="0"><span class="ne-text">能够探索更丰富的回答空间。</span></li><li id="u7475c7e4" data-lake-index-type="0"><span class="ne-text">奖励函数可动态调整，更贴近真实人类偏好。</span></li></ul></ul><ul class="ne-ul"><li id="u8ceed448" data-lake-index-type="0"><strong><span class="ne-text">典型应用</span></strong><span class="ne-text">：Qwen2.5 里的 </span><strong><span class="ne-text">GRPO（Group Relative Policy Optimization）</span></strong><span class="ne-text"> 就是在线 RL 的一个变体，通过不断采样和比较不同回答，实时优化模型行为。</span></li></ul><p id="u52f0cdd7" class="ne-p"><span class="ne-text">👉</span><span class="ne-text"> 类比：就像学生答题后，老师立即给反馈（比如打分或点评），学生据此改进答题策略。</span></p><p id="ue2ed8f12" class="ne-p"><span class="ne-text"></span></p></details>
### dense模型架构
#### GQA group query attention
参考：

[https://zhuanlan.zhihu.com/p/686149289](https://zhuanlan.zhihu.com/p/686149289)

了解这个之前先了解下KV cache

[https://zhuanlan.zhihu.com/p/662498827](https://zhuanlan.zhihu.com/p/662498827)

这篇说的蛮好的，在每次进行attention score计算的时候，由于mask的效应，**<font style="color:rgb(25, 27, 31);">推理第 xk个字符的时候只需要输入xk-1的字符即可。</font>**

<font style="color:rgb(25, 27, 31);">因此每次只要把xk-1的K和V全部存到cache当中，在需要的时候取出来用，这样每次只需要计算第k个K和V即可的即可</font>

<font style="color:rgb(25, 27, 31);">计算的方式也就是kv矩阵的大小b*s*h*d*d_type再乘上模型的层数d再乘*2，这个数字量很容易上1G，这时候就需要MQA和GQA上场了</font>

<font style="color:rgb(25, 27, 31);">MQA是只用一个k和v，当时再经过wq，wk，wv的时候把矩阵投影成了（b,s,head,head_dim)</font>

<font style="color:rgb(25, 27, 31);">现在因为k,v占比太大了，就把投影成了(b,s,head_dim)，之后通过repeat重复并扩充成head个数，并做attention map</font>

<font style="color:rgb(25, 27, 31);">而GQA就是专门设计了个args.kv_head，之后经过wk,wv投影成这个大小。在score之前，rope之后重复，并做attention map</font>

#### Dual chunk attention
这里主要参考ChatGPT

用来减少复杂度，原来计算复杂度是n^2d

现在DCA吧attention拆分成了多个chunk，chunk内计算attnention，chunk间是对chunk内的token进行抽取，并计算chunk之间的score。综合计算复杂度如下计算：

<details class="lake-collapse"><summary id="u8ba9e9b5"><span class="ne-text">DCA score计算复杂度</span></summary><h3 id="NViEB"><span class="ne-text">局部块内 attention 的逻辑</span></h3><p id="ue41c46d9" class="ne-p"><span class="ne-text">在 </span><strong><span class="ne-text">dual chunk attention</span></strong><span class="ne-text"> 里，我们把序列切成若干个 </span><strong><span class="ne-text">chunk</span></strong><span class="ne-text">，每个 chunk 大小是 </span><span class="ne-text">c</span><span class="ne-text">c</span><span class="ne-text">c</span><span class="ne-text">。</span></p><ul class="ne-ul"><li id="u4163adfb" data-lake-index-type="0"><span class="ne-text">假设总长度是 </span><span class="ne-text">n</span><span class="ne-text">n</span><span class="ne-text">n</span><span class="ne-text">，那么总共有 </span><span class="ne-text">n</span><span class="ne-text">/</span><span class="ne-text">c</span><span class="ne-text">n/c</span><span class="ne-text">n</span><span class="ne-text">/</span><span class="ne-text">c</span><span class="ne-text"> 个 chunk。</span></li><li id="u0220ad69" data-lake-index-type="0"><strong><span class="ne-text">局部块内 attention</span></strong><span class="ne-text">：<br /></span><span class="ne-text">每个 token </span><strong><span class="ne-text">只和自己所在的 chunk 内的 token 交互</span></strong><span class="ne-text">，而不是和整个序列。</span></li></ul><hr id="ZQk8b" class="ne-hr"><h3 id="YXn5V"><span class="ne-text">复杂度推导</span></h3><ol class="ne-ol"><li id="u590e409c" data-lake-index-type="0"><strong><span class="ne-text">一个 token 的计算量</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u669eef9d" data-lake-index-type="0"><span class="ne-text">每个 token 需要和同 chunk 的 </span><span class="ne-text">c</span><span class="ne-text">c</span><span class="ne-text">c</span><span class="ne-text"> 个 token 做 dot-product（QK 乘法），</span></li><li id="u85b04ead" data-lake-index-type="0"><span class="ne-text">每次 dot-product 是 </span><span class="ne-text">O</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">)</span><span class="ne-text">O(d)</span><span class="ne-text">O</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">)</span><span class="ne-text"> 的运算（因为向量维度是 </span><span class="ne-text">d</span><span class="ne-text">d</span><span class="ne-text">d</span><span class="ne-text">）。</span></li><li id="ucc741d8f" data-lake-index-type="0"><span class="ne-text">所以一个 token 的复杂度 = </span><span class="ne-text">O</span><span class="ne-text">(</span><span class="ne-text">c</span><span class="ne-text">⋅</span><span class="ne-text">d</span><span class="ne-text">)</span><span class="ne-text">O(c \cdot d)</span><span class="ne-text">O</span><span class="ne-text">(</span><span class="ne-text">c</span><span class="ne-text">⋅</span><span class="ne-text">d</span><span class="ne-text">)</span><span class="ne-text">。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="u1dad9ecb" data-lake-index-type="0"><strong><span class="ne-text">n 个 token 的计算量</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ue9c11c90" data-lake-index-type="0"><span class="ne-text">一共有 </span><span class="ne-text">n</span><span class="ne-text">n</span><span class="ne-text">n</span><span class="ne-text"> 个 token，</span></li><li id="u5b883b83" data-lake-index-type="0"><span class="ne-text">所以总复杂度 = O(n⋅c⋅d)O(n \cdot c \cdot d)O(n⋅c⋅d)。</span></li></ul></ul><p id="ud9f41677" class="ne-p"><span class="ne-text"></span></p><h3 id="6e3b8327"><span class="ne-text">(2) 跨块（global / summary）attention</span></h3><p id="u69fb6f72" class="ne-p"><span class="ne-text">这里有多种实现方式，但一般不做稠密的全局 attention，而是</span><strong><span class="ne-text">块级稀疏</span></strong><span class="ne-text">：</span></p><ul class="ne-ul"><li id="u09cf9d4b" data-lake-index-type="0"><span class="ne-text">如果每个 chunk 提取 1 个 summary token（或少量 </span><span class="ne-text">s</span><span class="ne-text">s</span><span class="ne-text">s</span><span class="ne-text"> 个 token），<br /></span><span class="ne-text">那么总 summary token 数量 </span><span class="ne-text">≈</span><span class="ne-text">m</span><span class="ne-text">\approx m</span><span class="ne-text">≈</span><span class="ne-text">m</span><span class="ne-text">。</span></li><li id="u86e82b20" data-lake-index-type="0"><span class="ne-text">summary token 之间全连接：</span></li></ul><p id="u2f5c9cf0" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1759232324680-ff76887a-7c6f-4128-bec5-19d76136bdb6.png" width="262.6666666666667" id="u754de320" class="ne-image"></p><ul class="ne-ul"><li id="u882ad564" data-lake-index-type="0"><span class="ne-text">或者每个 token attends 到 summary token：</span></li></ul><p id="u07097785" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1759232331703-20341d5f-5f98-4a0f-b201-fd7d2a41bcc2.png" width="268" id="uc639dde3" class="ne-image"></p></details>


<font style="color:rgb(25, 27, 31);"> </font>



<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);"></font>

## 
