---
title: AI infra
urlname: vzdegyu9s524idvl
date: '2025-08-22 12:58:40'
updated: '2025-11-10 10:23:44'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1755842288827-339ef51f-cbcd-4bd1-8670-e8ddb49ab192.png'
description: '1. deepspeed学习参考https://zhuanlan.zhihu.com/p/1940441036477436779'
---
## deepspeed学习
参考

1. [https://zhuanlan.zhihu.com/p/1940441036477436779](https://zhuanlan.zhihu.com/p/1940441036477436779)
2. [https://datawhalechina.github.io/happy-llm/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B?id=_42-%e5%a6%82%e4%bd%95%e8%ae%ad%e7%bb%83%e4%b8%80%e4%b8%aa-llm](https://datawhalechina.github.io/happy-llm/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B?id=_42-%e5%a6%82%e4%bd%95%e8%ae%ad%e7%bb%83%e4%b8%80%e4%b8%aa-llm)

<font style="color:rgb(25, 27, 31);">进入大模型时代后，一张卡的显存不足以加载完整的模型或者完成一个训练过程。</font>

### <font style="color:rgb(25, 27, 31);">显存占用</font>
<font style="color:rgb(25, 27, 31);">首先要弄清楚的是，消耗显存的都有哪些？</font>

    - <font style="color:rgb(25, 27, 31);">模型的参数。</font>
    - <font style="color:rgb(25, 27, 31);">前向过程中，一些中间计算结果以及激活值（即激活函数的执行结果）。</font>
    - <font style="color:rgb(25, 27, 31);">反向过程中，每个参数的梯度值。</font>
    - <font style="color:rgb(25, 27, 31);">优化器的状态。比如 </font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">adam</font>`<font style="color:rgb(25, 27, 31);"> 算法，需要为每个参数再保存一个一阶动量和二阶动量</font>
    - <font style="color:rgb(25, 27, 31);">MasterCopy</font>

<font style="color:rgb(25, 27, 31);">举个例子：</font>

<font style="color:rgb(25, 27, 31);">加入模型参数大小为1b，使用混合精度</font>

**Mixed-precision (FP16 params + FP32 master + Adam in FP32) 的常见情形**：

+ <font style="color:rgb(25, 27, 31);">params FP16 = </font>`<font style="color:rgb(25, 27, 31);">2.0 GB</font>`
+ <font style="color:rgb(25, 27, 31);">master params FP32 = </font>`<font style="color:rgb(25, 27, 31);">4.0 GB</font>`
+ <font style="color:rgb(25, 27, 31);">gradients (通常 FP16 存，或在反传时以 FP16/FP32 混合) 约 = </font>`<font style="color:rgb(25, 27, 31);">2.0 GB</font>`
+ <font style="color:rgb(25, 27, 31);">Adam 的两个 momentum（通常以 FP32 存） = </font>`<font style="color:rgb(25, 27, 31);">2 * 4.0 GB = 8.0 GB</font>`<font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">合计（不含 activations） = </font>`<font style="color:rgb(25, 27, 31);">2 + 4 + 2 + 8 = 16.0 GB</font>`<font style="color:rgb(25, 27, 31);">（注意：和纯 FP32 的 16GB 数字看起来相似，原因是 optimizer moments 与 master copy 主导）</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">模型的参数好理解。</font>

#### <font style="color:rgb(25, 27, 31);">激活值</font>
<font style="color:rgb(25, 27, 31);">看看激活值，这里指的不是激活层，是</font>**前向传播时，输入数据经过模型参数计算后产生的中间结果**<font style="color:rgb(25, 27, 31);">。  </font>

<font style="color:rgb(25, 27, 31);"> 特点：  </font>

+ **只在前向/反向时需要**<font style="color:rgb(25, 27, 31);">，训练完成后推理时很多可以丢弃或重算。</font>
+ <font style="color:rgb(25, 27, 31);">体量常常比参数还大！尤其是长序列 / 大 batch 时。</font>

<font style="color:rgb(25, 27, 31);">举个例子：</font>

假设：hidden size = 2048，batch size = 32，seq_len = 512。

+ **参数（固定）**<font style="color:rgb(25, 27, 31);">：</font>
    - <font style="color:rgb(25, 27, 31);">一个 FFN 的权重：</font>`<font style="color:rgb(25, 27, 31);">W1: 2048x8192, W2: 8192x2048</font>`
    - <font style="color:rgb(25, 27, 31);">大小约几十 MB，不随 batch/seq_len 变化。</font>
+ **激活（随输入变化）**<font style="color:rgb(25, 27, 31);">：</font>
    - <font style="color:rgb(25, 27, 31);">每一层的 hidden states：</font>`<font style="color:rgb(25, 27, 31);">32 x 512 x 2048 ≈ 33.5M</font>`<font style="color:rgb(25, 27, 31);"> 个 float</font>
    - <font style="color:rgb(25, 27, 31);">FP16 存储 ≈ 67 MB（这一层的 activations）。</font>
    - <font style="color:rgb(25, 27, 31);">多层叠加，整个 batch 的 activations 就会很大。</font>

<font style="color:rgb(25, 27, 31);">所以训练大模型时，</font>**激活显存占用常常比参数还多**<font style="color:rgb(25, 27, 31);">。这也是为什么要有 </font>**activation checkpointing（重算）**<font style="color:rgb(25, 27, 31);"> 或 </font>**activation offload**<font style="color:rgb(25, 27, 31);">。这些提到的技术等下再说</font>

<font style="color:rgb(25, 27, 31);">用处的话主要在反向传播用，计算 loss 之后，会通过链式法则对参数求梯度，在这个过程中，需要用到之前存下来的 </font>**激活值**<font style="color:rgb(25, 27, 31);">，因为它们决定了梯度怎么传播。</font>

<font style="color:rgb(25, 27, 31);">梯度值也好理解。</font>

> 比如训练Lora的时候，使用的是Qwen7 instruct 7b，激活值大小就是 隐藏层3500左右*token平均大小1024
>
> *batch8 *层数28 = 
>

#### <font style="color:rgb(25, 27, 31);">优化器</font>
<font style="color:rgb(25, 27, 31);">之后理解下优化器的状态，这里因为好久没有看优化器了，恶补了一下优化器知识</font>

<font style="color:rgb(25, 27, 31);">参考：</font>

1. [https://www.zhihu.com/tardis/zm/art/208178763?source_id=1005](https://www.zhihu.com/tardis/zm/art/208178763?source_id=1005)
2. [https://www.bilibili.com/video/BV1NZ421s75D/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1NZ421s75D/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

<font style="color:rgb(25, 27, 31);">从头开始说</font>

##### <font style="color:rgb(25, 27, 31);">SGDM</font>
相比于传统的SGD多了个动量

以下两种写法我都看见过，原始的论文按照a写法，但是感觉b更好说一点，先按b来吧

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755842475632-31d74027-1705-4272-a498-47b083128b74.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755844282214-1487ba28-70db-4ff6-867a-5ed4d9a40a16.png)

<font style="color:rgb(25, 27, 31);">每一个动量跟前面的所有动量有关，并且之前所有的动量是通过指数加权平均求到了一起，μ这个系数用来控制前面所有梯度的占比，一般0.9</font>

<font style="color:rgb(25, 27, 31);">负号是因为梯度的反方向。</font>

<font style="color:rgb(25, 27, 31);">有个问题是一开始v0为0，会导致v1= 0.7*0-0.3*g1，会距离一开始的g1偏离很远，对短序列梯度更新效果不好</font>

<font style="color:rgb(25, 27, 31);">于是可以用下式对vt进行修正</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755843858590-c9e7aba7-44cb-43f2-8855-f28af576f2ac.png)

<font style="color:rgb(25, 27, 31);">其中t>=1,越小β越大，放大的越多，使原序列得以修正。</font>

<font style="color:rgb(25, 27, 31);">好问题是为什么这么修正？详细请了解无偏估计细节（我没了解）</font>

<font style="color:rgb(25, 27, 31);">还有个问题，就是还是非常看学习率η的选取，超参</font>**学习率难以选择**

+ <font style="color:rgb(25, 27, 31);">SGDM 依然对 η（学习率）或者说α 非常敏感。太大容易发散，太小收敛慢。</font>
+ <font style="color:rgb(25, 27, 31);">在不同参数方向（例如某些维度梯度大、某些维度梯度小），用同一个学习率会导致收敛效率不佳。</font>

<font style="color:rgb(25, 27, 31);">GPT说同样也有一些别的问题，引申出了一下别的解法，没有细细研究，列举如下：</font>

> 1. **在非凸优化问题中容易卡住**
>     - 深度学习的 loss surface 很复杂（局部极小值、鞍点、平坦区）。
>     - SGDM 可能会停在鞍点附近，或者在平坦区域走得非常慢。
> 2. **各个参数维度更新速度不均衡**
>     - SGDM 没有考虑 **不同参数梯度的尺度差异**。
>     - 比如，某些参数的梯度非常小，更新几乎停滞，而另一些梯度大的参数更新过快。
> 3. **动量积累可能导致过冲 (overshoot)**
>     - 动量会不断积累，如果梯度方向突然改变，参数可能会“冲过头”，在最优点附近震荡。
>
> ### (1) Nesterov Accelerated Gradient (NAG)
> + 改进点：在计算梯度时，不是对当前位置算，而是对 **“预估的下一步位置”** 算。
>
> 更新规则：
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755842802771-2ba53c40-9d79-4394-8503-9e2c88fc2b72.png)
>
> + 优点：能“提前感知”方向变化，避免动量导致的 overshoot，收敛更快更稳
>

##### Adagrad
解决学习率的一个方法就是Adagrad提到的

<font style="color:rgb(18, 18, 18);">它利用迭代次数和累积梯度，对学习率进行自动衰减，2011年提出。从而使得刚开始迭代时，学习率较大，可以快速收敛。而后来则逐渐减小，精调参数，使得模型可以稳定找到最优点。其参数迭代公式如下</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755842977608-cb1e91da-f155-4062-8c47-c9344432b183.png)

其中 Gt 是历史梯度平方和 。

<font style="color:rgb(18, 18, 18);">主要缺点就是没有考虑迭代衰减。极端情况，如果刚开始的梯度特别大，而后面的比较小，则学习率基本不会变化了，也就谈不上自适应学习率了。</font>

+ **母单调增长**<font style="color:rgb(18, 18, 18);">：因为</font>![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755843194692-0e44ea13-d804-4d9b-9618-b7e8eb4ab905.png)<font style="color:rgb(18, 18, 18);"> 是累加的，会不断增大 → </font>![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755843220505-aa07b805-e9ae-43b4-944c-962cd423f33a.png)<font style="color:rgb(18, 18, 18);"> 会单调衰减到 0（大约按 </font>![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755843233923-ecb6016d-ff3f-41c1-addc-26cd9b3a10f1.png)<font style="color:rgb(18, 18, 18);"> 缩小）。</font>
+ <font style="color:rgb(18, 18, 18);">结果：训练会在某个时刻变得极慢，几乎停止学习（尤其是在非凸深度网络里）。</font>
+ **对非稀疏、大规模深网效果不一定好**<font style="color:rgb(18, 18, 18);">：衰减过快导致后期学习受阻。</font>

<font style="color:rgb(18, 18, 18);"></font>

<font style="color:rgb(18, 18, 18);">这个问题在RMSProp中得到了修正</font>

##### RMSprop
<font style="color:rgb(18, 18, 18);">结合了adagrad和SGDM。</font>

<font style="color:rgb(18, 18, 18);">它与Adagrad基本类似，只是加入了SGDM的</font>**<font style="color:rgb(18, 18, 18);">迭代衰减</font>**<font style="color:rgb(18, 18, 18);">，2013年提出，如下</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755843091201-610d778d-3f2c-475c-8dc2-bddff1cf98b7.png)

<font style="color:rgb(18, 18, 18);">观察上式和Adagrad的区别，在于RMSProp中，梯度累积不是简单的前t-1次迭代梯度的平方和了，</font>

<font style="color:rgb(18, 18, 18);"> 而是采用 </font>**指数加权移动平均**<font style="color:rgb(18, 18, 18);"> 来平滑梯度平方：  </font>

<font style="color:rgb(18, 18, 18);">简单理解就是学习率除以前t-1次迭代的梯度的指数加权平方和。加入衰减时make sense的，因为与当前迭代越近的梯度，对当前影响应该越大。另外也完美解决了某些迭代梯度过大，导致自适应梯度无法变化的问题。</font>

<font style="color:rgb(18, 18, 18);">有个自然而然的问题是RMS这里为什么这里动量不使用一次方，而模仿adagrad使用了二次方做指数移动平均？</font>

关键原因是还是学习率是标量，想要根据之前的移动距离进行自适应调整，不想带方向：

+ **动量（一次方）**<font style="color:rgb(18, 18, 18);">：追踪“方向”（梯度平均值），让参数更新时避免高频震荡。</font>
+ **RMSprop（二次方）**<font style="color:rgb(18, 18, 18);">：追踪“幅度”（梯度平方的平均），用来动态调节学习率。</font>

<font style="color:rgb(18, 18, 18);">换句话说：</font>

+ ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755845541199-d5387372-7759-4964-882d-22c074094903.png)

<font style="color:rgb(18, 18, 18);">如果只用一次方来做学习率缩放，那它会受到梯度正负号抵消的影响（例如连续两个 +5 和 -5 的梯度，平均梯度为 0，但强度其实很大）。  
</font><font style="color:rgb(18, 18, 18);">而平方后能真实反映梯度的“大小/能量”，不管正负。这样 RMSprop 才能根据梯度大小来缩放每个参数的学习率。</font>

##### Adam
Adam的motivation就是大一统，把前面所有的方法综合了一边，详细说说

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755845052263-3e4ad842-fc9d-4964-8190-fe57f609f3ea.png)

可以看出来Vm就是Momentum中的一阶动量,Sw就是RMSprop中的二阶动量

之后对两个动量进行修正，修正原因参考momentum

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755845095286-33a1169d-8b6a-48c2-ab67-6f3cb059729a.png)	

最后得出Adam的更新方法：  
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755845291027-f247234a-2adc-45fc-8b7f-d1478178228f.png)

眼熟不，就是结合了RMSprop和SGDM，也就是对学习率的更新使用了RMSProp，梯度的更新项使用了SGDM。

因此可以知道存储的参数主要就是v和s这两个一阶二阶的动量

至于后面的Adam和SGDM的进一步优化就不在这里讨论了，毕竟是了解deepspeed。。。



深入理解下RMSprop，要用二阶动量的原因是希望学习率下降时符合一开始下降较大，后面下降较小的。

如果使用一阶动量会使学习率那块收敛的不太稳定，毕竟一阶动量求和的时候可能会出现波动



##### 优化器在显存中的占用
假设参数总数 P。

+ 每个参数本身（模型权重）如果用 FP16 存：占用 P×2 bytes。
+ master weights 通常以 FP32 存（mixed precision）：占用 P×4bytes。
+ 梯度（grad）通常与参数同 dtype（FP16 或 FP32）：占用 P×dtype_bytes
+ **m、v（通常 FP32）**：占用 2×P×4bytes。

所以 Adam 的 m/v 总占用约等于 **参数量的两倍（以 FP32 计）**，这常常是 optimizer-related memory 里最大的一块。

**例如（1e9 参数）**：

+ params FP16 = 2.0 GB
+ master FP32 = 4.0 GB
+ m + v (FP32) = 8.0 GB ← 就是很大的来源

假设下为什么不能降精度（GPT）

+ m 和 v 是历史累积（exponential moving averages），直接丢弃会破坏优化轨迹导致训练不收敛或收敛变差。
+ 把 m/v 直接用 FP16 存可能导致数值不稳定（尤其 v 包含平方项，动态范围大）。因此常见实现选 FP32 保存，或用特殊压缩方法（见下）。



#### MasterCopy
补一下梯度更新的工程向知识：

 在很多训练场景下（特别是 **混合精度训练 FP16/BF16**），模型的参数会以 **低精度 (FP16/BF16)** 存储和前向/反向计算，以节省显存和加快速度。但在更新权重时，如果直接在低精度参数上做累加，会出现 **数值精度不足、累积误差过大** 的问题。  

因此，大多数优化器（如 **Adam, SGD+momentum**）会维护一份 **FP32 精度的参数副本**，这份副本就叫 **master params**：

+ **模型参数（FP16/BF16）** → 用来前向和反向传播，省显存和加速
+ **master params（FP32）** → 用来保存权重的高精度拷贝，保证更新时的稳定性
+ 每次更新步骤：
    1. 梯度在 FP16/BF16 中计算出来
    2. 梯度转换到 FP32
    3. 在 FP32 的 master params 上进行更新（比如 SGD 或 Adam）
    4. 更新后的 FP32 参数再 cast 回 FP16/BF16，用于下一次训练

这里假设模型参数是 **1B（10^9 个参数）**：

+ 每个参数在 **FP32** 下占 `4 byte`

所以 master params 需要：

+ 109×4 byte=4GB10^9 \times 4 \text{ byte} = 4 \, \text{GB}109×4 byte=4GB

也就是说，**master params 本质上就是完整的 FP32 权重副本**。

### 单一精度可能出现的问题：
照抄：[https://zhuanlan.zhihu.com/p/103685761](https://zhuanlan.zhihu.com/p/103685761)

fp32会出现计算很慢，并且显存占用很大的情况，具体不细说了

#### fp16
##### 数据溢出
<font style="color:rgb(25, 27, 31);"> fp16 的有效的动态范围约为 ( </font><font style="color:rgb(25, 27, 31);">2−24∽65504</font><font style="color:rgb(25, 27, 31);"> )，比单精度的float要狭窄很多。对于深度学习而言，最大的问题在于 Underflow（下溢出），在训练后期，例如激活函数的梯度会非常小， 甚至在梯度乘以学习率后，值会更加小。</font>

##### 舍入误差
<font style="color:rgb(25, 27, 31);">何为舍入误差，引用[2]中的一张图说的比较透彻：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755939495070-437a670f-c29a-4f2a-b558-b110ff4e1654.png)

<font style="color:rgb(25, 27, 31);">这个例子非常直观的阐述了『舍入误差』这个说法。而至于上面提到的，FP16的最小间隔是一个比较玄乎的事，在</font>[wikipedia](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Half-precision_floating-point_format)<font style="color:rgb(25, 27, 31);">的引用上有这么一张图： 描述了 fp16 各个区间的最小gap。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755939510806-73a37b34-04c1-4327-94a9-886b73f0da56.png)

##### 精度丢失问题：
<font style="color:rgb(25, 27, 31);">在论文中还提到一个『计算精度』的问题：在某些模型中，fp16矩阵乘法的过程中，需要利用 fp32 来进行矩阵乘法中间的累加(accumulated)，然后再将 fp32 的值转化为 fp16 进行存储。 换句不太严谨的话来说，也就是利用</font><font style="color:rgb(25, 27, 31);"> </font>**<font style="color:rgb(25, 27, 31);">利用fp16进行乘法和存储，利用fp32来进行加法计算</font>**<font style="color:rgb(25, 27, 31);">。 这么做的原因主要是为了减少加法过程中的舍入误差，保证精度不损失。</font>

<font style="color:rgb(25, 27, 31);">在这里也就引出了，为什么网上大家都说，只有 Nvidia Volta 结构的 拥有 TensorCore 的CPU(例如V100)，才能利用 fp16 混合精度来进行加速。 那是因为 TensorCore 能够保证 fp16 的矩阵相乘，利用 fp16 or fp32 来进行累加。在累加阶段能够使用 FP32 大幅减少混合精度训练的精度损失。而其他的GPU 只能支持 fp16 的 multiply-add operation。这里直接贴出原文句子：</font>

<font style="color:rgb(83, 88, 97);">Whereas previous GPUs supported only FP16 multiply-add operation, NVIDIA Volta GPUs introduce Tensor Cores that multiply FP16 input matrices andaccumulate products into either FP16 or FP32 outputs</font>

### 混合精度前反向传播过程
混合精度中的使用fp32的部分，很大部分都是为了解决前面提到的两个问题，舍入误差和over/underflow，重点提一下维护master copy，loss scale还有计算中的fp32累加这三个过程。具体就融入到混合精度的传播过程里说

#### 取数据与预处理
 把下个 mini-batch 从 **pinned CPU** 拷到 **GPU**（非阻塞），做数据增强/归一化。  

+ **数据**：如果你的 **dataloader** 读出来的是 `fp32` 张量，那送到模型时，通常会做 **cast**：
    - 在 **混合精度训练 (AMP)** 里，大多数情况会把它转成 **fp16/bf16** 再喂给模型。
    - 这样第一层的 **权重 (fp16)** × **输入 (fp16)** → 输出也是 `fp16`。

如果你不转换，直接 `fp32` 输入 × `fp16` 权重，实际上也能算，算子会强制做精度匹配，一般会把权重 promote 成 `fp32`，这样就丧失了省显存/省带宽的优势了，所以实践中都会统一成半精度。

+ **位置**：CPU→GPU（`non_blocking=True`）。
+ **释放**：老 batch 的张量在下一轮开始前就绪可释放。

####  前向（autocast 半精度计算）
前向因为没有需要梯度更新的地方，所以激活值，权重保存都需要使用fp16进行保存。但是为了梯度更新方便存了一个fp32的副本，就是mastercopy这个部分，等到反向的时候具体说

整个前向的过程大概如下

+ **环境**：`with autocast(dtype=torch.float16 or bfloat16):`
+ **读**：`param_fp16`（模型半精度权重）
+ **写**：各层 **激活（activations）**（半精度），存 GPU，供反向用
+ **输出**：`logits`（半精度/混合），随后转 FP32 计算 `loss`（数值更稳）
+ **释放**：无（激活需保留到反向；若做 activation checkpoint，则保留更少的中间态，可重算）

补充下细节，在每层的activation存储之前肯定经过了矩阵乘法，但是过程有一个地方会存在精度丢失的情况，就是前面提到的计算精度问题，这时候不能使用fp16进行累加，详细说明下在经过每个层时矩阵乘法的过程。

+ 乘法部分：在 Tensor Core 上是 **FP16 × FP16** → 结果会先转成 FP32（不是直接截断为 FP16）。
+ 累加部分：很多个乘积相加时，用 **FP32 累加**（避免精度灾难，如果也用 FP16 累加，信息会迅速丢失）。
+ 最后结果：等到所有求和结束后，再把 CijC_{ij}Cij 从 FP32 cast 回 FP16，存储成 activations。

这里我当时还有个问题，按理说存了fp32之后不是占用的显存更大了吗？实际上

+ 这些 FP32 累加值只存在于 **寄存器（registers）或片上 SRAM（Tensor Core 寄存阵列）**，不会写回显存。
+ 等到整个 tile（分块矩阵）计算完成后，才把结果写回显存（FP16 或 FP32，取决于你设定的输出精度）。

也就是 **中间 FP32 累加值不会占用显存**，只消耗计算单元内部的寄存器。具体寄存器的相关知识我就没有细了解了

#### loss 计算与 loss scaling
之前算到的最后一层激活层由16cast到32，之后和target（int32)算loss。

同时对loss做了一个loss scaling操作，为了解决之前提到的第二个问题over/underflow，因为梯度太小了

+ **做什么**：用 FP32 计算 `loss`；然后 **缩放**：`scaled_loss = loss * scale`
+ **数据**：`loss`（FP32 标量），`scale`（FP32 标量）
+ **目的**：扩大反向链路上的数值动态范围，减少 FP16 下溢/NaN。

> <font style="color:rgb(25, 27, 31);">Loss Scale 主要是为了解决 fp16 underflow 的问题。刚才提到，训练到了后期，梯度（特别是激活函数平滑段的梯度）会特别小，fp16 表示容易产生 underflow 现象。 下图展示了 SSD 模型在训练过程中，激活函数梯度的分布情况：可以看到，有67%的梯度小于</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">2</font><font style="color:rgb(25, 27, 31);">−</font><font style="color:rgb(25, 27, 31);">24</font><font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">，如果用 fp16 来表示，则这些梯度都会变成0。</font>
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1755941019546-5cb50f5c-d555-43d9-abca-800b69a803fb.png)
>
> <font style="color:rgb(25, 27, 31);">为了解决梯度过小的问题，论文中对计算出来的loss值进行scale，由于链式法则的存在，loss上的scale会作用也会作用在梯度上。这样比起对每个梯度进行scale更加划算。 scaled 过后的梯度，就会平移到 fp16 有效的展示范围内。</font>
>
> <font style="color:rgb(25, 27, 31);">这样，scaled-gradient 就可以一直使用 fp16 进行存储了。只有在进行更新的时候，才会将 scaled-gradient 转化为 fp32，同时将scale抹去。论文指出， scale 并非对于所有网络而言都是必须的。而scale的取值为也会特别大，论文给出在 8 - 32k 之间皆可。</font>
>

#### 反向传播
1. 梯度的反向传播

之前算的32的loss会cast成16当作grad一直往前传，所以grad都是以16的形式存储的 

    - **做什么**：`scaled_loss.backward()`
    - **生成**：**梯度 **`**param.grad**`**（半精度）** 以及中间反向张量
    - **释放**：反向一层层回溯时，前向激活被依次释放（checkpoint 会触发前向重算，进一步降峰值显存）
2. 对梯度缩放
+ **做什么**：GradScaler 做
+ `unscale_(optimizer)`：把半精度梯度按 `scale`**就地缩小**到正确数值域，并可转 FP32 缓冲以便后续更新/裁剪。
+ `check_inf_nan`：如有溢出/NaN，**跳过本次 step**，只降 `scale`，不更新 master/状态。
    - 就是这里的sclae过了，为了解决舍入误差从而导致梯度过大，从而溢出了，所以设置一下
+ **可选**：**梯度裁剪**（clip-norm/clip-value）——在 **unscale 之后**、**step 之前** 执行，通常在 FP32 梯度上做。
3. 优化器更新
    - **输入**：之前前向中提到的`param_fp32_master`（FP32 主权重）、16cast成的FP32 梯度（unscale 后）、优化器状态（FP32）
    - **SGD**：
        * 动量：`v = μ v - lr * grad_fp32`
        * 更新：`param_fp32_master += v`（或无动量：`- lr * grad`）
        * **权重衰减（AdamW 风格的 decoupled）**：`param_fp32_master *= (1 - lr*wd)`
    - **Adam/AdamW**：
        * `m = β1*m + (1-β1)*grad_fp32`
        * `v = β2*v + (1-β2)*grad_fp32^2`
        * （可做 bias-correction）
        * `param_fp32_master -= lr * m / (sqrt(v)+eps)`
        * **AdamW**：与梯度解耦的 weight decay 同样直接对 `param_fp32_master` 衰减
    - **释放**：无（状态要保留供下步使用）

> 工程实现多用**融合 kernel（fused optimizers）**：把 `m,v` 更新与参数更新合并，减少读写与 kernel 启动开销。
>

注意m和v也得使用32保存，否则精度丢的太多了，每次更新的时候最需要精度了。合着真正更新梯度的过程就梯度是后cast成的32，剩下的m,v,parm都是32

4. 回写半精度参数（供下轮前/后向）
    - **做什么**：`param_fp16.copy_(param_fp32_master.to(dtype=fp16_or_bf16))`
    - **位置**：GPU 上 **就地** 覆写半精度权重
    - **意义**：下一轮前/后向仍用半精度算子与 TensorCore

> FSDP/ZeRO：只回写**当前 shard/当前子模块**；非活跃 shard 可留在 CPU/NVMe，按需再拉回。
>

#### 时序（单卡简化）——一口气看懂
1. **GPU(半精度)**：前向→存激活
2. **GPU(半精度)**：`loss`（FP32）→`scaled_loss`→反向→得半精度梯度
3. **GPU(FP32)**：`unscale_`（半精度 grad → 正确数值域 / FP32 缓冲）→可选裁剪
4. **GPU(FP32)**：在 master 上 **optimizer.step**（更新 `param_fp32_master` 与状态）
5. **GPU(半/脑)**：把 master **cast 回半精度** 覆写模型参数
6. **CPU/GPU**：`scaler.update()`；`zero_grad(None)`；预取下批

分布式/Offload 时，3–5 步会有 **通信（all-reduce/shard gather/scatter）** 与 **异步 H2D/D2H**；DeepSpeed ZeRO-Offload 会把优化器状态（甚至参数）放 CPU/NVMe，更新时**按需搬运**再写回。

---

#### 关键数据的 dtype/位置速查
| 数据 | 用途 | dtype | 典型位置 |
| --- | --- | --- | --- |
| `param_fp16` | 前/后向计算 | FP16/BF16 | GPU |
| **master **`**param_fp32**` | 精度更新 | FP32 | GPU（或分片/CPU/NVMe） |
| 梯度（反向产出） | 初始梯度 | FP16/BF16 | GPU |
| 梯度（unscale 后） | 用于 step/裁剪 | FP32 | GPU |
| Optimizer 状态（m/v/动量） | 累积历史 | FP32 | GPU（或 CPU/NVMe with offload） |
| 激活 | 反向所需 | FP16/BF16 | GPU（checkpoint 时更少） |
| `scale`<br/>（GradScaler） | 放大系数 | FP32 | CPU/GPU |




















### DDP
#### DP
[https://www.bilibili.com/video/BV1mm42137X8/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1mm42137X8/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

CPU从硬盘之中读取之后分发放到所有GPU当中。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756211191285-06101c68-eeb2-452c-b375-18ef5dd35cab.png)

接下来的计算在每个GPU之中进行

在反传计算梯度之后，其他卡再把梯度全部同步到GPU0做平均

问题1：

这样会导致GPU0通信量过高，因为GPU梯度通信发送都是靠卡0

问题2：

单进程多线程，PythonGIL只能占一个CPU核做处理

详细解释一下问题2：

DP 的工作方式（单进程 + 多线程）

+ **DataParallel** 其实就是**一个进程**里跑着**多个线程**，去控制多张 GPU，在 DP 里，0 号 GPU 的主线程负责：  
    1. 主进程把数据切分成 N 份；
    2. 主进程把模型复制到每张 GPU 上；
    3. 各线程在 GPU 上算前向/反向；
    4. 最后所有梯度/结果都**汇总到 0 号 GPU**，在那做优化器更新。

👉 这导致几个问题：

+ **Python GIL（全局解释器锁）**：多线程模式下，Python 层的逻辑没法真正并行（虽然 CUDA 内核释放了 GIL，但线程的 Python 调度部分还是有开销）。
+ **0 号 GPU 负担重**：因为它既负责自己的一份计算，还要做梯度归并和优化更新，容易成为瓶颈。
+ **扩展性差**：只能在**单机**使用，没法直接用在多机场景。

其中对GIL有如下几个问题：  
1. 什么是 GIL？

+ Python（准确说是 **CPython 解释器**）内部有个叫 **GIL** 的锁。
+ 这个锁的作用是：**同一时间，只允许一个线程执行 Python 字节码**。
+ 所以在 **多线程** 程序里，即使有多个 CPU 核心，Python 解释器也只能“轮流”让线程执行。

2.为什么会有 GIL？

+ CPython 里的内存管理（比如垃圾回收、对象引用计数）不是线程安全的。
+ 为了避免多线程同时改 Python 对象导致数据错乱，干脆用 GIL 来串行化。
+ 这样实现简单，但副作用就是：**Python 多线程不能真正并行执行 Python 代码**。

3. 那为什么 PyTorch 还能用多线程？

这里有个细节：

**计算部分（如 GPU CUDA 内核、C++ 后端算子）不在 Python 解释器里跑**，所以它们会在进入 C/CUDA 底层时**释放 GIL**。

核心点：GIL 只锁 Python 解释器，不锁 C/CUDA 底层

+ Python 里的操作分两类：
    1. **纯 Python 逻辑**（循环、if、list append 等） → 受 GIL 限制
    2. **调用 C/CUDA 实现的算子**（如 `torch.matmul`、`conv2d`、反向传播里的矩阵运算） → 不受 GIL 限制
+ 当你写下：

```plain
y = torch.matmul(x, w)
```

这行 Python 代码本身会走一小段解释器逻辑（检查参数），但真正的矩阵乘法会**调用底层的 C++/CUDA 函数**。

+ **进入 C++/CUDA 的时候，PyTorch 会释放 GIL**，这样即使多个线程调用算子，Python 解释器不会卡住。
+ 比如 `torch.matmul()`、反向传播里的大部分算子，其实都是调用了 C/CUDA 实现，执行时不会被 GIL 限制。

#### DDP
DDP同时解决了上述提到的两个问题

在前向传播阶段（Forward Pass）

+ **数据分片**：  
每个 GPU（rank）拿到 **本地 batch**（DataLoader 通过 DistributedSampler 事先切分好的），比如全局 batch size = 256，4 张卡 → 每卡 64 个样本。
+ **显存加载**：  
该 rank 先把模型参数复制到本地显存（初始化时 broadcast），再加载对应的输入样本到本地显存。
+ **前向计算**：  
GPU 核心（Tensor Cores / CUDA cores）执行矩阵乘法、卷积等算子，生成本地的 logits / loss。

硬件上这一阶段完全是 **计算为主，显存读写+算子运算**，没有跨 GPU 通信。

反向传播阶段（Backward Pass）

+ **局部梯度计算**：  
每张 GPU 对本地 loss 反向传播，得到 **该 rank 的梯度副本**。  
这些梯度先存在本地 GPU 显存里。
+ **触发通信（梯度 bucket）**：  
DDP 会把参数按 bucket（桶）打包，比如 25MB 一桶。  
每个 bucket 在本地计算完成后，就会触发一次 **all-reduce**，把该梯度和其他 GPU 对应梯度做平均。

梯度同步阶段（AllReduce） ，相对于DP的主要区别

这里涉及到最核心的硬件数据流：

+ **通信拓扑**：
    - 单机多卡 → 通过 **NVLink / PCIe**。
    - 多机多卡 → 通过 **InfiniBand / Ethernet + RDMA**。
+ **环形 AllReduce（Ring AllReduce）数据流**  
假设 3 张 （n个）GPU：
    1. 每张 GPU 把自己梯度分成 3 份（1/n份）。其中每一份
1. scatter reduce阶段：即把每一份的数据按照一个环形通信给下一个GPU
    1. **第 1 轮**：每张卡把其中一份发给下一个 GPU，同时接收上一个 GPU 的一份。并且做加法。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756214696460-a7dcd445-3889-49ac-b776-931c760d30a6.png)

    2. 经过 N-1 轮，所有 GPU 的梯度份额完成加和。即为下图所示，每一个GPU中都有一份数据完成加合![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756214628026-8a3a5ae0-43ed-4ede-9288-11255fbd9baa.png)
2. ALLgather阶段
    1. **多轮广播回去**，得到所有 GPU 都拥有完整、平均后的梯度。![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756214826947-e507d59e-025d-4eba-b890-0d86e5d08176.png) 
    2. ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756214926111-50d8ce6c-5fa9-4fcf-b82f-bfc9ac27a578.png)

硬件上就是 **GPU 显存 → NVLink/PCIe DMA → 另一块 GPU 显存** 的循环流动。

实际上在把一个数据分为n份chunk之前，所有数据已经被分为多个Bucket为了通信时提高进行overlap（注意一下提到的对每个桶进行的操作是一整个allreduce）

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756215572271-85a26c79-f241-4f30-8eb7-1a775ebc9f21.png)

DDP 会把模型参数的梯度按 **桶（bucket）** 打包（默认 `bucket_cap_mb=25`），在反向传播过程中，当**某个桶里最后一个参数的梯度计算完成**时，DDP 就把该桶内的所有梯度打平/合并后发起一次 **异步 AllReduce**。这样可以把通信与正在进行的反向计算并行（overlap），减少等待时间并降低大量小通信的开销。 

什么是 bucket（桶）？

+ **Bucket** = 一组参数对应的梯度张量的集合
+ 顺序参考1：设有层 L4 → L3 → L2 → L1（forward），反向会先算 L4 的梯度。 顺序是先把L4，L3..放在一个桶里，当 L4、L3 的梯度都就绪后就立刻开始 AllReduce，而 CPU/GPU 还在计算 L2、L1 的梯度，合并后大小 ≤ `bucket_cap_mb`（单位 MB）。
+ 顺序参考2：**按 dtype 分桶**：不同 dtype（float32/float16）通常分到不同桶，避免类型转换开销。  
+ 主要目的：
    1. **减少小量多次的 AllReduce 调用**（合并成更少更大的通信）。
    2. **允许在反向传播中早期触发通信，从而与剩下的计算并行**（通信-计算重叠）。

什么时候触发 AllReduce？

+ 在反向传播时，每个参数的梯度在该参数反向路径到达时由 Autograd 产生。
+ DDP 会在每个参数上注册一个 **autograd hook**（`tensor.register_hook`），用于收到“这个参数的梯度已经就绪”的事件。
+ DDP 为每个 bucket 跟踪“已就绪的参数数”。当**桶内最后一个参数的梯度就绪**时，DDP 就：
    - 把桶内这些梯度合并为一个连续/扁平的缓冲区（flatten/coalesce）；
    - 发起一次 **非阻塞的 AllReduce（NCCL）**（异步通信，返回 work handle）；
    - 将该 work handle 存起来，通信继续在后台进行。
+ 反向传播继续向前（对更早层计算梯度），通信与计算并行进行。

> 实际的实现要点和调优建议：实际进行DDP训练为了提升效率可以参考
>
> ## 4) 实现与细节要点
> + **按 dtype 分桶**：不同 dtype（float32/float16）通常分到不同桶，避免类型转换开销。
> + **扁平化（flatten / coalescing）**：在发起 AllReduce 前，DDP 会把桶内的小张量拼成一个大张量，以减少 NCCL 启动次数与小消息开销。
> + **异步通信 + work handles**：AllReduce 是异步发起（`async_op=True`），DDP 保存返回的 work 对象，并确保在进行 `optimizer.step()` 时或在必要点上 `wait()` 完成（保证参数被同步后再更新）。
> + **bucket 的边界与模型参数顺序相关**：DDP 根据 `model.parameters()` 的遍历顺序来划分桶（并考虑 dtype 和 device）。因此**参数在代码中出现的顺序会影响哪些参数被打入同一个桶**，进而影响 overlap 效果。
> + `**find_unused_parameters**`** / **`**static_graph**`：
>     - 如果模型偶尔有未用到的参数（控制流导致部分参数没有梯度），需要 `find_unused_parameters=True`，但这会增加额外开销（DDP 要处理未产生梯度的参数），可能降低一些 overlap 效率。
>     - 如果训练图**固定不变**，可以用 `static_graph=True`（或在新版 PyTorch 中使用相关选项）来让 DDP 假定参数使用固定，这能移除某些检查，从而更高效地利用 buckets。5) 桶大小（`bucket_cap_mb`）与调优建议
>
> 5) 桶大小（`bucket_cap_mb`）与调优建议
>
> + **默认 25 MB** 是折中：既能减少小消息的开销，又能在一般情况下保留一定的重叠。
> + **更小的 bucket（比如 1–5MB）**：
>     - 优点：更细粒度触发，可能在非常深且每层计算很久的模型上带来更早的 overlap。
>     - 缺点：很多小 AllReduce，NCCL 启动/调度开销上升，吞吐可能下降。
> + **更大的 bucket（例如 100MB）**：
>     - 优点：通信次数少、单次吞吐高（更接近带宽峰值）。
>     - 缺点：重叠机会下降（必须等更多参数的梯度都就绪才能触发），并且会占用较大连续缓冲区（内存峰值上升）。
> + **调优原则**：在你的硬件/网络条件与模型上做小范围 sweep：监控 GPU 利用率、通信带宽利用、并使用 nvprof / NCCL 调试工具查看小消息开销。通常：单机多 GPU 用默认 25MB 很合适；跨机或网络慢时考虑增大一点；非常小层计算时间时考虑减小一点，但注意 NCCL 开销。
>
> ## 6) 混合精度（AMP）与 bucket
> + bucketing 是按 **dtype** 分的，所以使用 AMP 时，若出现 FP16 与 FP32 混合，DDP 会分别为不同 dtype 建桶。
> + 如果使用 `torch.cuda.amp` + `GradScaler`，通常参数仍是 FP32，梯度也最终以 FP32 参与同步（取决于你如何实现混合精度），但你需要确认实际 dtype 与 buckets 的匹配情况以避免额外拷贝。
>



+ **多机RDMA 场景**：  
多机时，梯度数据直接从 **GPU 显存经 NIC（RDMA-capable，支持 GPUDirect RDMA）** 发到对端 GPU 显存，中间跳过 CPU 内存。  
流程：显存 → DMA → NIC → InfiniBand 交换机 → NIC → DMA → 显存。

参数更新阶段

+ **优化器（Adam/SGD）执行**：
    - GPU 显存里已经有了平均梯度。
    - 优化器更新参数（也是显存内计算）。
+ **新的参数副本留在各自 GPU 显存里**。  
下个 iteration 直接用。

对于整体的通信量而言，如下图所示

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756218732623-011d5e31-8db2-43f3-8d4d-946fbaa445c7.png)

N为进程数，也是GPU个数，比之前的DP更加平均了，但是通信效果还是不算好，接下来介绍一下DeepSpeed



### DeepSpeed
ZeRO1生命周期

[https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py)

初始化

+ 所有 rank 都 **加载完整的模型参数副本（weights）**（与普通 DDP 一样）。
+ **优化器状态（例：Adam 的 first/second moment）被按 rank 均匀分成 N 份**（N = data-parallel 世界大小），每个 rank 只保留自己的那一份（shard）。这就是 Stage-1 的核心：**优化器状态分片**。

Forward（前向）

+ 每个 rank 用本地完整参数副本执行 forward（无额外网络通信）。激活按常规存放（除非你同时启用 activation checkpointing）。

Backward（反向）

+ 每个 rank 局部计算梯度（每个参数都有本地梯度）。
+ **默认**情况下（Stage-1），梯度会被同步（平均）——DeepSpeed 在 Stage-1 通常使用 `all-reduce` 或在可配置下通过分桶/分阶段的方式（reduce/scatter pipeline）来做梯度合并（有些实现/配置会表现为 all-reduce；Stage-2/3 引入更多 reduce-scatter 约定）。也就是说梯度在逻辑上是“全量可用”的。
    - DeepSpeed 通过 `**contiguous_gradients**`**、**`**reduce_bucket_size**`**、**`**backward_hooks**`**、**`**overlap_comm**` 等设置把梯度打包为桶并在 backward 中逐桶触发通信，从而减少峰值内存并实现通信/计算重叠。[DeepSpeed](https://www.deepspeed.ai/docs/config-json/?utm_source=chatgpt.com)

他是先对梯度进行scatter-reduce/或者直接分发吗？这个梯度的传播方式不是很确定

Optimizer step（关键差异点）

在DDP中，所有梯度同步完，all-gather之后才会进入这个点，但是在Stage1当中是scatter reduce之后就直接对参数进行step,之后再把参数而不是梯度进行all-gather，详细看下面

+ 在传统 DDP：每个 rank 用本地完整 optimizer state（m/v）更新完整参数。
+ 在 ZeRO-1：**每个 rank 只负责更新“自己那份参数/对应的 optimizer-state 分片”**：
    1. 全局梯度已经同步后，DeepSpeed 根据分片映射决定：rank `r` 取出它“拥有”的参数索引集合（例如参数按某种顺序分成 N 段），读取对应的 optimizer state 分片（本地存在），使用这些本地的 m/v 和对应的梯度计算更新 -> 更新该分片的参数值（在本地参数副本上替换对应 slice）。
    2. **更新完成后**，各 rank 需要把自己更新后的参数分片传播给所有其他 rank（使每个 rank 最终仍有完整同步的参数副本）。DeepSpeed 在这一步会做 **all-gather / 分桶 gather**，把所有分片汇集回每个 rank，恢复完整参数副本以供下一步 forward 使用。

简言之：**优化器状态内存被分片（节省显存） → 梯度仍为全量 → 每个 rank 只更新自己负责的参数分片 → 然后把更新的分片 all-gather 回去以恢复完整参数副本。**[arXiv](https://arxiv.org/abs/1910.02054?utm_source=chatgpt.com)[Intel](https://www.intel.com/content/www/us/en/developer/articles/training/memory-efficient-training-on-gaudi-with-deepspeed.html?utm_source=chatgpt.com)

Checkpoint / 保存

+ DeepSpeed 提供配置（例如 `gather_16bit_weights_on_model_save`）来控制保存时是否需要把分片参数收集到单 rank 输出完整权重，或把分片单独写文件。Stage-1 保存时经常需要做 gather 操作以得到整模型。[DeepSpeed](https://www.deepspeed.ai/docs/config-json/?utm_source=chatgpt.com)

对于通信量而言，ZeRO的通信量和DDP一致

+ 因为首先scatter reduce的流程不变，通信量不变。
+ 广播阶段无非把同步fp16的梯度变成了fp16的参数，所以通信量依旧没变化

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756286783400-43e7cfc0-ed00-4919-9255-87a0dae7fe57.png)

在这里可以看看deepspeed官方给出的显存节省图

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756286908876-0dba34a3-1f13-4171-9ace-568372d3306e.png)

可以看出stage1对kpsi/进程数，可以节省很大一部分

这里主要区别如下：

ZeRO-1 的梯度同步

ZeRO-1 和 **原生 DDP** 的做法非常类似：

    - 每个 rank 上保留完整的梯度副本。
    - 在 backward 完成后，梯度需要 all-reduce（通常是 **ring-reduce/scatter-reduce** 实现）。
    - 因为 ZeRO-1 不做梯度分片，所以同步过程和 DDP 基本一致。

📌 梯度同步是 **all-reduce** → 得到每个 rank 上完整梯度 → 优化器更新 → 参数再分片。

ZeRO-2 的梯度同步

+ ZeRO-2 引入了 **梯度分片 (gradient partitioning)**：
    - 每个 rank 只保存一部分梯度（它负责的参数 shard 对应的梯度）。
    - 因为梯度 shard 在同步完成后就可以 **立刻释放**（避免 OOM）。

👉 所以这里不能再用 DDP/ZeRO-1 的 **ring all-reduce**。  
原因：all-reduce 会在每个 rank 上生成完整梯度副本，但 ZeRO-2 不允许这样做。

取而代之，ZeRO-2 使用的是：

+ **reduce-scatter**（把梯度分发到各个负责的 rank）注意与scatter-reduce的区别
+ 然后直接在负责 shard 的 rank 上执行 optimizer step
+ 这样其他 rank 就可以马上释放对应梯度。

总结下

+ **ZeRO-1**：梯度同步用 **all-reduce (ring/scatter-reduce)**，每个 rank 都有完整梯度。
+ **ZeRO-2**：梯度同步改为 **reduce-scatter**，每个 rank 只保留自己负责的梯度 shard（同步完就能释放）。

直接在scatter reduce中收集满一个桶之后直接发送到对应保存梯度的rank之中进行同步

---



## 量化
### 量化精度对比
量化成不同精度的格式对比

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758093754803-fb546b8f-74d2-45dc-be3b-8eae4ad790d7.png)

量化技术：

1. 一些基础的量化技术
+ 零点量化，最大值绝对值量化
+ 最大值绝对值量化就是缩放区间用最大值的绝对值来控制，默认零点是0，这样会浪费区间。

> 如果张量的绝对值值太接近，就会出现浪费区间较大的情况
>

以下是最大值最小值量化：



3.14——缩放系数为（4-0）——，无符号int8的范围是0-255，所以看4/255 = 0.0157,之后再

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758094027529-cdcec29f-f072-4189-95cb-f79165581ab0.png)

于是int存两百，实际代表的是3.1372左右，精度存在丢失

零点量化按照最大值最小值决定量化区间，浪费区间较少，

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758434460760-31c6493d-f94d-441a-9baf-43fa9dc35b3a.png)



量化性能损失程度：[https://arxiv.org/abs/2505.02214](https://arxiv.org/abs/2505.02214)

### 量化哪些数据
### QLoRA
## RL
![](https://cdn.nlark.com/yuque/0/2025/jpeg/43288584/1762741319590-55b2e1fb-900a-4924-b39b-e1c223f75ffd.jpeg)![](https://cdn.nlark.com/yuque/0/2025/jpeg/43288584/1762741338714-b9ba7eac-1433-4ffb-b6e3-265b0d86f091.jpeg)![](https://cdn.nlark.com/yuque/0/2025/jpeg/43288584/1762741360255-dcbdac9a-78a6-429d-b7f4-cb3de9e230be.jpeg)

