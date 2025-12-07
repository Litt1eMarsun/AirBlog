---
title: Reinforce learning
urlname: ba1rci9twmbyy9el
date: '2025-08-31 22:45:44'
updated: '2025-11-21 11:48:49'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1756695687556-60a9504f-3e45-4ac7-9959-ecb284783d97.png'
description: '参考：基础概念，从PG到PPO，DPO原理讲解https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.337.search-card.all.click&amp;vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4...'
---
## 参考：
基础概念，从PG到PPO，DPO原理讲解

[https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

代码部分：

minimindDPO

## 基础概念
Environment

Agent

+ **状态（State）**：机器人现在所处的情况，比如“在迷宫的哪个格子里”。
+ **动作（Action）**：机器人能做的选择，比如“往上走”“往左走”。
+ **策略（Policy）**：机器人用来决定动作的方法，比如“看到出口就在朝它走”。
+ 环境：执行动作之后和环境进行交互，从而获得奖励
+ **奖励（Reward）**：环境给的反馈，比如“走到出口 +10 分”，“撞到墙 -5 分”。
+ 轨迹：一连串执行了策略之后的状态和动作叫做轨迹
+ return回报：奖励累计和最大	

类比一下

在PPO中，大模型的状态就是之前已经生成的token，动作集合就是vocab当中的所有token。策略就是大模型网络，可以用来决定生成哪个token。

环境就是奖励模型，把token/logits输入到奖励模型当中获得奖励。轨迹就是先前的token+预测的下一个token的集合

回报就是reward模型给出的总概率

### Onpolicy/offpolicy
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761030647429-656353a2-7cb5-49e2-be23-cdad19b241ba.png)

on policy是在本模型采集数据，并且更新本模型，也就是采集数据和更新是串行的。

但是Off policy不是采集本模型数据，是采集另一个reference model去获得数据 

## PG policy gradient
### 参考：
基础概念，从PG到PPO，DPO原理讲解

[https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

代码部分：

minimindDPO

[https://www.cnblogs.com/xumaomao/p/18805908](https://www.cnblogs.com/xumaomao/p/18805908)

### PG
顾名思义，就是对策略进行求梯度，从而优化均值

tao在当前策略网络的分布下开始采样，根据当前轨迹获得的奖励的期望如下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760430939167-7fecc1ab-e5f9-48d4-a08e-ac50ad4da5b6.png)

其中_**<font style="color:rgb(15, 17, 21);">P_θ</font>**_**<font style="color:rgb(15, 17, 21);">(</font>**_**<font style="color:rgb(15, 17, 21);">τ</font>**_**<font style="color:rgb(15, 17, 21);">)</font>**<font style="color:rgb(15, 17, 21);"> 表示在策略参数为 </font>_<font style="color:rgb(15, 17, 21);">θ</font>_<font style="color:rgb(15, 17, 21);"> 时，</font>**<font style="color:rgb(15, 17, 21);">轨迹 </font>**_**<font style="color:rgb(15, 17, 21);">τ</font>**_**<font style="color:rgb(15, 17, 21);"> 出现的概率</font>**<font style="color:rgb(15, 17, 21);">。R(tao)代表的是轨迹tao获得的回报</font>

因为可以改变的之后参数theta，所以如果要最大化期望，应该朝着Ptheta梯度增加的方向走，求下梯度，并进行如下图转化

#### ReinForce 算法
具体的推导不解释，

RAINFORCE算法的核心是用蒙特卡洛代替策略函数求和，也就是下述蓝色到下一步的过程，所有轨迹出现的概率求和的无偏估计是对所有其出现情况求平均（大数定律）

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760431356734-6a897e61-8016-46ea-ad2f-96f4240f0b91.png)

最终得出的形式就是

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760431434833-440d8ec6-4af6-4863-8e9f-6cde6a1fb5eb.png)

上图所示，可以把tratajey拆分成具体的action和state的组合形式，也就是：

根据以往的状态预测下一个action的概率分布的连乘

因为之前引入了log，所以连乘变成了加法之和，最终表达式就是

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760431617627-0ada0f07-db16-44e2-a3af-3cc3e8d8ce84.png)

这个公式的具体意义就是，如果tratjy的奖励大于0，则增大对应的整条轨迹出现的概率，如果小于0，则减小整条轨迹出现的概率

取个负数就是loss

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760431800199-8f9a915c-8f2d-49fd-864f-29af4a5b2c38.png)

但是感觉这种形式不是很好记，最好应该是下面去掉梯度符号

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760856938119-dd30b8b5-fab6-43f9-8a20-4d1b5db2fb24.png)

解释下公式：

当前动作做了之后，之后所有的动作（轨迹）出现的概率对数和与奖励函数乘积，从而获得当前采取行动的评分。

对整条轨迹都这样，从而获得整条轨迹的评分

最后对所有轨迹求平均，就是这一组数据的loss，以这个loss去优化策略网络Ptheta

#### ReinForce优化方法：
RAINFORCE 这样做也有缺点

1. 可能是某个action导致的奖励降低，直接降低整条tratjy出现的概率不是很好
2. 某个错误的action应该是对t越近，其所造成的影响越大，t越远，其造成的影响越小

这样可以调整下奖励函数

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760432155830-59b78813-00e8-4d3f-a9ce-f9ae765d4d8a.png)

增加了衰减因子，0<γ<1，离当前的决策越远，其应该越不重要，奖励函数的权重也会越低，这里把这个作为奖励函数

同时还有缺点就是：方差太大，这也是后面一些方法的核心的优化点。

先从感性上理解一下为什么要减去baseline:

在好的局势下，reward function给出reward的概率会大于坏的局势下，

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760857468836-4caa8e68-3975-4ae2-b2aa-14707af3f0cc.png)

但是希望的是奖励函数能够奖励当前做的决策，而跟局势没有关系，所以这里给所有的reward_function减去了一个base，这样奖励函数的评价效果会更好，公式如下  
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760881716427-49fc8b16-a257-4376-84a6-6a62a8cacd7c.png)

再从方差的角度理解一下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760887074041-9959e331-cd19-495c-8743-38033049a0c6.png)

<font style="color:rgb(15, 17, 21);">在我们的情况中：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760887088390-89876c8f-a21f-4090-a98d-c85256a5f0fa.png)

没有详细推导，直观理解下，两个相关的变量相减，只要c控制好了，方差必然会减小的

但是这里的方差仍然比较大，因为根据定义，每个轨迹的奖励如下式求和：

<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">实际采样回报 = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...</font>

<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">因为每条轨迹从t开始的回报是 rt奖励函数*从当前步数开始，后面所有步数随机采样获得的tratajy，引入了大量的随机采样的随机变量rt...，所以方差很大</font>

<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">那么有没有办法是让采样回报的表示用不着那么多随机变量呢？介绍一下下一步的A-C方法</font>

#### <font style="color:rgb(15, 17, 21);">Actor-Critic 方法</font>
##### 简介及定义
实际上上述-baseline的方法可以重新表述如下：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760948983214-adaab109-d55f-4f76-a9a9-95812a14a0ac.png)

其中

![image](https://cdn.nlark.com/yuque/__latex/ff83bb5178226fd8754417abdd0f5005.svg)：优势函数（Advantage Function）

    - 衡量动作比平均水平好多少 ![image](https://cdn.nlark.com/yuque/__latex/e5c7d47982b8736d999ede2c24130cd9.svg)
    - 如果 ![image](https://cdn.nlark.com/yuque/__latex/117243a60f6282d1b6e00c9cce320bd5.svg)，说明动作好；如果 < 0，说明动作不好。

![image](https://cdn.nlark.com/yuque/__latex/5d330876260b9503fdc6820fce34bf5a.svg) —— 状态值函数

+ **意思**：在状态 s_t 下，智能体**平均能获得的未来总奖励**
+ 公式上：	

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756696848572-8276d836-1a25-491b-8b56-1dc42db29769.png)

+ **直观理解**：就像告诉智能体：“你现在这个位置，未来大概能拿到多少奖励”。
+ 只依赖状态，不依赖具体动作。
+  用 **值函数网络** 直接预测的 , 训练时，会让 $V_\phi(s)$ 去拟合真实的折扣回报 $R_t$：

![image](https://cdn.nlark.com/yuque/__latex/3315e5e52c9463051080389e2323b573.svg)——动作价值函数

+ 在状态 $s_t$ 下，采取动作 $a_t$ 后，智能体**平均能获得的未来总奖励**。
+ 公式上，价值函数的定义如下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756697330771-0f238761-6749-4551-9de5-16060ef113a1.png)

+ 可以由状态值函数表示：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760949188473-e5a5b680-2ff1-458d-979d-7a3381777f15.png)

理解：采取行动a之后，预期的回报期望=这一步获得的即时收益+采取这一步之后，下一个状态可获得总奖励的期望

##### 优势函数的拆解与表达
优势函数A可以表达如下：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760952319746-16821148-2d2c-4d9c-b82b-2ca574bc9b09.png)

动作价值函数可以表达成一步采样，也可以表达成多步采样，因此优势函数可以表达如下，以此类推：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760951886743-4010c1a7-d6f4-46a2-bf54-c2fb07613d61.png)

采样的步数越多，方差越大，偏差越小。

这里的方差指的就是之前提到的，c-a算法的优势，因为我们可以控制采样的步数，让方差变得很小，其余的采样步骤全用Vtheta去替代。但是这样对Vtheta的要求就越高，偏差相比于完全采样来看就会有偏差，这就是为什么说步骤越多方差越大，偏差越小，反之亦然。

这就是C—A相比于ReinForce优化的优点，给了一个trade-off的方法

这里需要再详细总结一下上述提到的值函数和回报函数的递推式子

    - 对于回报函数，更希望贴近真实值，也就是G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...  但是为了减小方差，用Vtheta进行代替，也就定义成了上述形式，其实也应该是≈
    - 对于值函数，更希望他能够代替的是γ（一系列真实随机变量的乘积），所以定义成了上述的![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761045543166-43ee35a7-89f0-4f9c-8d60-d9d3f678af1f.png)这个形式（st+2..)就是后面一系列成绩的代替式子
    - 因此回报函数可以写成一系列的Vtheta的递归式，本质上是用值函数对真实采样的一系列预估（可以理解成泰勒展开可以将后续任意个高次项用神经网络预测）

因为这种全部拆开的式子表达太麻烦，所以提出

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760952444648-b41a2760-cafc-43fd-8fa4-09935e8b9bce.png)

解释下，δt表示t步采取a取得的优势，因此A当前步骤的优势函数可以重新表述为如下：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760952760637-49cb5aac-4a8a-4d2a-ad01-e840d47e7ec2.png)

##### 优势函数实际选取
之前说到了优势函数的采样步骤是可以自由选取的，常见的优势函数的选取方式如下：

1. GAE优势函数

GAE优势函数的想法是全部采样都要，若lambda很小~0，则说明GAE非常短时，只希望最近的rt+预估的作为优势参考，如果lambda~1，则说明希望所有的r都要用上，更接近蒙特卡洛采样。因此他能平衡方差与偏差，更准确的估计在t时采取了a之后，有多大的优势。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760952987974-5026f029-3c05-437e-ba91-819b300a6b2a.png)

因此整体的损失如下：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760976326876-11dc9f78-8558-41f0-8a86-1224fbaac6e0.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760976345795-b3d92b0e-4a4e-49a3-bab1-2c41af1c11b5.png)

上面的label就是状态价值函数的标注，V(s_t) = ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760977444861-772753f4-ad66-4dc7-93fe-251c5e325546.png)，也就是从当前状态到后面所有状态的奖励函数加权求和，用这个标注信息去训练状态价值函数

## PPO
[https://www.bilibili.com/video/BV1iz421h7gb?spm_id_from=333.788.videopod.sections&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1iz421h7gb?spm_id_from=333.788.videopod.sections&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

### 基础
重要性采样，也就是把f(x)的期望进行拆分，转换分布，x~q(x)，同时用大数定律做一下估计，就是最终的期望

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761030962801-5047d5aa-6f74-4ed6-aa3b-b844f3ea8531.png)

函数f(x)，其中x~p(x)，但只能通过分布q(x)来采样，这时候可以通过p(x)/q(x)来修正期望值，就是如图最后一步所示，带入到之前的A-C算法当中，会修正成如下所示：

Ptheta就是需要训练的参数网络，Ptheta'是参考模型

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761033007733-16bad1bf-1c40-462f-a19e-89661002e71b.png)

之后化简，推导ppo的loss，核心就是对lgPtheta求梯度这块的化简，最后就是只对Ptheta求梯度，最后转化成loss的形式

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761033082394-dca16706-9119-4140-ac5f-522b7c2f5493.png)

优势函数等是通过参考模型求出来的，注意参考模型和实际的训练模型差距不能过大，这个直观上很好理解，同时提一嘴，重要性采样的要求即P(x)被Q(x)分布包含，也说明了 参考模型和训练模型的分布差距不能过大。

为了衡量策略不偏离过大，有两种方法可以衡量，第一个就是KL散度，在ppo的loss上增加个kl散度的正则项。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761036033417-c5f23cdd-fb5b-4024-b063-5752998f8512.png)

第二种就是增加裁剪函数，防止新策略偏离旧策略太多

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761036022678-1f332fd9-7842-48d3-87a5-5f887fee5f8f.png)

### PPO过程
#### 整体架构
PPO一共有四个模型，如果四个模型全部加载，则需要占用大量显存，于是可以使用Lora代替部分

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761066400480-3648c522-40ce-4e21-b8bf-591dee28cb01.png)

奖励模型为什么还要带hiddensize是因为后面还要过个分类头？状态价值模型同理,输入head 的时候是b*h*1，输出就是b*1

数据准备如下：

1. 用户偏好数据

#### Reward模型
##### Reward loss
损失是chosen的得分-reject的得分之后过了个logsigmoid，使得分在正的时候loss很小，负的时候loss很大

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761065415915-d37c6abc-e1eb-4660-aa0a-f349efb22e62.png)

##### reward 训练过程
1. 模型搭建

换了个模型头，变成class分类任务，分类头只有一个类就是得分（可以看看和casualhead有什么区别，我感觉casualhead除了更大之外没有啥和分类头的区别了。还有就是不需要mask？直接用之前的所有token自回归生成一个label?

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761066253304-73c71d5b-09aa-451a-88ed-3f646f770a01.png)

2. 数据处理

 ![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761066201437-b98672ee-d360-4052-baa8-c9d6c128a25e.png)



问题+回答进行拼接，并进行mask，猜测这里的mask是mask全部序列（input+reject,input+chosen)用的，毕竟只要输出最后一个？

reward推理的时候，获得每个token的reward rt是KL散度的*系数*-1,所以每个token（state)的reward作用很小，没啥意义，最后一个token的reward最有意义。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761110365305-f6476b04-896b-4f6c-8326-078d340360d8.png)

有了reward之后，可以根据V(theta) 直接获得GAE优势函数，也就是如下图所示

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761110532057-e7783d62-0cdf-4251-8c75-594796c56ccd.png)

上述GAE优势函数可以拆解成下面的步骤，方便写代码的步骤，也就是当前的优势+下一步GAE优势的衰减之后的结果。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761110596885-52f2a322-110c-4ecd-bddf-2d2d028a073b.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761110608085-d7d57727-d48a-4159-87b0-58003a056c20.png)

解释下代码，从t=T-1开始算，每个delta（当前step产生的优势） = 当前step的reward + （下一步状态的平均价值-当前状态的平均价值）*衰减系数

gae（优势函数） 项从最后t开始，应该是当前步骤带来的优势+之后步骤带来的优势之和（最后一步t+1不存在导致gae优势为0，所以直接是delta）

最后能够获得每个t采取行动a对应的GAE优势（后续所有的优势）函数作为advantages



#### 价值网络的优化
critic网络也要在迭代中不断获得优化，所以loss中应该增加V网络的loss

价值网络训练的label比较奇怪，并不是有个直接的标注，而是通过bootstraping（自举）来设置label。

具体来说，沿用之前的思路，label有三种可以标注的情况

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761113493916-80386b4f-b8fc-41c0-980d-d8e452448704.png)

之前提到的ppo的标注方法其实是蒙特卡洛法，但是实际上这样方差会比较大，TD法因为只有一步及时的reward，和自举得出的V，所以导致偏差会比较大，而GAE因为加权了多步的优势函数，所以效果会比较好。

这样在训练时不断标注，同时不断训练，会导致V的偏差越来越小，最终训练出还不错的状态价值网络。

有了label，按照下述训练损失：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761114004368-fa15735b-3c91-4d48-982a-09ea5c0f21f8.png)

在实际训练时，critic网络（状态价值网络）损失如下：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1761114035270-5173ced0-9124-41a0-a001-471cda35f8ec.png)

adavantage就是之前算GAE用到的每个step的产生的优势+当前step的状态，

之后在每个step得出V和r之后，就可以获得训练参数网络用的PPO了。









### 策略目标函数
### PPO 的核心思想
PPO 的核心是**限制策略更新幅度**，避免“策略更新过猛”，同时又能稳步提高。

公式上，PPO 使用 **剪切（clip）策略目标函数**，定义为：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756695687556-60a9504f-3e45-4ac7-9959-ecb284783d97.png)

我们来拆解：

### 每个符号含义
1. ** **![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)策略网络的参数。 
2. ![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)：策略比率（新策略 / 旧策略）

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756695977040-fc05c51b-b521-4bff-b957-b2029138af60.png)

    - 直观理解：如果 r_t = 1，说明新策略和旧策略一样。
    - 如果 r_t > 1，说明新策略在这个动作上“提高了概率”。
    - 旧策略rollout的概率结果和新策略概率相除即可
5. 
6. 
6. **clip()**：限制 r_t 在 ![image](https://cdn.nlark.com/yuque/__latex/18019f476bfde1722b8e0d074f8cd5d1.svg) 之间
    - 这样就实现了“安全护栏”，防止策略跳得太大。

### Rollout
先理清一下单个step,trajectory/episode，rollout的关系

####  层次 1：单步交互 (step)
+ 智能体在环境里执行 **一步动作**。
+ 数据：$(s_t, a_t, r_t, s_{t+1})$
+ 类似「机器人迈一步」。

#### 层次 2：一次尝试 (trajectory / episode)
+ 连续的多步，直到 **任务结束** 或达到设定步数。
+ 数据：${(s_1,a_1,r_1), (s_2,a_2,r_2), \dots}$
+ 就是我最早给你举的「走三步」的例子。
+ 在代码里通常称为 **episode**。

#### 层次 3：Rollout（采样批次）
+ 用当前旧策略 $\pi_{\theta_\text{old}}$，在环境里收集一批尝试（可能是多个 episode，或者固定步数，比如 2048 步）。
+ 存储数据：
    - 状态 $s_t$
    - 动作 $a_t$
    - 奖励 $r_t$
    - 旧策略概率 logp_old
    - 值函数预测 $V(s_t)$
+ Rollout 就是 **大规模的数据收集阶段**，为后续训练准备数据。

####  层次 4：一次更新循环 (update epoch)
1. Rollout：先收集一批数据（states, actions, rewards, logp_olds, values）。
2. 计算优势函数 $\hat{A}_t$。
3. 计算策略比率 $r_t(\theta)$。
4. 用 PPO 的损失函数更新策略参数 $\theta$。
5. 更新值函数参数 $\phi$。
6. 把新策略设为旧策略，进入下一轮。

这就是 PPO 的一个完整训练循环。

#### 具体例子展示计算过程
假设数据

假设智能体在某次尝试中收集到以下一条数据：

| t | 状态 $s_t$ | 动作 $a_t$ | 奖励 $r_t$ |
| --- | --- | --- | --- |
| 1 | s1 | 左 | 1 |
| 2 | s2 | 上 | 2 |
| 3 | s3 | 右 | 0 |


另外我们已经有旧策略 $\pi_{\theta_\text{old}}$ 和当前值函数 $V(s)$：

| 状态 | $V(s)$ |
| --- | --- |
| s1 | 1.5 |
| s2 | 2.0 |
| s3 | 0.5 |


步骤 1：计算优势函数 $\hat{A}_t$

先计算动作值函数 $Q(s_t,a_t)$，在实际应用中通常用**折扣奖励累加**计算：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698650343-fa0d349b-8000-446e-a58e-d7cfae81c659.png)

假设折扣因子 $\gamma = 0.9$：

+ t=1: $Q(s_1,a_1) = 1 + 0.9_2 + 0.9^2_0 = 1 + 1.8 + 0 = 2.8$
+ t=2: $Q(s_2,a_2) = 2 + 0.9*0 = 2$
+ t=3: $Q(s_3,a_3) = 0$

然后优势函数：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698667854-6b58d489-5dca-4ab7-9252-dabf3201e60b.png)

+ t=1: $\hat{A}_1 = 2.8 - 1.5 = 1.3$
+ t=2: $\hat{A}_2 = 2 - 2 = 0$
+ t=3: $\hat{A}_3 = 0 - 0.5 = -0.5$

直观理解：

+ 动作 1 比平均好 → 想提高概率
+ 动作 2 平均 → 不改
+ 动作 3 差 → 想降低概率

步骤 2：计算策略比率 $r_t(\theta)$

假设旧策略概率 $\pi_{\theta_\text{old}}(a_t|s_t)$ 和新策略 $\pi_\theta(a_t|s_t)$：

| t | $\pi_{\theta_\text{old}}(a_t|s_t)$ | $\pi_\theta(a_t|s_t)$ |  
|---|----------------------------------|----------------------|  
| 1 | 0.2 | 0.25 |  
| 2 | 0.5 | 0.55 |  
| 3 | 0.3 | 0.2 |

策略比率：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698797511-ef3caeec-6a92-4ea9-90d1-541545dc3e24.png)

+ t=1: $r_1 = 0.25/0.2 = 1.25$
+ t=2: $r_2 = 0.55/0.5 = 1.1$
+ t=3: $r_3 = 0.2/0.3 \approx 0.667$

步骤 3：计算 PPO 损失 $L^{CLIP}$

假设 $\epsilon = 0.2$，clip 范围 $[0.8,1.2]$：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698823121-d9d3ef44-67e9-4d15-a2ba-d5e7f44bf035.png)

+ t=1: clip(1.25) → 1.2
+ t=2: clip(1.1) → 1.1
+ t=3: clip(0.667) → 0.8

然后 PPO 损失：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698833838-33f780c8-1e74-464c-ae14-991a21f53f90.png)

+ t=1: min(1.25_1.3, 1.2_1.3) = min(1.625, 1.56) = 1.56
+ t=2: min(1.1_0, 1.1_0) = 0
+ t=3: min(0.667*(-0.5), 0.8*(-0.5)) = min(-0.3335, -0.4) = -0.4

 最终损失取负梯度上升（因为我们想**最大化奖励**）：  

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1756698859620-38bf941e-c1aa-4b92-9ab7-4a0c4837d00c.png)



## DPO
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760168156473-4e500b6d-244c-4bcf-bd10-93366212a715.png)

通过偏好数据，利用最大似然估计训练奖励模型

<details class="lake-collapse"><summary id="u8e95b547"><span class="ne-text">最大似然估计</span></summary><p id="ue1977dfe" class="ne-p"><span class="ne-text">最大似然估计是通过实际数据估计参数的参数估计手段</span></p></details>
但是在DPO当中，不需要训练奖励模型，直接用DPO中的loss+原参数模型就可以训练出模型。

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760168653042-c8008a96-9513-4aad-8475-27dbae288620.png)

具体loss的推导手段如下

首先KL散度的定义如下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760168687870-3cf10656-bc0d-4fb7-a9ed-6e82d8d75cf5.png)

可以通过这个文章了解一下KL散度

[https://zhuanlan.zhihu.com/p/37452654](https://zhuanlan.zhihu.com/p/37452654)

[https://zhuanlan.zhihu.com/p/1950257135775642370](https://zhuanlan.zhihu.com/p/1950257135775642370)

首先理解一下每个式子的含义。P(x)表示原始分布，Q(x)表示近似分布

这里可以简要了解一下为什么要用log来定义。

首先定义一下信息熵：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760189951472-3f4638dd-bb4e-4f40-8908-ac2eb556dbcf.png)

越靠近0越无穷大，越靠近1越为0。符合直觉，因为越少概率发生的事情信息量越大。

这时如果有两个分布，p(x)和Q(x)，我想衡量这两个模型之间的相似程度，在信息熵的背景下，肯定是希望这两个分布之间信息熵的差异是越少越好的，信息熵趋于0==已知了一个模型就知道了另一个模型了

这时候就会出现一个问题，用Ip-Iq还是反过来呢？我们先搁置，继续理解KL散度，假设是Q-P

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760190307396-8be22934-9776-4e31-b185-fa9c3baafb0e.png)

这样可以得出

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760190851874-b3ee120a-fc92-4220-bde6-06272b2f56ee.png)

这个就是KL散度其中的一项。

对于两个分布而言，分布中某个随机变量发生的概率越高，则该随机变量更有价值，所以

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760190894887-971a4207-dd1e-4ae8-b61e-860a8ed638ea.png)

这样就完成了整个KL散度的定义。及发生的概率*信息熵的差值，获得两个分布之间的差异程度

至于为什么两个分布之间要Q(X)-P(X)呢？这个文章说的很好。

使用KL散度作为损失的时候，肯定希望KL散度越少越好，这样预测的分布Q更贴近于原始分布P

对比两种设计

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760191046382-330da183-6b9a-4123-a776-76da896ecd6d.png)

第一种，如果Q(x)趋于0，但是P（x)不趋于0，则整个P/Q会趋于无穷大，整个系统的损失也会趋于无穷大

P是不是0无所谓，反正KL散度始终大于0（通过不等式推导）

这句驱动Q(x)绝对会覆盖P(x)

这样其实更有利于多样性，不会绝对正确，所以大模型选择的也是这个

第二种反之也可以推导





之后介绍一下TellyBerry模型

简而言之可以通过AB获胜概率的不同推导出A和B的实力

首先我们要获得A和B pairwise 的对比数据，并获得pairwise好坏的分别得分

定义出现好的I的概率大于差的J的概率如下

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760193128106-7694a656-f5f1-4f7c-9ad7-e25503e38d8b.png)

其中theta就是得分，e^theta表示实力大小，因为实力大小必须大于0，所以给得分套了个e

则所有pairwise数据出现的概率和的似然函数为

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760193293328-8eb97e3b-9139-48c8-8114-73c8c9cfb833.png)

现在获得对数似然（因为更方便求解）

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760193344839-af3b4f1f-6f3e-4d5f-aa2b-1aee066a3a4c.png)

我们希望所有数据的似然函数最小，因此

希望似然函数取－之后最小，也就是似然函数取最大值，因此loss可以写成如下形式

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760196768594-f94f5d59-1421-4511-bb6b-0cf281455028.png)

因为P可以写成

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760193189705-fab91781-3ba6-48d0-b0a7-688ea3a051c1.png)

其中这个是sigmoid函数

所以可以推导loss为



之后定义：

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760198294128-dac49c67-083e-4d91-9506-db77ac24b105.png)

接下来就是将整个式子尽量的大就行了，也就是奖励尽可能的多的同时，不偏离原有分布太多

具体的推导过程明天再看![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760196795404-089218db-3b0c-40d0-96e2-7984ac981bb6.png)

[https://www.bilibili.com/video/BV1GF4m1L7Nt/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d](https://www.bilibili.com/video/BV1GF4m1L7Nt/?spm_id_from=333.337.search-card.all.click&vd_source=5d6b3d0e0ed93a1eb9dcb61a5d4a906d)

直接跳跳跳到最后一步

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1760198058800-7c2a88be-a9bc-4d9b-87b6-fbb41d06092a.png)

对于这个公式的思考？

1. 中间缺失很多步骤
2. 最后的公式缺少思考

具体的实现可以看一下DPO的Minimind代码，把每个参数和每部分都弄懂了

同时可以对比下DPO微调Lora和全量微调的区别

[https://zhuanlan.zhihu.com/p/8362625598](https://zhuanlan.zhihu.com/p/8362625598)













## GRPO


