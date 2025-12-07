---
title: Llama factroy学习
urlname: lxgxp0trf2syfxaf
date: '2025-09-17 15:14:43'
updated: '2025-09-22 22:35:37'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1758094939030-56328b6e-09bc-4034-a36c-f6633919607f.png'
description: 简介：单机多卡可以支持1. 数据准备LF包含了两种数据格式，一种是shareGPT，一种是AIpaca不同数据集下面有子任务的不同数据集格式，详细阅读下原始链接值得一提的是如果只需要训练对话数据，可以使用shareGPT的一种特殊格式openAI格式，只包含了对话AIpaca数据集详解可以看出...
---
## 简介：
单机多卡可以支持

## 数据准备
1. LF包含了两种数据格式，一种是[shareGPT](https://docs.llamafactory.com.cn/docs/documents/guide/dataProcessing/datasetType#sharegpt)，一种是[AIpaca](https://docs.llamafactory.com.cn/docs/documents/guide/dataProcessing/datasetType#alpaca)

不同数据集下面有子任务的不同数据集格式，详细阅读下原始链接

值得一提的是如果只需要训练对话数据，可以使用shareGPT的一种特殊格式openAI格式，只包含了对话

2. AIpaca数据集详解

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758520888677-5405cd1a-c1a3-47fd-ad2c-316866b64029.png)

可以看出instruction是必须有的，input可以没有。

同时也包括了单轮对话的例子，即没有history

例子比如单轮（alpaca_data_zh_51k.json 中的例子, 数据集在data/dataset_info.json中注册为alpaca_zh）

```markdown
```json


{
  "instruction": "写一个有效的比较语句",
  "input": "篮球和足球",
  "output": "篮球和足球都是受欢迎的运动。"
}
和多轮 （oaast_sft_zh.json 中的例子, 数据集在data/dataset_info.json中注册为oaast_sft_zh）

{
  "instruction": "谢谢",
  "input": "",
  "output": "不用谢! 很高兴我提供的信息能够帮助到你! 如果还有什么其他问题也可以向我提问。",
  "history": [
    [
      "请你给我写一个面试准备计划，我想要去面试微软的程序员岗位",
      "首先，你可以去微软官网寻找招聘信息并申请面试。\n其次，您可以在社交媒体平台寻找微软公司对程序员的面试问题，并做好准备。\n最后，您可以自己对面试过程进行模拟，熟悉话题并减少紧张感。\n我希望你能面试成功。"
    ]
  ]
}
```
```

### 数据转换为标准数据集的方法：、
把data。dump()之后提取字段并设置sharegpt创建对应字段即可

这里的intent指的是输出的字段前面的<font style="color:rgb(68, 68, 68);">缩进空格数（美化输出）</font>

```python
data = {
    "question_id": 34,
    "db_id": "california_schools",
    "question": "What is the free rate for students between the ages of 5 and 17 at the school run by Kacey Gibson?",
    "evidence": "Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`",
    "SQL": "SELECT CAST(T2.`Free Meal Count (Ages 5-17)` AS REAL) / T2.`Enrollment (Ages 5-17)` FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.AdmFName1 = 'Kacey' AND T1.AdmLName1 = 'Gibson'",
    "difficulty": "moderate"
}

# 转换为 ShareGPT 格式
sharegpt_format = [
    {
        "id": str(data["question_id"]),
        "conversations": [
            {"from": "human", "value": data["question"]},
            {"from": "gpt", "value": data["SQL"]}
        ]
    }
]

import json
print(json.dumps(sharegpt_format, indent=4))
```

### data_info详解：
在data_info中可以实现上述过程的简单实现， 直接输入data，指定替换的列名就可以实现数据集快速转换为sharegpt或者alphaca

```markdown
对于 alpaca 格式的数据集，其 dataset_info.json 文件中的列应为：

"dataset_name": {
  "file_name": "dataset_name.json"(自己命名的文件名称及相对路径即可),
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system"(选填),
    "history": "history"(选填)
  }
}

```
```



对于 sharegpt 格式的数据集，dataset_info.json 文件中的列应该包括：

```markdown
"dataset_name": {
    "file_name": "dataset_name.json"(自己命名的文件名称及相对路径即可),
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"(选填),
      "tools": "tools"(选填)
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }



```

同时因为sharegpt天生支持多个tag的描述，所以对于各种数据集sharegpt的格式能够操作

```markdown
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "images": [
      "图像路径（必填）"
    ]
  }
]
```
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "videos": [
      "视频路径（必填）"
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "videos": "videos"
  }
}
```
```

在目前版本中已经省去了一定的对字段的转换，但是如果配置了column也没什么关系![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758551087675-02610f9b-a939-4e13-963e-faa71faf4c57.png)





3. LF中可以配置数据集的地址和数据集名称
    1. dataset_info数据集详解：
    2. dataset目录下dataset_info.json包含了定义的数据集配置

<details class="lake-collapse"><summary id="u7337058c"><span class="ne-text">dataset_info配置哪些内容</span></summary><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol ne-level="2" class="ne-ol"><li id="u438d6b3e" data-lake-index-type="0"><span class="ne-text">加载方式</span></li></ol></ol></ol><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol ne-level="3" class="ne-ol"><li id="u0055b609" data-lake-index-type="0"><span class="ne-text">在线：</span></li><li id="u3b5a16a1" data-lake-index-type="0"><span class="ne-text">    </span><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758094939030-56328b6e-09bc-4034-a36c-f6633919607f.png" width="293" id="GsyQl" class="ne-image"></li><li id="u42476402" data-lake-index-type="0"><span class="ne-text">本地</span></li></ol></ol></ol></ol><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol ne-level="4" class="ne-ol"><li id="u09ff5222" data-lake-index-type="0"><span class="ne-text">指定数据集路径即可</span></li></ol></ol></ol></ol></ol><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol start="4" ne-level="3" class="ne-ol"><li id="ud70b3c61" data-lake-index-type="0"><span class="ne-text">本地数据集脚本生成</span></li></ol></ol></ol></ol><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol start="2" ne-level="2" class="ne-ol"><li id="ub88b508a" data-lake-index-type="0"><span class="ne-text">直接兼容</span></li><li id="u6771727c" data-lake-index-type="0"><span class="ne-text">字段映射（字段内部是tag)</span></li><li id="uae1c1c27" data-lake-index-type="0"><span class="ne-text">tags映射</span></li><li id="ue655ff86" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095153347-6d9cee5b-f180-4b1e-b55b-6482803e28aa.png" width="286" id="yHC8S" class="ne-image"></li></ol></ol></ol></details>
    1. 配置的具体数据流（从原始数据格式文件获得真正送入模型文件的过程）

<details class="lake-collapse"><summary id="u0512d2e1"><span class="ne-text">配置的具体数据流</span></summary><ol class="ne-list-wrap"><ol class="ne-list-wrap"><ol ne-level="2" class="ne-ol"><li id="u711f5b05" data-lake-index-type="0"><span class="ne-text">input原始alpaca数据集</span></li><li id="uee9c9ead" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095363208-21012498-bde4-4a8e-ba42-3d3b0ef2db0a.png" width="867" id="OHusq" class="ne-image"></li><li id="u562481b1" data-lake-index-type="0"><span class="ne-text">通过之前的json配置路径</span></li><li id="u717d87bc" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095395419-48478420-d6eb-4498-9cf3-3dcc9054448b.png" width="706" id="lhDUr" class="ne-image"></li><li id="ude86efaa" data-lake-index-type="0"><span class="ne-text">转化成一般的结构，这里应该包含了tag和字段映射</span></li><li id="u41b8e9a9" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095435318-b78e40f5-9778-48d7-bff5-882abaa9b357.png" width="1216" id="dZR1l" class="ne-image"></li><li id="u95067637" data-lake-index-type="0"><span class="ne-text">转化成模型要求的templet</span></li><li id="u1a87da87" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095521325-c7eb64d0-9185-4a3e-ad32-da66ba569b67.png" width="1312" id="dNTub" class="ne-image"></li><li id="u9a462f74" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095533265-dc4b92dc-b4e7-4a67-9ecb-fd2ed1fe36e2.png" width="683" id="RvBpa" class="ne-image"></li><li id="u8793b22b" data-lake-index-type="0"><span class="ne-text">tokenzier</span></li><li id="u9736797b" data-lake-index-type="0"><span class="ne-text">标签匹配，来设置mask </span></li><li id="u46df2401" data-lake-index-type="0"><span class="ne-text">获得实际数据格式</span></li><li id="u9ed1f4b4" data-lake-index-type="0"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1758095642677-3fb83cd5-0355-43ad-9e9d-119c581ed184.png" width="1025" id="hsYbt" class="ne-image"></li></ol></ol></ol></details>
3. 数据集构造思路：
    1. 数据大小
        1. 领域注入类
            1. 7b的话要1000条起步
    2.  噪声小
        1. 领域注入类
            1. 其他领域的小
        2. 标注数据
        3. 乱码
        4. 重复数据



### Easydataset学习


4. 



### 微调参数学习
1. 学习率
    1. 对于lora

5e-5到4e-5之间



    2. 全参
    3. 1e-5
2. epoch
    1. 3epoch就够了，动态调整，只要没有val_loss上升可以一直进行增加，最好不要超过10
    2. loss不要太低
3. batch_size和梯度累计步数
    1. batch_size * 梯度累计步数决定了每次训练了多少梯度更新一次参数
4. 搭配
    1. batchsize大，则学习率也要大，小同理
    2. 小数据集小模型建议batchsize一开始调小一点
        1. 本次batch_size是1，梯度累计步数是8
5. 截断长度
    1. 决定了输入的最长token大小
    2. 统计数据集文本p99/p95分布cun_length大小，决定截断长度
        1. 可以选择剔除过长，也可以保留
    3. lF中支持了length-cdf的统计，如下![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758100507062-a0696142-11dd-40fa-9b28-419b1a38ab2b.png)

换个参数就可以使用这个脚本了（路径，数据集）![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758100532129-b926f546-5a79-4f9b-b88c-1207ebfda03c.png)



6. 验证机划分
    1. 小规模数据集，（1000，划分0.1-0.2
    2. 大规模数据集（10000，划分0.05-0.1
    3. 复杂任务可以调高比例

### 训练过程
加载以前模型

## 推理加速
unslot

