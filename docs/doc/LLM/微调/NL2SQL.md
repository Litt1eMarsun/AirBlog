---
title: NL2SQL
urlname: ctg4hbnf1wcw7u0w
date: '2025-09-22 14:22:42'
updated: '2025-10-09 21:30:17'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1758545783658-f18a3e9f-46f8-4b1b-b787-a45c400f0028.png'
description: '数据准备常用nl2sqlSQL数据集wikisql:dev.json对于phase就是数据集的第几个阶段，方便进行切分数据集sql就是具体的sql语句，后面可能会经过转换生成具体的query，其中cond代表的是三元组列表，实际的列，\spider数据库：dev.sjon内部数据格式如下{ "...'
---
## 数据准备
#### 常用nl2sqlSQL数据集
wikisql:

dev.json  
![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758549485997-b235f831-343b-4b04-8eef-5bf3d45827d0.png)

对于phase就是数据集的第几个阶段，方便进行切分数据集

sql就是具体的sql语句，后面可能会经过转换生成具体的query，其中cond代表的是三元组列表，实际的列，\

spider数据库：

dev.sjon

内部数据格式如下

```markdown
{
        "db_id": "concert_singer",
        "query": "SELECT count(*) FROM singer",
        "query_toks": [
            "SELECT",
            "count",
            "(",
            "*",
            ")",
            "FROM",
            "singer"
        ],
        "query_toks_no_value": [
            "select",
            "count",
            "(",
            "*",
            ")",
            "from",
            "singer"
        ],
        "question": "How many singers do we have?",
        "question_toks": [
            "How",
            "many",
            "singers",
            "do",
            "we",
            "have",
            "?"
        ],
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        1
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        3,
                        [
                            0,
                            [
                                0,
                                0,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    }
```

详细说明一下：  
query_toks 和 question_toks很好理解，分词之后的结果

sql这个字段中

from括号里是查询的表的名称

["table_unit", 1]：表示从表1中获取数据。对应.sql文件中就是表singer

conds：字段内部就是Where后面跟着的查询query

query就是最后生成的查询语句，把所有的特殊字符转换回了正式的列名

db_schema.json

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758545783658-f18a3e9f-46f8-4b1b-b787-a45c400f0028.png)

约定好的列名称对应的特殊名称

数据库的具体数值（json格式）

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758545830844-aa0a90e2-0305-4475-b8e3-b25dccffce80.png)

### 数据集选取
现在使用的是modelscope的多轮sql数据集，数据集具体格式如下：  


```python
data = [
    {
        "final": {
            "utterance": "Find the name of the department which has the highest average salary of professors.",
            "query": "SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg(salary) DESC LIMIT 1"
        },
        "database_id": "college_2",
        "interaction": [
            {
                "utterance": "Find out the average salary of professors?",
                "utterance_toks": [
                    "Find",
                    "out",
                    "the",
                    "average",
                    "salary",
                    "of",
                    "professors",
                    "?"
                ],
                "query": "SELECT avg ( salary )  FROM instructor",
                "query_toks_no_value": [
                    "select",
                    "avg",
                    "(",
                    "salary",
                    ")",
                    "from",
                    "instructor"
                ],
                "sql": {
                    "from": {
                        "table_units": [
                            [
                                "table_unit",
                                3
                            ]
                        ],
                        "conds": []
                    },
                    "select": [
                        False,
                        [
                            [
                                5,
                                [
                                    0,
                                    [
                                        0,
                                        14,
                                        False
                                    ],
                                    None
                                ]
                            ]
                        ]
                    ],
                    "where": [],
                    "groupBy": [],
                    "having": [],
                    "orderBy": [],
                    "limit": None,
                    "intersect": None,
                    "union": None,
                    "except": None
                }
            },
            {
                "utterance": "Find the average salary of the professors of each department?",
                "query": "SELECT avg ( salary ) , dept_name FROM instructor GROUP BY dept_name"
            },
            {
                "utterance": "Which department has the highest average salary of professors?",
                "query": "SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )  DESC LIMIT 1"

            },
            {
                "utterance": "Which department has the lowest average salary of professors?",
                "query": "SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )   LIMIT 1"

            },
            {
                "utterance": "In which department Mr. Mird work for?",
                "query": "SELECT dept_name FROM instructor where name  =  'Mird'"
            },
            {
                "utterance": "How much is the salary Mr. Mird earns currently?",
                "query": "SELECT salary FROM instructor where name  =  'Mird'"

            }
        ]
    }
]
```

第一条数据处理完之后就是这种

```markdown
"utterance": "How much is the salary Mr. Mird earns currently?",
    "query": "SELECT salary FROM instructor where name  =  'Mird'"
```

  之后进行脚本处理成sharegpt格式，主要注意一下处理的逻辑字段转换就行，sharegpt需要单独一条一条添加，因为sharegpt数据格式是一个人说话作为一个{}

```python
# 创建文件 concert_sharegpt.py
import json
import argparse

def convert_to_sharegpt_format(input_file, output_file):
    # 读取输入 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 初始化一个空列表来存储转换后的对话
    sharegpt_format = []

    # 遍历每个条目（在这个例子中可能有多个条目）
    for entry in data:
        # 初始化一个空列表来存储当前条目的对话
        conversation = []

        # 将 final 中的对话添加到对话中
        if "final" in entry:
            conversation.append({
                "from": "human",
                "value": entry["final"]["utterance"]
            })
            conversation.append({
                "from": "gpt",
                "value": entry["final"]["query"]
            })
        
        # 遍历每个交互
        for interaction in entry["interaction"]:
            # 将用户的指令添加到对话中
            conversation.append({
                "from": "human",
                "value": interaction["utterance"]
            })
            
            # 将模型的响应添加到对话中
            conversation.append({
                "from": "gpt",
                "value": interaction["query"]
            })
        
        # 将当前对话添加到最终的格式中
        sharegpt_format.append({
            "conversations": conversation
        })

    # 将转换后的数据写入输出 JSON 文件
    with open(output_file, 'w') as output_file_handle:
        json.dump(sharegpt_format, output_file_handle, indent=4)

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Convert CoSQL train data to ShareGPT format.")
    
    # 添加输入文件路径参数
    parser.add_argument("input_file", type=str, help="Path to the input JSON file (e.g., cosql_train.json)")
    
    # 添加输出文件路径参数
    parser.add_argument("output_file", type=str, help="Path to the output JSON file (e.g., sharegpt_cosql_train.json)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用转换函数
    convert_to_sharegpt_format(args.input_file, args.output_file)
```

对于alpaca直接使用字段替换就行 

## 微调
参数解释

```markdown
```json
1. 基本配置
--stage sft：指定训练阶段，这里是监督微调（Supervised Fine-Tuning, SFT）。
--do_train True：指定是否进行训练，True表示进行训练。 false 为不训练
--model_name_or_path /home/util/muyan/Qwen/Qwen2.5-7B-Instruct：指定预训练模型的路径。
--output_dir saves/Qwen2.5-7B-Instruct/lora/train_2024-11-20-16-00-00：指定输出目录，保存训练结果和日志。

2. 数据处理
--preprocessing_num_workers 16：指定预处理数据时使用的线程数。提高数据预处理速度，但可能增加CPU和内存资源的消耗，通常是总核数-1或者-2
--dataset_dir LLaMA-Factory/data：指定数据集目录。
--dataset alpaca_dev,sharegpt_cosql_train：指定使用的数据集，可以是多个数据集，用逗号分隔。
--cutoff_len 2048：指定输入序列的最大长度。 调小模型只能处理较短的上下文信息，可能会丢失一些重要的上下文，影响模型的性能，但是性能会节约
--max_samples 100000：指定最多使用的样本数量。
                    增大：更多的样本可以提供更多的训练信号，有助于模型学习到更丰富的特征和模式，可能提高模型的泛化能力。
                    减小：较少的样本可能导致模型过拟合，无法充分学习到数据的多样性，影响模型的泛化能力。

3. 训练配置
--num_train_epochs 3.0：指定训练的总轮数。
    模型会在整个数据集上进行更多的轮次训练，有机会学习到更多细节，提高模型性能，但也可能增加过拟合的风险。
--per_device_train_batch_size 2：指定每个设备上的训练批次大小。
    更大的批次大小可以提供更稳定的梯度估计，有助于模型收敛，但可能会导致模型过拟合。
--gradient_accumulation_steps 8：指定梯度累积步骤数，用于模拟更大的批次大小。
    提高模型稳定性，但增加内存使用和训练时间。
--learning_rate 5e-05：指定学习率。等价于 0.00005 相当与小数点往前移动了5位数 如果是3e-5 对应的是0.00003，e-6小数点往前移动六位
    权重更新的步长可能会太大，导致模型在损失函数的最小值附近震荡，甚至发散，无法收敛到最优解。或者是过拟合状态
    反之容易欠拟合很难发挥模型最好性能
--lr_scheduler_type cosine：指定学习率调度器类型，这里是余弦退火。帮助模型在接近最优解时进行更细致的调整，提高收敛性和最终的模型性能
    Constant:适合于需要稳定学习率的任务，尤其是在模型已经经过预训练并且只需微调的情况下。
    Cosine Annealing：训练周期较长的情况下，可以有效避免在训练后期的震荡。
--warmup_steps 0：指定学习率预热步数。
--max_grad_norm 1.0：指定梯度裁剪的最大范数。
    调大会加速收敛，但可能会造成梯度爆炸，小会慢，但是相对收敛速度较慢
--logging_steps 5：指定每多少步记录一次日志。
    主要记录日志信息，调大会节约空间，小了会详细但是相对花费时间多一些
--save_steps 100：指定每多少步保存一次模型检查点。
    会做中间节点的保存，调大频率会低，会节约空间，调小保存步骤更加详细。
--packing False：指定是否使用打包技术。
    True:打包技术可以提高训练效率，尤其是在处理短序列数据时，但可能会增加数据预处理的复杂性。
    False:不使用打包技术，训练过程更简单，但可能会因为填充带来的计算浪费而降低效率
--report_to none：指定报告训练进度的方式，none表示不报告。
    类似TensorBoard 这样的外部系统

4. 优化器和混合精度
--fp16 True：指定是否使用混合精度训练。
    如果显示fp32 则不会有当前命令生成
--optim adamw_torch：指定优化器类型，这里是AdamW。 在其他参数设置一栏中进行设置{"optim": "adamw_torch"}
    如 Adam、SGD 会影响模型的收敛速度和最终性能。
--ddp_timeout 180000000：指定分布式训练的超时时间。
    更大的超时时间可以给进程更多的时间来同步，避免因网络延迟或计算差异导致的训练中断，小了可能因为某些网络动荡而造成中断。

5. LoRA 配置
--finetuning_type lora：指定微调类型，这里是LoRA（Low-Rank Adaptation）。
    选择其他类型会对应变化 例如：freeze
--lora_rank 8：指定LoRA的秩。
    LoRA 的秩决定了添加的低秩矩阵的大小。秩越小，添加的参数量越少，计算开销也越小。这个需要因微调场景效果来变更
--lora_alpha 16：指定LoRA的比例因子。
    LoRA 的比例因子（alpha）用于缩放低秩矩阵的贡献。较大的 alpha 值可以使低秩矩阵的贡献更大，从而增强微调的效果。让rank的作用更大，可能会导致过拟合。
--lora_dropout 0：指定LoRA的dropout概率。
    调大：增加模型的泛化能力，减少过拟合，调小：减少模型的正则化，可能提高模型的表达能力
--lora_target all：指定LoRA的目标层，all表示所有层。部分参数微调设置
    可以指定特定层调整
    q_proj：查询投影层
    v_proj：值投影层
    k_proj：键投影层
    fc：全连接层

6. 其他配置
--template qwen：指定使用的模板。
--flash_attn auto：指定是否使用Flash Attention，auto表示自动选择。
--plot_loss True：指定是否绘制损失曲线。
```
```

