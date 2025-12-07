---
title: Search R1
urlname: tw34d1en3p244u6d
date: '2025-10-21 19:28:00'
updated: '2025-10-29 13:19:50'
description: 'https://arxiv.org/pdf/2503.095161. 先前工作RAGagent（ReAct，Toolformer）2. 方法方法结合了三个问题去提出检索引擎如何集成到RL当中，同时保证训练是稳定的如何保证多轮对话中的查+思考循环进行？奖励函数如何设计？提出了主要的几个方法：训练...'
---
[https://arxiv.org/pdf/2503.09516](https://arxiv.org/pdf/2503.09516)

## 先前工作
RAG

agent（ReAct，<font style="color:rgba(0, 0, 0, 0.85);">Toolformer）</font>

## <font style="color:rgba(0, 0, 0, 0.85);">方法</font>
方法结合了三个问题去提出

1. 检索引擎如何集成到RL当中，同时保证训练是稳定的
2. 如何保证多轮对话中的查+思考循环进行？
3. 奖励函数如何设计？

提出了主要的几个方法：

1. 训练时mask掉检索相关的token
2. 需要检索的内容通过生成special token<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0.06);"><search>查询内容</search></font>来触发检索，检索到的内容和生成的部分用<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0.06);"><information>检索内容</information></font>封装，放到当前轨迹中作为后续推理的上下文

## 部署


