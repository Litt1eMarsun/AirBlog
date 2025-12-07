---
title: Tongyi Deepresearch
urlname: ximimpyu1n6wfftv
date: '2025-12-03 14:12:18'
updated: '2025-12-03 14:54:05'
description: 文件结构Agent三种推理模式ReactIterResearch HeavyReSum整个agent架构中分为五层 Mult
---
## 文件结构
### Agent
三种推理模式

1. React
2. **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 247, 246);">IterResearch Heavy</font>**
3. **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 247, 246);">ReSum</font>**

**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 247, 246);">整个agent架构中分为五层</font>**

+  MultiTurnReactAgent   多轮推理循环  
+  call_server   与 VLLM 服务器通信，带重试与回退机制  
+   TOOL_MAP  工具注册表，支持搜索、文件解析、Python 执行等  
+ `custom_call_tool`：工具调用分发，含特殊处理逻辑。  

