---
title: Agent工程搭建
urlname: kg67eumiz00ecdma
date: '2025-09-06 12:04:02'
updated: '2025-12-02 18:04:05'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1757174898149-3b9708ab-b295-410e-a57c-8e268c62ea34.png'
description: 1. agent搭建流程1.1. 简单的zero-shot agent1.1.1. 工具定义1.1.1.1. 装饰器装饰器是 LangChain 框架中的一个重要组件，用于将普通的 Python 函数转换为可被 AI 智能体调用的工具。主要作用函数注册：将函数注册为 LangChain 工具，...
---
## agent搭建流程
### 简单的zero-shot agent
#### 工具定义
##### 装饰器
装饰器是 LangChain 框架中的一个重要组件，用于将普通的 Python 函数转换为可被 AI 智能体调用的工具。

1. 主要作用
+ 函数注册：将函数注册为 LangChain 工具，使其能被 AI 模型识别和调用
+ 自动生成工具描述：

从函数的 docstring 生成工具描述

从函数签名提取参数信息

自动推断参数类型

参数验证：基于函数的类型注解进行输入验证

手写一个装饰器代码如下：

```python
def log_level(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{level}] 调用函数: {func.__name__}")
            result = func(*args, **kwargs)
            print(f"[{level}] 函数执行完毕")
            return result
        return wrapper
    return decorator

# 使用带参数的装饰器
@log_level("INFO")
def calculate(x, y):
    return x + y

@log_level("DEBUG")
def process_data(data):
    return len(data)

result = calculate(5, 7)
print(f"计算结果: {result}")
length = process_data([1, 2, 3, 4, 5])

```

其实就是在已有的函数外部包了一个函数，函数会调用其装饰的函数，并传参进去。

这里说一下*在传参的作用（收集参数，解包，传参用）

<details class="lake-collapse"><summary id="uf844337c"><span class="ne-text">*和**</span></summary><pre data-language="python" id="yOnWT" class="ne-codeblock language-python"><code>def my_function(*args):
    print(args)  # args 是一个元组

my_function(1, 2, 3, 4)  # 输出: (1, 2, 3, 4)</code></pre><p id="uadefc241" class="ne-p"><span class="ne-text">可以看出作为形参就是合并用，输入是散的但是在函数调用的时候，参数变成了一个元组</span></p><pre data-language="python" id="nxNuh" class="ne-codeblock language-python"><code>def my_function(**kwargs):
    print(kwargs)  # kwargs 是一个字典

my_function(a=1, b=2, c=3)  # 输出: {'a': 1, 'b': 2, 'c': 3}</code></pre><p id="u3fb4278b" class="ne-p"><span class="ne-text">也一样，收集参数用，但是这里收集的是关键字参数，可以看出，输入的是关键字参数，在函数中就变成了一个字典</span></p><p id="u8c5f5f12" class="ne-p"><span class="ne-text">如果函数调用</span></p><pre data-language="python" id="KMRii" class="ne-codeblock language-python"><code>def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
result = add(*numbers)  # 等同于 add(1, 2, 3)
print(result)  # 输出: 6</code></pre><p id="ua30d42e8" class="ne-p"><span class="ne-text">可以看出，对输入的参数直接进行了拆开</span></p><p id="ucf0d8af2" class="ne-p"><span class="ne-text">同理</span></p><pre data-language="python" id="YfROH" class="ne-codeblock language-python"><code>def greet(name, age):
    return f&quot;Hello {name}, you are {age} years old&quot;

person = {&quot;name&quot;: &quot;Alice&quot;, &quot;age&quot;: 25}
message = greet(**person)  # 等同于 greet(name=&quot;Alice&quot;, age=25)
print(message)  # 输出: Hello Alice, you are 25 years old</code></pre></details>
#### 定义状态
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

```

 这段代码是在定义 LangGraph 中的代理状态结构，用人话说就是定义一个在agent各个node中流动的数据结构

下面我来逐个解释每个元素的含义和作用：  

```python
messages: Annotated[list[AnyMessage], add_messages]
```

这里的List[message]就是在agent中流动的消息及其所有历史，历史的添加方式就是通过add_message这个归约器函数去添加的

##### 🧠 什么是 `Annotated`？
+ `Annotated` 是 Python 的一种类型增强机制，允许你给类型附加元信息，相当于把归约器函数和数据封装在一起，让数据自带一个增长行为。
+ 这里annotated后面的reducer（归约器函数）详细解释如下

<details class="lake-collapse"><summary id="u6f5a049c"><span class="ne-text">reducer函数</span></summary><h3 id="NorRW"><span class="ne-text">🔧</span><span class="ne-text"> 函数定义回顾</span></h3><p id="u4eeffce2" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="EaYSG" class="ne-codeblock language-plain"><code>def reducer(a: list, b: int | None) -&gt; list:
    if b is not None:
        return a + [b]
    return a</code></pre><h3 id="FmXdR"><span class="ne-text">🧠</span><span class="ne-text"> 函数逻辑解析</span></h3><p id="u6237911b" class="ne-p"><span class="ne-text">这个函数是一个典型的 </span><strong><span class="ne-text">状态聚合器</span></strong><span class="ne-text">，用于将新值 </span><code class="ne-code"><span class="ne-text">b</span></code><span class="ne-text"> 合并进已有的状态列表 </span><code class="ne-code"><span class="ne-text">a</span></code><span class="ne-text"> 中：</span></p><ul class="ne-ul"><li id="uf2dc5cd9" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">a</span></code><span class="ne-text">：当前状态值，是一个列表（例如 </span><code class="ne-code"><span class="ne-text">[0.5]</span></code><span class="ne-text">）</span></li><li id="u0c387b90" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">b</span></code><span class="ne-text">：某个节点输出的新值，可能是 </span><code class="ne-code"><span class="ne-text">None</span></code><span class="ne-text"> 或一个数字（例如 </span><code class="ne-code"><span class="ne-text">0.75</span></code><span class="ne-text">）</span></li></ul><h4 id="xLSzA"><span class="ne-text">行为：</span></h4><ul class="ne-ul"><li id="uc09f0255" data-lake-index-type="0"><span class="ne-text">如果 </span><code class="ne-code"><span class="ne-text">b</span></code><span class="ne-text"> 是有效值（非 </span><code class="ne-code"><span class="ne-text">None</span></code><span class="ne-text">），就把它追加到列表 </span><code class="ne-code"><span class="ne-text">a</span></code><span class="ne-text"> 中。</span></li><li id="u0ec32931" data-lake-index-type="0"><span class="ne-text">如果 </span><code class="ne-code"><span class="ne-text">b</span></code><span class="ne-text"> 是 </span><code class="ne-code"><span class="ne-text">None</span></code><span class="ne-text">，就保持原样。</span></li></ul></details>
reducer函数是用来更新状态中的字段用的。实际输入一个question之后经过assisitant包装的runnable变量（这里是绑定了工具，并且设置了上下文的llm)_之后肯定会生成一个llm的回应之类的。这个时候把新生成的回应（或者是工具调用结果）等等自动归约器增加到state当中

补充一下graph调用state的机制：

<details class="lake-collapse"><summary id="uf82e97be"><span class="ne-text">graph调用机制</span></summary><h3 id="ZjTqc" data-lake-index-type="2"><span class="ne-text" style="color: #000000; background-color: #FFFFFF">schema</span></h3><p id="u5b92d37c" class="ne-p"><span class="ne-text">输入的state是作为</span></p><ul class="ne-ul"><li id="u6dfd8cc6" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">state_schema：全局状态结构（驱动所有节点读/写）这个参数输入的。</span></li><li id="uc547d7dd" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">同时还有别的schema，比如说</span></li><li id="u49b8e862" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">input_schema：首次输入（默认同 state_schema）。</span></li><li id="u4f06bb6b" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">output_schema：最终输出筛选（只暴露其中定义的 channel）。</span></li><li id="u642d655a" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">context_schema：只读运行上下文（非状态，提供 run-scoped 信息）。</span></li></ul><p id="u2dde90fc" class="ne-p"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">这里的schema就是输入的状态，状态就是一个输入类，包含了需要使用到的字段，归约器之类的</span></p><p id="uc9b7c21d" class="ne-p"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">所有 schema 的字段会被解析为 channel（数据流通道）或 managed value（受框架托管）。</span></p><h2 id="f3286008"><span class="ne-text" style="color: #000000; background-color: #FFFFFF"> Channel / ManagedValue 解析</span></h2><p id="ud94a1ba1" class="ne-p"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">sechmea的字段都会经过如下这个方法进行解析，按照不同的字段+归约器类型解析为不同的类，具体不同字段的解析逻辑如下：</span></p><p id="ud2bc353f" class="ne-p"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">注解经 _get_channels：</span></p><ul class="ne-ul"><li id="u8b383256" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">Annotated[T, BinaryOp] → BinaryOperatorAggregate：同一键多次写入用 reducer 折叠。</span></li><li id="u2d14c336" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">Annotated[T, SomeChannelSubclass] → 指定 channel 类型（例如 LastValue / EphemeralValue 等）。</span></li><li id="ua1301dfe" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">普通类型 → 默认 LastValue。</span></li><li id="uef35d29a" data-lake-index-type="0"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">ManagedValue（如特定标记）→ 不进入普通更新通道，仅框架持有。 特殊单字段且命名为 </span><strong><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">root</span></strong><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px"> 时允许“根值模式”。</span></li></ul><p id="u61548df3" class="ne-p"><span class="ne-text" style="color: #000000; background-color: #FFFFFF; font-size: 13px">这就完成了状态的调用与解析</span></p></details>








+ 在 LangGraph 中，它被用来告诉系统：这个字段在更新时应该使用哪个归约器函数。

✅ 简单理解：`Annotated` 就是“这个字段是 list 类型，但更新时请用 add_messages 来合并新值”。

##### 🔁 `add_messages` 是干什么的？
这是 LangGraph 提供的一个内置归约器函数，用于智能地更新消息列表：

+ 如果是新消息 → 自动追加
+ 如果是已有消息（同一个 ID）→ 自动替换
+ 如果是原始字典 → 自动转为 LangChain 消息对象

✅ 它确保每次代理执行后，消息列表都能正确更新，不会重复、不丢失。

#### Assistant节点定义
```python
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        log_state_change(state, "Assistant节点开始处理")
        
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            logger.info(f"👤 当前乘客ID: {passenger_id}")
            
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            
            logger.info(f"🤖 LLM响应类型: {type(result)}")
            if hasattr(result, 'tool_calls') and result.tool_calls:
                logger.info(f"🔧 LLM决定调用工具: {[tc.get('name', 'unknown') for tc in result.tool_calls]}")
            
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                logger.warning("⚠️ LLM返回空响应，重新提示")
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        
        log_state_change({"messages": [result]}, "Assistant节点处理完成")
        return {"messages": result}
```

这段代码是 LangGraph 客户支持机器人教程中的核心部分，是一个包装类，可以输入LLM和定义的content，八其封装成了一个assistant类，方便后面和tools等别的节点做交互：

##### 🧠 整体目标
**构建一个具备工具调用能力的智能助手（作为后面graph的node），要求输入一个runnable变量（预定义prompt和搭建好的llm作为runnable），具体封装了哪些东西到assitant呢？**：

+ 使用 Anthropic Claude 模型（或可替换为 GPT-4）
+ 绑定一组工具（如航班查询、政策检索）
+ 使用提示词引导模型合理调用工具
+ 自动处理空响应或无效输出

把上述一起封装成一个assitant，具体说一下assitant封装的过程

##### 🧩 1. Assistant 类定义
python

```plain
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
```

+ `runnable` 是一个可执行对象，通常是提示词 + 模型 + 工具绑定后的组合
+ 这个类封装了一个 LangGraph 节点的行为逻辑，有了上述工具之后，可以handle一些情况
1. **🔁**** **`**__call__**`** 方法**

python

```plain
def __call__(self, state: State, config: RunnableConfig):
```

这是代理节点的执行入口，接收：

+ `state`：当前图状态（如消息列表）
+ `config`：运行时配置（如用户 ID）
2. **🔐**** 注入用户信息**

```python
configuration = config.get("configurable", {})
passenger_id = configuration.get("passenger_id", None)
state = {**state, "user_info": passenger_id}
```

+ 从运行时配置中提取 `passenger_id`
+ 注入到状态中，供提示词使用（如 `{user_info}`）

<details class="lake-collapse"><summary id="u20eb3263"><span class="ne-text">state = {**state, &quot;user_info&quot;: passenger_id}语法（吧passen..注入到state字典当中）</span></summary><p id="u0dbca7f2" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="zUS75" class="ne-codeblock language-plain"><code>state = {**state, &quot;user_info&quot;: passenger_id}</code></pre><p id="u73d206e5" class="ne-p"><span class="ne-text">是 Python 中的</span><strong><span class="ne-text">字典解包合并语法</span></strong><span class="ne-text">，它的作用是：在原有 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 字典的基础上，新增或更新一个键 </span><code class="ne-code"><span class="ne-text">&quot;user_info&quot;</span></code><span class="ne-text">，其值为 </span><code class="ne-code"><span class="ne-text">passenger_id</span></code><span class="ne-text">。</span></p><h2 id="e402a60b"><span class="ne-text">🧩</span><span class="ne-text"> 语法拆解</span></h2><ul class="ne-ul"><li id="u3c25674f" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">**state</span></code><span class="ne-text">：表示将原字典 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 中的所有键值对“展开”出来</span></li><li id="u8c0f85ad" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">&quot;user_info&quot;: passenger_id</span></code><span class="ne-text">：是一个新的键值对，添加到展开后的字典中</span></li><li id="u79749b2b" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">{...}</span></code><span class="ne-text">：重新构造一个新的字典</span></li></ul><p id="ud0f7acaa" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 如果 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 中原本没有 </span><code class="ne-code"><span class="ne-text">&quot;user_info&quot;</span></code><span class="ne-text">，这就是新增字段 </span><span class="ne-text">✅</span><span class="ne-text"> 如果 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 中已经有 </span><code class="ne-code"><span class="ne-text">&quot;user_info&quot;</span></code><span class="ne-text">，这就是覆盖原值</span></p><h2 id="c7a13a94"><span class="ne-text">🧠</span><span class="ne-text"> 举个例子</span></h2><p id="u47231c64" class="ne-p"><span class="ne-text">假设原来的 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 是：</span></p><p id="u1c1e1a24" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="GLTSo" class="ne-codeblock language-plain"><code>state = {
    &quot;messages&quot;: [...],
    &quot;step&quot;: 3
}
passenger_id = &quot;3442 587242&quot;</code></pre><p id="ub9d2276b" class="ne-p"><span class="ne-text">执行这句后，新的 </span><code class="ne-code"><span class="ne-text">state</span></code><span class="ne-text"> 就变成：</span></p><p id="u47fc1477" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="hTRkn" class="ne-codeblock language-plain"><code>state = {
    &quot;messages&quot;: [...],
    &quot;step&quot;: 3,
    &quot;user_info&quot;: &quot;3442 587242&quot;
}</code></pre><ul class="ne-ul"><li id="uc1ea0420" data-lake-index-type="0"><span class="ne-text"></span></li></ul></details>
✅ 这样做可以避免 LLM 直接处理身份信息，提升安全性。

3. **🧠**** 调用 LLM 并处理空响应**

```python
result = self.runnable.invoke(state)
```

+ 执行提示词 + 模型 + 工具组合，生成响应

如果响应为空或没有工具调用：

```python
if not result.tool_calls and (
    not result.content
    or isinstance(result.content, list)
    and not result.content[0].get("text")
):
    messages = state["messages"] + [("user", "Respond with a real output.")]
    state = {**state, "messages": messages}
```

+ 自动追加一个“重新回答”的提示，重新调用模型
+ 直到模型返回有效响应为止

✅ 提升鲁棒性，避免代理卡死或无回应。

##### 
#### 定义图


![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1757174898149-3b9708ab-b295-410e-a57c-8e268c62ea34.png)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = InMemorySaver()
part_1_graph = builder.compile(checkpointer=memory)
```

这段代码是 LangGraph 客户支持机器人教程中构建智能代理图的核心部分。它定义了一个最小可运行的图结构，具备工具调用能力和状态持久化能力。我们来逐步拆解它的结构和工程意图：

##### 整体目标
构建一个简单的 2 节点代理图：

+ 节点 1：assistant → 调用 LLM（如 Claude）生成响应
+ 节点 2：tools → 执行工具调用（如查票、改签、订酒店）
+ 边：根据是否有工具调用决定是否进入 tools 节点
+ 状态：使用 `State` 类型，包含消息列表
+ 持久化：使用 `InMemorySaver` 保存状态，支持多轮对话

##### 🧩 1. 导入模块
python

```plain
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
```

+ `InMemorySaver`：图状态的持久化器，保存在内存中（适合教学和调试）
+ `StateGraph`：LangGraph 的核心类，用于构建有状态的执行图
+ `START` / `END`：图的起点和终点标识符
+ `tools_condition`：内置条件函数，用于判断是否需要进入工具节点（如果消息中有 tool_calls）

##### 为assistant节点的创建做准备
1. ** 模型选择与绑定**

python

```plain
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
```

+ 使用 Claude Sonnet 模型作为语言模型
+ 可替换为 GPT-4 或 Claude Haiku（更快但准确率低）
2. **提示词定义**

python

```plain
primary_assistant_prompt = ChatPromptTemplate.from_messages([...]).partial(time=datetime.now)
```

+ 使用 `ChatPromptTemplate` 构建系统提示词
+ 包含系统角色说明 + 当前用户信息 + 当前时间
+ 使用 `{messages}` 占位符插入对话历史

✅ 提示词中强调“使用工具”、“搜索时要坚持”、“扩大搜索范围”等行为策略。

3. **工具列表定义**



```python
part_1_tools = [ TavilySearchResults(...), fetch_user_flight_information, ... ]
```

+ 包含所有可调用工具，如航班查询、政策检索、酒店预订等
+ 每个工具都用 `@tool` 装饰器注册，支持 LangChain 调用
4. **绑定提示词 + 模型 + 工具作为一个runnable变量**

python

```plain
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
```

+ 使用管道符 `|` 将提示词与模型组合
+ 使用 `.bind_tools()` 绑定工具列表
+ 得到一个完整的 `Runnable` 对象，供 `Assistant` 类调用

|语法糖用来组合pipline，详细解释如下

<details class="lake-collapse"><summary id="u43b9ae4b"><span class="ne-text">|语法糖</span></summary><p id="u91bb0a6a" class="ne-p"><span class="ne-text">这个管道符 </span><code class="ne-code"><span class="ne-text">|</span></code><span class="ne-text"> 是 LangChain 中的一个非常重要的语法糖，叫做 </span><strong><span class="ne-text">“可运行对象链式组合”</span></strong><span class="ne-text">（Runnable Piping）。它的作用是把多个可执行组件（如提示词、模型、工具）</span><strong><span class="ne-text">串联成一个完整的执行链</span></strong><span class="ne-text">，形成一个新的 </span><code class="ne-code"><span class="ne-text">Runnable</span></code><span class="ne-text"> 对象。</span></p><h2 id="sfewp"><span class="ne-text">🧩</span><span class="ne-text"> 这句代码的结构解析</span></h2><p id="ub804fcac" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="zIARm" class="ne-codeblock language-plain"><code>part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)</code></pre><p id="ud634a079" class="ne-p"><span class="ne-text">它的含义是：</span></p><ol class="ne-ol"><li id="ubccbeaf3" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">primary_assistant_prompt</span></code><span class="ne-text">：一个提示词模板（Prompt），用于将状态格式化为模型输入。</span></li><li id="u981f224e" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">llm.bind_tools(part_1_tools)</span></code><span class="ne-text">：一个语言模型（Claude Sonnet），绑定了一组工具，使它具备调用工具的能力。</span></li><li id="u0e3b5922" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">|</span></code><span class="ne-text">：管道符，把提示词和模型组合成一个新的 </span><code class="ne-code"><span class="ne-text">Runnable</span></code><span class="ne-text">，表示“先格式化提示 → 再调用模型”。</span></li></ol><p id="u3950957d" class="ne-p"><span class="ne-text">最终得到的 </span><code class="ne-code"><span class="ne-text">part_1_assistant_runnable</span></code><span class="ne-text"> 是一个完整的智能助手执行链，可以通过 </span><code class="ne-code"><span class="ne-text">.invoke(state)</span></code><span class="ne-text"> 来运行。</span></p><h2 id="ajDOU"><span class="ne-text">🔧</span><span class="ne-text"> 管道符的底层行为</span></h2><p id="u2e8f38c4" class="ne-p"><span class="ne-text">这个语法糖的底层逻辑是：</span></p><p id="ub2fed6dd" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="WgFr0" class="ne-codeblock language-plain"><code>runnable = prompt.pipe(model)</code></pre><p id="u9ab6ac3f" class="ne-p"><span class="ne-text">也就是说：</span></p><ul class="ne-ul"><li id="ue2135ef8" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">prompt.invoke(state)</span></code><span class="ne-text"> → 得到格式化后的消息列表</span></li><li id="u37bfb75c" data-lake-index-type="0"><code class="ne-code"><span class="ne-text">model.invoke(messages)</span></code><span class="ne-text"> → 得到模型响应（可能包含工具调用）</span></li><li id="u001d3f28" data-lake-index-type="0"><span class="ne-text">整个链条变成一个新的 </span><code class="ne-code"><span class="ne-text">Runnable</span></code><span class="ne-text">，可以继续组合、调用、调试</span></li></ul><p id="ue27a0cca" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 每个 </span><code class="ne-code"><span class="ne-text">Runnable</span></code><span class="ne-text"> 都是一个“函数对象”，可以组合成更复杂的执行链。</span></p><h2 id="N1Ijw"><span class="ne-text">🧠</span><span class="ne-text"> 为什么要用管道符？</span></h2><p id="ua2353004" class="ne-p"><span class="ne-text">优势</span></p><p id="u584ebea8" class="ne-p"><span class="ne-text">说明</span></p><p id="u2a1cc1b1" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 简洁</span></p><p id="udf1678a6" class="ne-p"><span class="ne-text">一行代码就能组合多个步骤</span></p><p id="u7410d30b" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 可读性强</span></p><p id="ua08f6d31" class="ne-p"><span class="ne-text">从左到右表示执行顺序</span></p><p id="u9925ea2d" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 可复用</span></p><p id="u58e0acd4" class="ne-p"><span class="ne-text">每个组件都是独立的 </span><code class="ne-code"><span class="ne-text">Runnable</span></code></p><p id="ue9109e7e" class="ne-p"><span class="ne-text">，可单独测试</span></p><p id="ue3508cc5" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 可扩展</span></p><p id="uabbd0bdb" class="ne-p"><span class="ne-text">可以继续加上 </span><code class="ne-code"><span class="ne-text">.pipe(output_parser)</span></code></p><p id="u8a24dfe7" class="ne-p"><span class="ne-text"> 等步骤</span></p><p id="u63620292" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 与 LangGraph 兼容</span></p><p id="uc282d5b8" class="ne-p"><span class="ne-text">可作为图节点的执行体，支持 </span><code class="ne-code"><span class="ne-text">.invoke()</span></code></p><p id="uae79a5ba" class="ne-p"><span class="ne-text">、</span><code class="ne-code"><span class="ne-text">.stream()</span></code></p><p id="u7c9f6110" class="ne-p"><span class="ne-text"> 等方法</span></p><h2 id="japxk"><span class="ne-text">📦</span><span class="ne-text"> 举个例子</span></h2><p id="u9f9ae839" class="ne-p"><span class="ne-text">假设你有一个提示词和一个模型：</span></p><p id="u0c9ec164" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="sm2vO" class="ne-codeblock language-plain"><code>prompt = ChatPromptTemplate.from_template(&quot;Tell me a joke about {topic}&quot;)
model = ChatAnthropic(model=&quot;claude-3-haiku&quot;)</code></pre><p id="ua81f0ed6" class="ne-p"><span class="ne-text">你可以这样组合：</span></p><p id="u9948be33" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="gfM06" class="ne-codeblock language-plain"><code>chain = prompt | model
chain.invoke({&quot;topic&quot;: &quot;bears&quot;})</code></pre><p id="u27b79c25" class="ne-p"><span class="ne-text">这就完成了：</span><strong><span class="ne-text">格式化 → 推理 → 输出</span></strong><span class="ne-text"> 的完整流程。</span></p><h2 id="xYgO0"><span class="ne-text">🧩</span><span class="ne-text"> 在 LangGraph 中的应用场景</span></h2><p id="u8acec4a5" class="ne-p"><span class="ne-text">在 LangGraph 中，每个节点都可以绑定一个 </span><code class="ne-code"><span class="ne-text">Runnable</span></code><span class="ne-text">。比如：</span></p><p id="ucf2e8f78" class="ne-p"><span class="ne-text">python</span></p><pre data-language="plain" id="nJt5s" class="ne-codeblock language-plain"><code>builder.add_node(&quot;assistant&quot;, Assistant(part_1_assistant_runnable))</code></pre><p id="u90f13ad6" class="ne-p"><span class="ne-text">这里的 </span><code class="ne-code"><span class="ne-text">part_1_assistant_runnable</span></code><span class="ne-text"> 就是通过管道符组合出来的执行链，封装了提示词、模型和工具调用能力。</span></p><p id="u45c4af77" class="ne-p"><span class="ne-text">如果你正在构建自己的代理系统，这种管道式组合是非常推荐的模式。需要我帮你扩展这个链条支持响应解析器（如 </span><code class="ne-code"><span class="ne-text">StrOutputParser</span></code><span class="ne-text">）或多模型切换吗？我们可以一起设计一个更灵活的执行链。</span></p></details>
##### 🧠 2. 初始化图构建器
python

```plain
builder = StateGraph(State)
```

+ 创建一个图构建器，状态类型为 `State`（通常是一个 TypedDict，包含 `messages` 字段）
+ 所有节点的输入输出都遵循这个状态结构

##### 🔧 3. 添加节点（Node）
实例化assistant在添加节点的过程中完成了

```plain
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
```

+ `"assistant"` 节点：调用 LLM（如 Claude）生成响应，可能包含工具调用
+ `"tools"` 节点：执行工具调用，并将结果写入消息列表
+ `create_tool_node_with_fallback(...)`：构建一个带错误处理的工具节点，避免工具调用失败导致图崩溃

输入满足以下三种情况即可

1. 可调用（函数/方法/实现了__call__的对象/Runnable）

2. 接受状态字典（和可选的config）

3. 返回合法的状态更新（字典/Command/None/列表）

<details class="lake-collapse"><summary id="u259b207d"><span class="ne-text" style="color: #000000">Add_node方法部分解释</span></summary><p id="u5870eba2" class="ne-p"><br></p><p id="u5615b74a" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">具体看一下源码，</span></p><p id="ufe9bf63c" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">之后再add_node中的处理逻辑如下：（当然这个函数根据是否输入name和schema有多个重载，看情况决定，这里跳了一个</span></p><pre data-language="python" id="PO9cr" class="ne-codeblock language-python"><code>  def add_node(
        self,
        node: str | StateNode[NodeInputT, ContextT],
        action: StateNode[NodeInputT, ContextT] | None = None,
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: type[NodeInputT] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -&gt; Self:
        &quot;&quot;&quot;Add a new node to the state graph.

        Args:
            node: The function or runnable this node will run.
                If a string is provided, it will be used as the node name, and action will be used as the function or runnable.
            action: The action associated with the node. (default: None)
                Will be used as the node function or runnable if `node` is a string (node name).
            defer: Whether to defer the execution of the node until the run is about to end.
            metadata: The metadata associated with the node. (default: None)
            input_schema: The input schema for the node. (default: the graph's state schema)
            retry_policy: The retry policy for the node. (default: None)
                If a sequence is provided, the first matching policy will be applied.
            cache_policy: The cache policy for the node. (default: None)
            destinations: Destinations that indicate where a node can route to.
                This is useful for edgeless graphs with nodes that return `Command` objects.
                If a dict is provided, the keys will be used as the target node names and the values will be used as the labels for the edges.
                If a tuple is provided, the values will be used as the target node names.
                NOTE: this is only used for graph rendering and doesn't have any effect on the graph execution.

        Example:
            ```python
            from typing_extensions import TypedDict

            from langchain_core.runnables import RunnableConfig
            from langgraph.graph import START, StateGraph

            class State(TypedDict):
                x: int

            def my_node(state: State, config: RunnableConfig) -&gt; State:
                return {&quot;x&quot;: state[&quot;x&quot;] + 1}

            builder = StateGraph(State)
            builder.add_node(my_node)  # node name will be 'my_node'
            builder.add_edge(START, &quot;my_node&quot;)
            graph = builder.compile()
            graph.invoke({&quot;x&quot;: 1})
            # {'x': 2}
            ```

        Example: Customize the name:
            ```python
            builder = StateGraph(State)
            builder.add_node(&quot;my_fair_node&quot;, my_node)
            builder.add_edge(START, &quot;my_fair_node&quot;)
            graph = builder.compile()
            graph.invoke({&quot;x&quot;: 1})
            # {'x': 2}
            ```

        Returns:
            Self: The instance of the state graph, allowing for method chaining.
        &quot;&quot;&quot;
        if (retry := kwargs.get(&quot;retry&quot;, MISSING)) is not MISSING:
            warnings.warn(
                &quot;`retry` is deprecated and will be removed. Please use `retry_policy` instead.&quot;,
                category=LangGraphDeprecatedSinceV05,
            )
            if retry_policy is None:
                retry_policy = retry  # type: ignore[assignment]

        if (input_ := kwargs.get(&quot;input&quot;, MISSING)) is not MISSING:
            warnings.warn(
                &quot;`input` is deprecated and will be removed. Please use `input_schema` instead.&quot;,
                category=LangGraphDeprecatedSinceV05,
            )
            if input_schema is None:
                input_schema = cast(Union[type[NodeInputT], None], input_)

        if not isinstance(node, str):
            action = node
            if isinstance(action, Runnable):
                node = action.get_name()
            else:
                node = getattr(action, &quot;__name__&quot;, action.__class__.__name__)
            if node is None:
                raise ValueError(
                    &quot;Node name must be provided if action is not a function&quot;
                )
        if self.compiled:
            logger.warning(
                &quot;Adding a node to a graph that has already been compiled. This will &quot;
                &quot;not be reflected in the compiled graph.&quot;
            )
        if not isinstance(node, str):
            action = node
            node = cast(str, getattr(action, &quot;name&quot;, getattr(action, &quot;__name__&quot;, None)))
            if node is None:
                raise ValueError(
                    &quot;Node name must be provided if action is not a function&quot;
                )
        if action is None:
            raise RuntimeError
        if node in self.nodes:
            raise ValueError(f&quot;Node `{node}` already present.&quot;)
        if node == END or node == START:
            raise ValueError(f&quot;Node `{node}` is reserved.&quot;)

        for character in (NS_SEP, NS_END):
            if character in node:
                raise ValueError(
                    f&quot;'{character}' is a reserved character and is not allowed in the node names.&quot;
                )

        inferred_input_schema = None

        ends: tuple[str, ...] | dict[str, str] = EMPTY_SEQ
        try:
            if (
                isfunction(action)
                or ismethod(action)
                or ismethod(getattr(action, &quot;__call__&quot;, None))
            ) and (
                hints := get_type_hints(getattr(action, &quot;__call__&quot;))
                or get_type_hints(action)
            ):
                if input_schema is None:
                    first_parameter_name = next(
                        iter(
                            inspect.signature(
                                cast(FunctionType, action)
                            ).parameters.keys()
                        )
                    )
                    if input_hint := hints.get(first_parameter_name):
                        if isinstance(input_hint, type) and get_type_hints(input_hint):
                            inferred_input_schema = input_hint
                if rtn := hints.get(&quot;return&quot;):
                    # Handle Union types
                    rtn_origin = get_origin(rtn)
                    if rtn_origin is Union:
                        rtn_args = get_args(rtn)
                        # Look for Command in the union
                        for arg in rtn_args:
                            arg_origin = get_origin(arg)
                            if arg_origin is Command:
                                rtn = arg
                                rtn_origin = arg_origin
                                break

                    # Check if it's a Command type
                    if (
                        rtn_origin is Command
                        and (rargs := get_args(rtn))
                        and get_origin(rargs[0]) is Literal
                        and (vals := get_args(rargs[0]))
                    ):
                        ends = vals
        except (NameError, TypeError, StopIteration):
            pass

        if destinations is not None:
            ends = destinations

        if input_schema is not None:
            self.nodes[node] = StateNodeSpec[NodeInputT, ContextT](
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=input_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )
        elif inferred_input_schema is not None:
            self.nodes[node] = StateNodeSpec(
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=inferred_input_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )
        else:
            self.nodes[node] = StateNodeSpec[StateT, ContextT](
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=self.state_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )

        input_schema = input_schema or inferred_input_schema
        if input_schema is not None:
            self._add_schema(input_schema)

        return self</code></pre><p id="u886cd738" class="ne-p"><span class="ne-text" style="color: #000000">从代码中可以看到，</span><span class="ne-text" style="color: #000000">add_node</span><span class="ne-text" style="color: #000000"> </span><span class="ne-text" style="color: #000000">接受的</span><span class="ne-text" style="color: #000000"> </span><span class="ne-text" style="color: #000000">action</span><span class="ne-text" style="color: #000000"> </span><span class="ne-text" style="color: #000000">参数类型是：</span></p><pre data-language="python" id="fJ8Kd" class="ne-codeblock language-python"><code>action: StateNode[NodeInputT, ContextT]</code></pre><p id="uc4cb6c34" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">其中 </span><span class="ne-text" style="color: #000000">StateNode</span><span class="ne-text" style="color: #000000; font-size: 13px"> 是一个类型别名，定义为：</span></p><pre data-language="python" id="stFaO" class="ne-codeblock language-python"><code>StateNode = Union[
    Runnable[dict, Any],
    Callable[..., Any],
    Callable[..., Awaitable[Any]],
]</code></pre><p id="uc1b5070c" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">所以只要输入时runnable或可被调用的对象即可</span></p><h2 id="92238fe1"><span class="ne-text" style="color: #000000">自动转换机制</span></h2><p id="u15c0b637" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">在</span><span class="ne-text" style="color: #000000; font-size: 13px"> </span><span class="ne-text" style="color: #000000; font-size: 13px">add_node</span><span class="ne-text" style="color: #000000; font-size: 13px"> </span><span class="ne-text" style="color: #000000; font-size: 13px">方法中，所有节点都会通过</span><span class="ne-text" style="color: #000000; font-size: 13px"> </span><span class="ne-text" style="color: #000000; font-size: 13px">coerce_to_runnable</span><span class="ne-text" style="color: #000000; font-size: 13px"> </span><span class="ne-text" style="color: #000000; font-size: 13px">转换：</span></p><p id="u5d0c72ad" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">StateNodeSpec</span><span class="ne-text" style="color: #000000; font-size: 14px">[</span><span class="ne-text" style="color: #000000; font-size: 14px">NodeInputT, ContextT</span><span class="ne-text" style="color: #000000; font-size: 14px">]</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span></p><p id="u05ae2a4a" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">    coerce_to_runnable</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">action, </span><span class="ne-text" style="color: #000000; font-size: 14px">name</span><span class="ne-text" style="color: #000000; font-size: 14px">=</span><span class="ne-text" style="color: #000000; font-size: 14px">node, </span><span class="ne-text" style="color: #000000; font-size: 14px">trace</span><span class="ne-text" style="color: #000000; font-size: 14px">=</span><span class="ne-text" style="color: #000000; font-size: 14px">False</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px">,  </span><span class="ne-text" style="color: #000000; font-size: 14px"># 自动转换</span></p><p id="u41881213" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">    </span><span class="ne-text" style="color: #000000; font-size: 14px"># ... 其他参数</span></p><p id="u64d9195b" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">)</span></p><p id="u9659b6e6" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">这个函数会：</span></p><ul class="ne-ul"><li id="u6b566978" data-lake-index-type="0"><span class="ne-text" style="color: #000000; font-size: 13px">如果已经是 Runnable → 直接使用</span></li><li id="uccf189e1" data-lake-index-type="0"><span class="ne-text" style="color: #000000; font-size: 13px">如果是普通函数 → 包装成 RunnableLambda</span></li><li id="u78263e34" data-lake-index-type="0"><span class="ne-text" style="color: #000000; font-size: 13px">如果是异步函数 → 包装成支持异步的 Runnable</span></li></ul><h2 id="53a0ca02"><span class="ne-text" style="color: #000000">你的代码示例</span></h2><p id="u43de0a69" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">在你的代码中可以看到三种用法：</span></p><p id="u5f9ff72d" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px"># 1. 直接使用工具函数（普通函数）</span></p><p id="u51be4153" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">builder.add_node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">&quot;user_info&quot;</span><span class="ne-text" style="color: #000000; font-size: 14px">, fetch_user_flight_information</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span></p><p id="u73a92c25" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px"># 2. 使用自定义类实例（实现了 __call__ 方法）</span></p><p id="u7213dc3f" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">builder.add_node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">&quot;assistant&quot;</span><span class="ne-text" style="color: #000000; font-size: 14px">, Assistant</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">part_1_assistant_runnable</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span></p><p id="u7f715573" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px"># 3. 使用预构建的工具节点（已经是 Runnable）</span></p><p id="u7f8d5b73" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">builder.add_node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">&quot;tools&quot;</span><span class="ne-text" style="color: #000000; font-size: 14px">, create_tool_node_with_fallback</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">part_1_tools</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span></p><h2 id="f784c1e2"><span class="ne-text" style="color: #000000">节点签名要求</span></h2><p id="u9927e48e" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">无论什么类型，节点都必须满足：</span></p><ul class="ne-ul"><li id="u4bc25cff" data-lake-index-type="0"><strong><span class="ne-text" style="color: #000000; font-size: 13px">输入</span></strong><span class="ne-text" style="color: #000000; font-size: 13px">：接受状态字典（和可选的 config）</span></li><li id="u5149f603" data-lake-index-type="0"><strong><span class="ne-text" style="color: #000000; font-size: 13px">输出</span></strong><span class="ne-text" style="color: #000000; font-size: 13px">：返回状态更新字典、Command 对象或 None</span></li></ul><p id="u001d432f" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px"># 正确的签名示例</span></p><p id="u597e917b" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">def</span><span class="ne-text" style="color: #000000; font-size: 14px"> </span><span class="ne-text" style="color: #000000; font-size: 14px">node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">state</span><span class="ne-text" style="color: #000000; font-size: 14px">: State</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px"> -&gt; </span><span class="ne-text" style="color: #000000; font-size: 14px">dict</span><span class="ne-text" style="color: #000000; font-size: 14px">: ...</span></p><p id="ud52c2a23" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">def</span><span class="ne-text" style="color: #000000; font-size: 14px"> </span><span class="ne-text" style="color: #000000; font-size: 14px">node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">state</span><span class="ne-text" style="color: #000000; font-size: 14px">: State, </span><span class="ne-text" style="color: #000000; font-size: 14px">config</span><span class="ne-text" style="color: #000000; font-size: 14px">: RunnableConfig</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px"> -&gt; </span><span class="ne-text" style="color: #000000; font-size: 14px">dict</span><span class="ne-text" style="color: #000000; font-size: 14px">: ...</span></p><p id="ucb7c70eb" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 14px">async</span><span class="ne-text" style="color: #000000; font-size: 14px"> </span><span class="ne-text" style="color: #000000; font-size: 14px">def</span><span class="ne-text" style="color: #000000; font-size: 14px"> </span><span class="ne-text" style="color: #000000; font-size: 14px">node</span><span class="ne-text" style="color: #000000; font-size: 14px">(</span><span class="ne-text" style="color: #000000; font-size: 14px">state</span><span class="ne-text" style="color: #000000; font-size: 14px">: State</span><span class="ne-text" style="color: #000000; font-size: 14px">)</span><span class="ne-text" style="color: #000000; font-size: 14px"> -&gt; </span><span class="ne-text" style="color: #000000; font-size: 14px">dict</span><span class="ne-text" style="color: #000000; font-size: 14px">: ...</span></p><p id="ubf15e126" class="ne-p"><span class="ne-text" style="color: #000000; font-size: 13px">所以答案是：</span><strong><span class="ne-text" style="color: #000000; font-size: 13px">不必须是 Runnable，但会被自动转换成 Runnable</span></strong><span class="ne-text" style="color: #000000; font-size: 13px">。<br /></span><span class="ne-text" style="color: #000000; font-size: 13px">具体addnode里面的逻辑没有看完<br /></span></p></details>
##### 🔀 4. 添加边（Edge）
python

```plain
builder.add_edge(START, "assistant")
```

+ 图从 `START` 节点开始，进入 `"assistant"` 节点

##### 给边添加条件
```plain
builder.add_conditional_edges("assistant", tools_condition)
```

+ 条件边：如果 `"assistant"` 节点的输出包含工具调用 → 跳转到 `"tools"` 节点
+ 否则 → 跳转到 `END`（默认行为）

```python
builder.add_edge("tools", "assistant")
```

+ 工具执行完后 → 回到 `"assistant"` 节点，让 LLM 继续处理工具结果
+ 形成一个 ReAct 回路：**LLM → 工具 → LLM → 工具 → … → END**

##### 💾 5. 状态持久化器
python

```plain
memory = InMemorySaver()
```

+ 使用内存持久化器保存图状态
+ 支持多轮对话、错误恢复、调试回溯
+ 在生产环境中可以替换为 Redis、SQLite、云存储等持久化方案

##### 🧱 6. 编译图
python

```plain
part_1_graph = builder.compile(checkpointer=memory)
```

+ 将图构建器编译为一个可执行图对象 `part_1_graph`
+ 可以通过 `.invoke()` 或 `.stream()` 方法运行图
+ 图执行时会自动管理状态流转、节点调度、边跳转、工具调用等

##### 🧩 总结：图结构一览
text

```plain
START → assistant ──┐
                    │
          [tool_calls?]──→ tools → assistant
                    │
                    └────→ END
```



#### 实现对话
```python
def run_demo():
    """Run the demo conversation"""
    logger.info("🎬 开始运行演示对话")
    
    # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "你好，我的航班是什么时间？",
        "我可以把航班改到更早的时间吗？我想今天晚些时候就出发。",
        "那把我的航班改到下周的某个时间吧",
        "下一个可用的选项很好",
        "住宿和交通怎么办？",
        "我想要一个经济实惠的酒店，住一周（7天）。我还想租一辆车。",
        "好的，你能为我预订你推荐的酒店吗？听起来不错。",
        "可以，去预订任何价格适中且有空房的酒店。",
        "现在租车方面，我有什么选择？",
        "太好了，我们选最便宜的选项。请预订7天。",
        "很好，现在你有什么短途旅行的推荐吗？",
        "我在那里期间这些活动可以参加吗？",
        "有意思 - 我喜欢博物馆，有什么选择？",
        "好极了，选一个并为我第二天预订。",
    ]

    logger.info(f"📝 准备了 {len(tutorial_questions)} 个测试问题")

    # Update with the backup file so we can restart from the original place in each section
    update_dates(db)
    thread_id = str(uuid.uuid4())
    logger.info(f"🆔 会话ID: {thread_id}")

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    # Create the agent
    part_1_graph = create_agent()
    all_messages = []

    _printed = set()
    for i, question in enumerate(tutorial_questions, 1):
        logger.info(f"🎯 处理第 {i}/{len(tutorial_questions)} 个问题: {question}")
        
        try:
            events = part_1_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)
                if 'messages' in event:
                    all_messages.extend(event['messages'])
        except Exception as e:
            logger.error(f"❌ 处理问题 {i} 时发生错误: {str(e)}")
    
    # 保存完整对话日志
    save_conversation_log(all_messages)
    logger.info("🎬 演示对话结束")

```

要注意的是最后一段

```python
    _printed = set()
    for i, question in enumerate(tutorial_questions, 1):
        logger.info(f"🎯 处理第 {i}/{len(tutorial_questions)} 个问题: {question}")
        
        try:
            events = part_1_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)
                if 'messages' in event:
                    all_messages.extend(event['messages'])
        except Exception as e:
            logger.error(f"❌ 处理问题 {i} 时发生错误: {str(e)}")
```

这里是把message,config作为输入输入到part_1_graph这个图当中，调用了stream函数，返回一个迭代器evnts，在每次调用evnt的时候，实现一个节点到另一个节点的切换，并打印event函数

### 进一步迭代
之前的问题是

1. 用户没有最终决定权
2. ass调用的工具太多了

先解决第一个问题，调用工具的时候应该由用户同意

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1757239421774-6e32c0b0-a9fc-49db-a335-6dd07e312a35.png)

#### 用户具有最终决定权
在创建图的时候增加一个参数即可

```python
part_2_graph = builder.compile(
        checkpointer=memory,
        # NEW: The graph will always halt before executing the "tools" node.
        # The user can approve or reject (or even alter the request) before
        # the assistant continues
        interrupt_before=["tools"],
    )
```

后面是中断处理逻辑，

```python
for question in tutorial_questions:
    events = part_2_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_2_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_2_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_2_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_2_graph.get_state(config)
```

可以看出，中断在内部发生后，events不继续生成了，直接从循环中跳出，之后通过state(config)输入线程id获得之前图状态，并加以人类手动同意的逻辑，下面拆开了说一下：

##### 中断状态获取
你看到的这行：

```python
snapshot = part_2_graph.get_state(config)
```

是在“从检查点存储器里读取该会话最新快照”的动作。它配合 checkpointer 和 interrupt 一起工作，用来判断图是否暂停在某个中断点、下一步要去哪。

1. **get_state 做了什么**
+ **按 thread_id 取回快照：**从 checkpointer（内存或持久化存储）中，用 config.configurable.thread_id 作为键，拿到该会话“最新一次保存的状态快照”（snapshot）。
+ **包含两类数据：**
    - **业务状态 State：**比如 messages 历史、你自定义的字段（如 user_info）、以及节点运行时产生的增量。
    - **运行元数据：**例如当前对话所处的对话栈/节点、以及一个非常关键的字段 `next`，表示“下一步要进入的节点或者等待外部输入的占位信息”。

一句话：get_state 返回“当前会话的最新状态+运行指示”，让你知道接下来是继续自动跑，还是卡在中断点等你干预。

2. **什么时候会进入到这行代码**
+ **典型用法：**你在每次 `.stream(...)` 或 `.invoke(...)` 之后，立刻调用 `get_state(config)`。这样可以判断刚才那一步有没有触发中断。
+ **与 interrupt 的关系：**如果你配置了 `interrupt_before=["tools"]`，当引擎判定下一步要进入 "tools" 节点时，会先写入快照并停止执行，然后你调用 `get_state` 就会看到 `snapshot.next` 被设置（表示“正等待你确认是否继续进入 tools”）。
+ **如果没有中断：**图正常跑到终点（或跑到下一个可继续的节点），`snapshot.next` 会是空/None。



## 自己搭建的笔记管理agent
### 需求分析，项目框架
#### 需求
##### 笔记增加
1. 按照特定



#### 项目架构


### 微调
项目参考：[https://github.com/xming521/WeClone/blob/master/README_zh.md](https://github.com/xming521/WeClone/blob/master/README_zh.md)

微信数据获取参考[https://qqqqqf-q.github.io/Qing-Digital-Self/guide/prepare-data.html](https://qqqqqf-q.github.io/Qing-Digital-Self/guide/prepare-data.html)

多模态数据处理：



数据读入及预处理：

#### WeClone sft源码阅读




#### 数据预处理
执行 `processor.main()` 时，流程如下：

1. **预解析数据集**

如果是 Telegram 平台，先调用 `process_telegram_dataset()` 做格式转换。

先参考下process_telegram_dataset() 处理tele是怎么处理的

[https://github.com/xming521/WeClone/blob/master/weclone/data/chat_parsers/telegram_parser.py#L285](https://github.com/xming521/WeClone/blob/master/weclone/data/chat_parsers/telegram_parser.py#L285)

在这里详细说明了处理成csv时，csv的具体格式是什么样的

```python
def to_csv(self, chat_messages: List[ChatMessage], output_file: str):
```

```python
 fieldnames = [
            "id",
            "MsgSvrID",
            "type_name",
            "is_sender",
            "talker",
            "room_name",
            "msg",
            "src",
            "CreateTime",
            "is_forward",
        ]

```

之后

2. **检查 CSV 数据目录**
    - 确保 `./dataset/csv` 存在且有文件。
3. **加载 CSV 文件列表**
    - `get_csv_files()` 按文件名中的序号排序。
4. **逐文件处理**
    - `load_csv()`：读取 CSV → 过滤跳过类型 → PII 检测 → 屏蔽词过滤 → 图片检查与标记。

```plain
load_csv(file_path)
    ├── 读取 CSV → DataFrame
    ├── 删除 skip_type_list 类型
    ├── 删除自己转发的消息
    ├── 文本消息：
    │     ├── 去换行
    │     ├── PII 检测
    │     └── 屏蔽词过滤
    ├── 非文本消息：
    │     ├── GIF → 动画表情/sticker
    │     ├── 图片 → 检查文件 → 标注 <image> 或 Cut
    │     └── 贴纸 → 清空 src
    ├── 删除空行
    ├── 转换 CreateTime 格式
    └── 转成 ChatMessage 列表
```

    - `group_consecutive_messages()`：合并同一人的连续消息，遇到 cut 类型插入 `CutMessage`。
    - 累积到 `message_list`。
5. **匹配问答对**
    - `match_qa()`：根据对话顺序和策略，把用户消息和助手回复配成 `QaPair`，附带图片和系统提示。
6. **图片识别（可选）**
    - 如果启用 `image_processor`，并行处理图片转文字。
7. **LLM 清洗（可选）**
    - 调用 `self.clean_strategy.judge()` 对 QA 对进行清洗。
8. **保存结果**
    - `save_result()`：将 QA 对保存为 `./dataset/res_csv/sft/sft-my.json`。
9. **执行统计脚本**
    - `_execute_length_cdf_script()`：调用 `length_cdf.py` 计算数据长度分布。



#### LoRA微调
1. 计算下在qwen2.5上，lora不同r时，参数量的大小

<details class="lake-collapse"><summary id="u29cd4cad"><span class="ne-text">LoRA参数量计算</span></summary><h2 id="j33q1"><span class="ne-text">1️⃣</span><span class="ne-text"> 已知条件</span></h2><ul class="ne-ul"><li id="uea46ead5" data-lake-index-type="0"><span class="ne-text">模型：Qwen2.5‑7B</span></li><li id="uf75c51ed" data-lake-index-type="0"><span class="ne-text">隐藏维度：</span><span class="ne-text">d</span><span class="ne-text">model</span><span class="ne-text">=</span><span class="ne-text">3584</span><span class="ne-text">d_{\text{model}} = 3584</span><span class="ne-text">d</span><span class="ne-text">model</span><span class="ne-text">=</span><span class="ne-text">3584</span></li><li id="u0ca39783" data-lake-index-type="0"><span class="ne-text">LoRA rank：</span><span class="ne-text">r</span><span class="ne-text">=</span><span class="ne-text">8</span><span class="ne-text">r = 8</span><span class="ne-text">r</span><span class="ne-text">=</span><span class="ne-text">8</span></li><li id="ubaae16a4" data-lake-index-type="0"><span class="ne-text">插入矩阵：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u05a9af82" data-lake-index-type="0"><span class="ne-text">文本注意力：</span><code class="ne-code"><span class="ne-text">q_proj</span></code><span class="ne-text">, </span><code class="ne-code"><span class="ne-text">v_proj</span></code></li><li id="u07be9841" data-lake-index-type="0"><span class="ne-text">视觉融合 MLP：</span><code class="ne-code"><span class="ne-text">visual.merger.mlp.0</span></code><span class="ne-text">, </span><code class="ne-code"><span class="ne-text">visual.merger.mlp.2</span></code></li></ul></ul><ul class="ne-ul"><li id="ub0b72729" data-lake-index-type="0"><span class="ne-text">层数：假设 32 层（Transformer 文本层数常用 32 层）</span></li><li id="u933bf90a" data-lake-index-type="0"><span class="ne-text">MLP 视觉层：假设每个矩阵大小 </span><span class="ne-text">d</span><span class="ne-text">i</span><span class="ne-text">n</span><span class="ne-text">×</span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">=</span><span class="ne-text">3584</span><span class="ne-text">×</span><span class="ne-text">3584</span><span class="ne-text">d_{in} \times d_{out} = 3584 \times 3584</span><span class="ne-text">d</span><span class="ne-text">in</span><span class="ne-text">×</span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">=</span><span class="ne-text">3584</span><span class="ne-text">×</span><span class="ne-text">3584</span><span class="ne-text"> （为了估算，实际可能略有差异）</span></li></ul><hr id="KlOup" class="ne-hr"><h2 id="MIu1c"><span class="ne-text">2️⃣</span><span class="ne-text"> LoRA 参数计算公式</span></h2><p id="u51d6d06f" class="ne-p"><span class="ne-text">单个矩阵可训练参数数目：</span></p><p id="ud4b8afc5" class="ne-p"><span class="ne-text">params_per_matrix</span><span class="ne-text">=</span><span class="ne-text">r</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">i</span><span class="ne-text">n</span><span class="ne-text">+</span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">)</span><span class="ne-text">\text{params\_per\_matrix} = r \cdot (d_{in} + d_{out})</span><span class="ne-text">params_per_matrix</span><span class="ne-text">=</span><span class="ne-text">r</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">in</span><span class="ne-text">+</span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">)</span></p><p id="u5609c368" class="ne-p"><span class="ne-text">低秩矩阵 A、B 的参数分别是 </span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">×</span><span class="ne-text">r</span><span class="ne-text">d_{out} \times r</span><span class="ne-text">d</span><span class="ne-text">o</span><span class="ne-text">u</span><span class="ne-text">t</span><span class="ne-text">×</span><span class="ne-text">r</span><span class="ne-text"> 和 </span><span class="ne-text">r</span><span class="ne-text">×</span><span class="ne-text">d</span><span class="ne-text">i</span><span class="ne-text">n</span><span class="ne-text">r \times d_{in}</span><span class="ne-text">r</span><span class="ne-text">×</span><span class="ne-text">d</span><span class="ne-text">in</span><span class="ne-text">，总和就是上式。</span></p><hr id="hr1IE" class="ne-hr"><h2 id="EnR91"><span class="ne-text">3️⃣</span><span class="ne-text"> 文本注意力矩阵（q_proj 和 v_proj）</span></h2><ul class="ne-ul"><li id="ube2e6ad2" data-lake-index-type="0"><span class="ne-text">每层两个矩阵：</span></li></ul><p id="u9f944bcc" class="ne-p"><span class="ne-text">per_layer_text</span><span class="ne-text">=</span><span class="ne-text">2</span><span class="ne-text">⋅</span><span class="ne-text">r</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">m</span><span class="ne-text">o</span><span class="ne-text">d</span><span class="ne-text">e</span><span class="ne-text">l</span><span class="ne-text">+</span><span class="ne-text">d</span><span class="ne-text">m</span><span class="ne-text">o</span><span class="ne-text">d</span><span class="ne-text">e</span><span class="ne-text">l</span><span class="ne-text">)</span><span class="ne-text">=</span><span class="ne-text">2</span><span class="ne-text">⋅</span><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">)</span><span class="ne-text">\text{per\_layer\_text} = 2 \cdot r \cdot (d_{model} + d_{model}) = 2 \cdot 8 \cdot (3584 + 3584) </span><span class="ne-text">per_layer_text</span><span class="ne-text">=</span><span class="ne-text">2</span><span class="ne-text">⋅</span><span class="ne-text">r</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">d</span><span class="ne-text">m</span><span class="ne-text">o</span><span class="ne-text">d</span><span class="ne-text">e</span><span class="ne-text">l</span><span class="ne-text">+</span><span class="ne-text">d</span><span class="ne-text">m</span><span class="ne-text">o</span><span class="ne-text">d</span><span class="ne-text">e</span><span class="ne-text">l</span><span class="ne-text">)</span><span class="ne-text">=</span><span class="ne-text">2</span><span class="ne-text">⋅</span><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">)</span></p><p id="u0219b51d" class="ne-p"><span class="ne-text">逐步算：</span></p><ul class="ne-ul"><li id="u0c85566a" data-lake-index-type="0"><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">=</span><span class="ne-text">7168</span><span class="ne-text">3584 + 3584 = 7168</span><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">=</span><span class="ne-text">7168</span></li><li id="ud63d6088" data-lake-index-type="0"><span class="ne-text">7168</span><span class="ne-text">×</span><span class="ne-text">8</span><span class="ne-text">=</span><span class="ne-text">57344</span><span class="ne-text">7168 \times 8 = 57344</span><span class="ne-text">7168</span><span class="ne-text">×</span><span class="ne-text">8</span><span class="ne-text">=</span><span class="ne-text">57344</span></li><li id="u238a316b" data-lake-index-type="0"><span class="ne-text">2 个矩阵：</span><span class="ne-text">57344</span><span class="ne-text">×</span><span class="ne-text">2</span><span class="ne-text">=</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">57344 \times 2 = 114,688</span><span class="ne-text">57344</span><span class="ne-text">×</span><span class="ne-text">2</span><span class="ne-text">=</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span></li><li id="ua6942a0f" data-lake-index-type="0"><span class="ne-text">每层文本 LoRA 参数 ≈ </span><strong><span class="ne-text">114,688</span></strong></li><li id="ua6feefc1" data-lake-index-type="0"><span class="ne-text">全层（32 层）：</span></li></ul><p id="uec88aff4" class="ne-p"><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">×</span><span class="ne-text">32</span><span class="ne-text">=</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">670</span><span class="ne-text">,</span><span class="ne-text">016</span><span class="ne-text">≈</span><span class="ne-text">3.67</span><span class="ne-text">M</span><span class="ne-text">114,688 \times 32 = 3,670,016 \approx 3.67\text{M}</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">×</span><span class="ne-text">32</span><span class="ne-text">=</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">670</span><span class="ne-text">,</span><span class="ne-text">016</span><span class="ne-text">≈</span><span class="ne-text">3.67</span><span class="ne-text">M</span></p><hr id="APS3W" class="ne-hr"><h2 id="DYwT9"><span class="ne-text">4️⃣</span><span class="ne-text"> 视觉融合 MLP 矩阵（mlp.0 和 mlp.2）</span></h2><p id="u80e9b7ff" class="ne-p"><span class="ne-text">假设 MLP 只有 1 层 LoRA，每个矩阵大小同样用 3584 × 3584 估算：</span></p><ul class="ne-ul"><li id="u72fcc0ff" data-lake-index-type="0"><span class="ne-text">单矩阵 LoRA 参数：</span></li></ul><p id="u1670cd07" class="ne-p"><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">)</span><span class="ne-text">=</span><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">7168</span><span class="ne-text">=</span><span class="ne-text">57</span><span class="ne-text">,</span><span class="ne-text">344</span><span class="ne-text">8 \cdot (3584 + 3584) = 8 \cdot 7168 = 57,344</span><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">(</span><span class="ne-text">3584</span><span class="ne-text">+</span><span class="ne-text">3584</span><span class="ne-text">)</span><span class="ne-text">=</span><span class="ne-text">8</span><span class="ne-text">⋅</span><span class="ne-text">7168</span><span class="ne-text">=</span><span class="ne-text">57</span><span class="ne-text">,</span><span class="ne-text">344</span></p><ul class="ne-ul"><li id="u6e543b89" data-lake-index-type="0"><span class="ne-text">两个矩阵：</span></li></ul><p id="ucc20eacf" class="ne-p"><span class="ne-text">57</span><span class="ne-text">,</span><span class="ne-text">344</span><span class="ne-text">×</span><span class="ne-text">2</span><span class="ne-text">=</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">57,344 \times 2 = 114,688</span><span class="ne-text">57</span><span class="ne-text">,</span><span class="ne-text">344</span><span class="ne-text">×</span><span class="ne-text">2</span><span class="ne-text">=</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span></p><p id="u62012b7b" class="ne-p"><span class="ne-text">视觉 LoRA 参数量一般不按层数叠加，mlp.0 和 mlp.2 只有一份权重，每层只有一套 LoRA。</span></p><hr id="qHG1R" class="ne-hr"><h2 id="lS5JD"><span class="ne-text">5️⃣</span><span class="ne-text"> 总可训练参数量</span></h2><p id="u2d899521" class="ne-p"><span class="ne-text">total LoRA params</span><span class="ne-text">=</span><span class="ne-text">文本部分</span><span class="ne-text">+</span><span class="ne-text">视觉部分</span><span class="ne-text">=</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">670</span><span class="ne-text">,</span><span class="ne-text">016</span><span class="ne-text">+</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">≈</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">784</span><span class="ne-text">,</span><span class="ne-text">704</span><span class="ne-text">\text{total LoRA params} = \text{文本部分} + \text{视觉部分} = 3,670,016 + 114,688 \approx 3,784,704</span><span class="ne-text">total LoRA params</span><span class="ne-text">=</span><span class="ne-text">文本部分</span><span class="ne-text">+</span><span class="ne-text">视觉部分</span><span class="ne-text">=</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">670</span><span class="ne-text">,</span><span class="ne-text">016</span><span class="ne-text">+</span><span class="ne-text">114</span><span class="ne-text">,</span><span class="ne-text">688</span><span class="ne-text">≈</span><span class="ne-text">3</span><span class="ne-text">,</span><span class="ne-text">784</span><span class="ne-text">,</span><span class="ne-text">704</span></p><p id="u139421c9" class="ne-p"><span class="ne-text">✅</span><span class="ne-text"> 结果：</span><strong><span class="ne-text">≈3.78M 可训练参数</span></strong></p><p id="uf4dc6fd6" class="ne-p"><span class="ne-text">这说明 rank=8 时，这种配置是非常轻量的 LoRA 微调，比全参数微调 7B（≈70亿参数）小了 </span><strong><span class="ne-text">上千倍</span></strong><span class="ne-text">。</span></p></details>
2. 计算下token数量

700kb大小的数据，一般token数量2-5倍，这里粗略估计一下3mb左右，r=8的话估摸着是得过拟合了

[https://docs.weclone.love/zh/docs/deploy/data_preprocessing.html#%E7%9B%B8%E5%85%B3%E5%8F%82%E6%95%B0](https://docs.weclone.love/zh/docs/deploy/data_preprocessing.html#%E7%9B%B8%E5%85%B3%E5%8F%82%E6%95%B0)

3. 第一次训练（跑通）

<details class="lake-collapse"><summary id="uf2e97e1d"><span class="ne-text">结果与问题：</span></summary><p id="u56325ca9" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2025/png/43288584/1757998474634-632b5253-0121-41fe-82af-cb2adf1c24c3.png" width="478" id="u043e40dc" class="ne-image"></p><ol class="ne-ol"><li id="u7640089f" data-lake-index-type="0"><span class="ne-text">数据规模与质量严重不足<br /></span><span class="ne-text">547 条样本 对一个 70 亿参数的模型来说几乎是九牛一毛，模型很可能只是记住了数据而没有真正学到泛化能力。</span></li></ol><p id="u640634de" class="ne-p"><span class="ne-text">没有做数据清洗（日志里明确提示 Data cleaning is not enabled），意味着噪声、格式不一致、甚至无关内容都被喂进去了，这会直接稀释有效训练信号。</span></p><p id="ue392053d" class="ne-p"><span class="ne-text">没有验证集，无法判断 loss 下降是否真的意味着泛化提升。</span></p><p id="u602c84af" class="ne-p"><span class="ne-text">批评：你是在用大炮打蚊子，还没瞄准就开火了。</span></p><ol start="2" class="ne-ol"><li id="uad779abc" data-lake-index-type="0"><span class="ne-text">LoRA 配置过于保守<br /></span><span class="ne-text">lora_rank=4 + 只改 q_proj,v_proj，可训练参数占比只有 0.0166%，这几乎是在给模型“戴手铐”训练。</span></li></ol><p id="u55b9cf0a" class="ne-p"><span class="ne-text">这种配置虽然显存占用低，但表达能力受限，尤其是你数据量本来就少，模型几乎没法学到足够的特征。</span></p><p id="u5b38c83c" class="ne-p"><span class="ne-text">批评：你给模型的“改造预算”太吝啬了，想让它学会新技能却不给它足够的自由度。</span></p><ol start="3" class="ne-ol"><li id="uf6286914" data-lake-index-type="0"><span class="ne-text">训练轮次与调度策略不匹配<br /></span><span class="ne-text">2 个 epoch 对小数据集来说可能还没完全收敛，尤其是 cosine 学习率衰减到接近 0 时，后期几乎没在学。</span></li></ol><p id="u248fad73" class="ne-p"><span class="ne-text">没有 early stopping 或中途评估，无法动态调整训练计划。</span></p><p id="u4667265c" class="ne-p"><span class="ne-text">批评：你像是在跑马拉松，但只跑了半程就停下，还没看成绩就收工了。</span></p><ol start="4" class="ne-ol"><li id="u0b4220f2" data-lake-index-type="0"><span class="ne-text">量化训练细节没跟进最佳实践<br /></span><span class="ne-text">你用了 4bit NF4 + double quantization，这本身没问题，但日志提示建议开启 upcast_layernorm，你没开。</span></li></ol><p id="u683766f6" class="ne-p"><span class="ne-text">这种细节会影响数值稳定性，尤其在低比特量化下，可能导致模型学不到最优解。</span></p><p id="ue91b8431" class="ne-p"><span class="ne-text">批评：你在开车上高速，但安全带没系好。</span></p><ol start="5" class="ne-ol"><li id="u5721f10d" data-lake-index-type="0"><span class="ne-text">缺乏效果验证与对比<br /></span><span class="ne-text">训练结束后没有做推理测试、没有和原模型对比输出质量。</span></li></ol><p id="u04c9e793" class="ne-p"><span class="ne-text">没有用指标（BLEU、ROUGE、准确率等）量化效果，完全凭 loss 猜测。</span></p><p id="u66756be9" class="ne-p"><span class="ne-text">批评：你造了一把新刀，却没试着切东西，就说它锋利。</span></p></details>
4. debug熟悉流程

先debug一下数据，把握一下整体代码运行流程，之后按照上述的结果问题进行调优，接下来单开一张专门说一下这里的llama_factory微调流程

##### Llama factroy微调


首先通过外层调用sft之前的准备

```python
def main():
    train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type="train_sft"))
    dataset_config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))

    device = get_current_device()
    if device == "cpu":
        logger.warning("Please note you are using CPU for training, non-Mac devices may encounter issues")

    dataset_info_path = os.path.join(dataset_config.dataset_dir, "dataset_info.json")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f) # 这里的dataset_info包含了数据集名称，数据集格式，数据集中的特殊tag
        data_path = os.path.join(
            dataset_config.dataset_dir, dataset_info.get(train_config.dataset, {}).get("file_name")
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file '{data_path}' does not exist, please check if make-dataset was executed"
            ) #检查数据是否存在

    if not dataset_config.clean_dataset.enable_clean or "image" in dataset_config.include_type:
        logger.info("Data cleaning is not enabled or images are included, will use the original dataset.")
    else:
        cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
        train_config.dataset = cleaner.clean()

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"Fine-tuning configuration:\n{formatted_config}")

    run_exp(train_config.model_dump(mode="json"))
```

主要加载：设备，数据（加载数据并检查数据集是否存在）

1. 加载配置

<details class="lake-collapse"><summary id="ue554ea79"><span class="ne-text">加载配置</span></summary><pre data-language="python" id="zCK8q" class="ne-codeblock language-python"><code>train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type=&quot;train_sft&quot;))</code></pre><p id="uc059cdb5" class="ne-p"><span class="ne-text">其中load_config加载用，详细代码</span></p><pre data-language="python" id="RYzP4" class="ne-codeblock language-python"><code>def load_config(arg_type: str) -&gt; BaseModel:
    &quot;&quot;&quot;Main function for loading configuration&quot;&quot;&quot;
    # Load base configuration
    wc_config = load_base_config()

    config_pydantic = create_config_by_arg_type(arg_type, wc_config)

    process_config_dict_and_argv(arg_type, config_pydantic)

    return config_pydantic</code></pre><p id="ua371f8b7" class="ne-p"><span class="ne-text">在这里load_base_config()加载的是jsonc中的common数据</span></p><p id="u2681474c" class="ne-p"><span class="ne-text">具体看一下加载的逻辑</span></p><pre data-language="python" id="u4qLi" class="ne-codeblock language-python"><code>def load_base_config() -&gt; WcConfig:
    &quot;&quot;&quot;Load base configuration file and create WcConfig object&quot;&quot;&quot;
    config_path = os.environ.get(&quot;WECLONE_CONFIG_PATH&quot;, &quot;./settings.jsonc&quot;)
    logger.info(f&quot;Loading configuration from: {config_path}&quot;)

    try:
        with open(config_path, &quot;r&quot;, encoding=&quot;utf-8&quot;) as f:
            s_config_dict: Dict[str, Any] = pyjson5.load(f)
    except FileNotFoundError:
        logger.error(f&quot;Configuration file not found: {config_path}&quot;)
        sys.exit(1)
    except Exception as e:
        logger.error(f&quot;Error loading configuration file {config_path}: {e}&quot;)
        sys.exit(1)

    # Use OmegaConf to parse configuration, then convert to Pydantic model for validation
    try:
        omega_config = OmegaConf.create(s_config_dict)
        config_dict_for_validation = OmegaConf.to_container(omega_config, resolve=True)
        if not isinstance(config_dict_for_validation, dict):
            raise TypeError(
                f&quot;Configuration should be a dictionary, but got {type(config_dict_for_validation)}&quot;
            )
        wc_config = WcConfig(**cast(Dict[str, Any], config_dict_for_validation))
    except Exception as e:
        logger.error(f&quot;Error parsing configuration with OmegaConf and WcConfig: {e}&quot;)
        sys.exit(1)

    return wc_config</code></pre><p id="u57b2f776" class="ne-p"><span class="ne-text">简述一下，先加载jsonc配置文件为typing.Dict，没有问题之后过一边OmegaConf去实现变量解析成dict，之后cast为Dict，并把字典解析成WcConfig</span></p><pre data-language="python" id="mBrM4" class="ne-codeblock language-python"><code>class WcConfig(BaseModel):
    model_config = {&quot;extra&quot;: &quot;forbid&quot;}

    version: str = Field(..., description=&quot;Configuration file version&quot;)
    common_args: CommonArgs = Field(..., description=&quot;Common parameters&quot;)
    cli_args: CliArgs = Field(..., description=&quot;Command line arguments&quot;)
    make_dataset_args: MakeDatasetArgs = Field(..., description=&quot;Dataset processing parameters&quot;)
    train_sft_args: TrainSftArgs = Field(..., description=&quot;SFT fine-tuning parameters&quot;)
    infer_args: InferArgs = Field(..., description=&quot;Inference parameters&quot;)
    vllm_args: VllmArgs = Field(VllmArgs())
    test_model_args: TestModelArgs = Field(TestModelArgs())

class CommonArgs(BaseConfigModel):
    &quot;&quot;&quot;NOTE that all parameters here will be parsed by `HfArgumentParser`. Non-HfArgumentParser parameters should be placed in make_dataset_args.&quot;&quot;&quot;

    model_name_or_path: str = Field(...)
    adapter_name_or_path: str = Field(&quot;./model_output&quot;, description=&quot;Also as output_dir of train_sft_args&quot;)
    template: str = Field(..., description=&quot;model template&quot;)
    default_system: str = Field(..., description=&quot;default system prompt&quot;)
    finetuning_type: FinetuningType = Field(FinetuningType.LORA)
    media_dir: str = Field(&quot;dataset/media&quot;)
    image_max_pixels: int = Field(409920, description=&quot;used in llama-factory, 409920 represents 720P&quot;)
    enable_thinking: bool = Field(False, description=&quot;used in llama-factory&quot;)
    trust_remote_code: bool = Field(True, description=&quot;used in huggingface&quot;)</code></pre><p id="u610144a2" class="ne-p"><span class="ne-text">这里的WcConfig包含的字段就是jsonc中的各个配置。Field(...)指的是必须输入。Feild(str，None）指的是如果没有输入默认的值</span></p><p id="u87216e63" class="ne-p"><span class="ne-text">这里的BaseModel是</span><strong><span class="ne-text" style="font-size: 13px">Pydantic</span></strong><span class="ne-text" style="font-size: 13px">中</span><span class="ne-text">的类，用来检验字段是否正常传入之类的，配合Field之类的使用</span></p><p id="u190f5b81" class="ne-p"><span class="ne-text">之后就是create_config_by_arg_type,实现合并</span></p><p id="u520ef485" class="ne-p"><span class="ne-text"></span></p><p id="u38e536b8" class="ne-p"><span class="ne-text"></span></p><p id="u44e857f8" class="ne-p"><span class="ne-text"></span></p></details>
##### 推理加速
可用unsloth加速

#### 部署
这里用FastAPI构建了一个web调用的服务，服务的Swagger文档也给出来了，可以直接测试

weclone-cli server即可启动服务器

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1758011572703-5c49011c-91d6-47cb-9fc4-716a7daa00cd.png)

在这个接口上提问就ok了，现在看效果lora效果几乎没有，后面调整一下

### 
