---
title: Deepsearch
urlname: fq4sz5g473cwwzqz
date: '2025-11-12 20:44:42'
updated: '2025-12-03 15:11:37'
description: '1. 跑完一个问题各个指标async def search_all_early_stop(query: str, num_results: int = 20, n_stop: int = 5) -&gt; str: content = "" #results = json.loads(serperd...'
---
## 跑完一个问题各个指标


```python
async def                                                                                                                                                                               search_all_early_stop(query: str, num_results: int = 20, n_stop: int = 5) -> str:
    content = ""
    #results = json.loads(serperdev_search(query,20))
    results = serperdev_search(query,num_results)
    organic_results = results.get("organic", [])

    sem = Semaphore(10)  # 最多10个并发
    stop_event = asyncio.Event()  # 控制提前停止

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(process_result(session, r, sem)) for r in organic_results]

        completed_count = 0
        for task in asyncio.as_completed(tasks):
            if stop_event.is_set():
                break  # 达到 n_stop，提前停止
            try:
                result_content = await task
                content += result_content
                completed_count += 1
                print(f"{completed_count}. {organic_results[completed_count-1]['title']} -- [{organic_results[completed_count-1]['link']}]\n", file=sys.stderr)
                if completed_count >= n_stop:
                    stop_event.set()
            except Exception as e:
                print(f"Error processing task: {e}", file=sys.stderr)

        # 取消剩余未完成的任务
        for t in tasks:
            if not t.done():
                t.cancel()

    return content
```

sem控制并发数量

async with aiohttp.ClientSession() as session:

1. session封装了http连接池

其中http连接池作用是尽可能的复用tcp连接

> 即在get方法的时候，尽可能地不重新创建tcp连接
>

2. with关键字的作用是安全的关闭的异步关闭session，防止session异常终止的时候会话没有关闭，从而导致内存泄露
3. 

## 指标
1. LLM API token消耗量
    1. Decompose阶段：
    2. gen_query阶段：
    3. extract阶段：
        1. 
        2. 3到5个query
        3. 每个query 70k token
        4. 总结大概
    4. 总结：
        1. 大概1Mtoken-2Mtoken之间/question
2. 时间
    1. 10min per quesiton
3. Jina token消耗量
    1. 500k token per question

### 示例
1. 以如今全球最大连锁酒店品牌第一家酒店成立当年中国正在公演的西方歌剧原著原作者的安葬地位于的城市中所在区的占地面积有多大？为例
+ decompose阶段：

{'token_usage': {'completion_tokens': 1470, 'prompt_tokens': 631, 'total_tokens': 2101, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-CiazucYE8iqLzPoX9JrVmHHh8DblD', 'finish_reason': 'stop', 'logprobs': None}

+ gen_query阶段：

{'token_usage': {'completion_tokens': 1586, 'prompt_tokens': 162, 'total_tokens': 1748, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cib3uXIN3csU1vQme40f5UlStVai0', 'finish_reason': 'stop', 'logprobs': None}

+ extract_node阶段：

{'token_usage': {'completion_tokens': 1374, 'prompt_tokens': 69703, 'total_tokens': 71077, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cib696ntWNyphkM9bWx05PI4qtRSN', 'finish_reason': 'stop', 'logprobs': None}

{'token_usage': {'completion_tokens': 2402, 'prompt_tokens': 49310, 'total_tokens': 51712, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cib68yNj4h19Z2hlntpsHCb2XzBbq', 'finish_reason': 'stop', 'logprobs': None}

{'token_usage': {'completion_tokens': 2197, 'prompt_tokens': 49308, 'total_tokens': 51505, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cib69gJbu5fOqr8nKl4h2Ie6T3cQV', 'finish_reason': 'stop', 'logprobs': None}

{'token_usage': {'completion_tokens': 2064, 'prompt_tokens': 106255, 'total_tokens': 108319, 'completion_tokens_details': {...}, 'prompt_tokens_details': {...}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cib698T7ox15XoxLCxev9J4aJPXmA', 'finish_reason': 'stop', 'logprobs': None}

