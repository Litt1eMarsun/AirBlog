---
title: Python
urlname: myobggs8tblnylu1
date: '2025-11-10 14:42:26'
updated: '2025-12-04 16:24:15'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1762915814480-7a59782f-467d-4f82-8432-b1aeb12e36e7.png'
description: 读取excelimport pandasiloc函数实现元素的操作与读取1. 生成器：Generator的motivation是较好的利用内存，并不像数组链表之类的，一下就把所有的对象全部生成，并占用内存，其是需要的时候再生成调用定义generator只需要将返回的return替换成yield...
---
1. 读取excel

import pandas

iloc函数实现元素的操作与读取

## 生成器：
Generator的motivation是较好的利用内存，并不像数组链表之类的，一下就把所有的对象全部生成，并占用内存，其是需要的时候再生成调用

定义generator只需要将返回的return替换成yield即可

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1762916544998-64bc7629-ab7e-4cfd-afa6-a42367c9bc74.png)

要理解generator理解yield即可，

yield实际上就是一个执行+暂停的指针

调用的过程是：如果上层调用了generator的时候，如果generator用yield返回，则系统就知道返回的类型是generator，并在yield之后暂停，指针停留再当前语句的最后，下一行语句之前

当再次调用generator，程序会跳到当前yield指针下继续执行

如果后面有yield（可以是循环，可以直接写）则程序继续运行，一直碰到下一个yield继续停止

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1762916666234-0a85e85d-fac3-4938-8511-fd91c897a61d.png)

如果后面没有yield了，则继续调用next会抛出stopIteration异常。

同时如果想调用生成器链，则需要使用到yield from关键词

如下图所示

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1762916952980-0f0aae33-d5f0-440b-8ded-627a59b8e4d1.png)

肯定不能用yield调用generator，因为yield只进行一次，也就是调用了生成器一次就结束了，而yield调用的过程如下图：

```python
# 等同于以下手动代码
def another_generator_manual():
    g = generator()
    for value in g:
        yield value  # 手动转发每个值
    # 还会自动处理异常和返回值
```

（当然自己写层类似的循环也可以，不过没必要）

即想声称其为外面再套生成器，则需要yiled from

## 异步编程
1. 手搓一个异步编程：详细包括了yield/generator知识点，和synacawait

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1762915814480-7a59782f-467d-4f82-8432-b1aeb12e36e7.png)

## 异步编程
有几个比较重要的关键字

asynic关键字，asynic关键字声明的方法内部会多一些实现，其中最重要的就是dunder方法__await__

也就是说，只有使用asynic关键字声明的方法内部才可以使用await关键字，否则会报错



asynic.run：携程开始跑，内部的写成写入一个task队列，task队列中的任务循环跑，如果await指针指向的任务结束了，则返回该任务，一直到指定的任务数量全部返回完毕则直接返回



await关键字调用的方法，类似于yield的逻辑，不过在外面套了一层方法。下面是asynico源码对await的定义

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1762918455839-1475d90e-61bf-40e7-891c-0612376373b8.png)

也就是如果当前代码执行完毕了，则返回当前结果，如果没有完成，则返回当前方法本身。

也类似于一个指针，不过把指针给了await声明的方法，在执行其标注的方法的时候，将指针停在方法之中，如果await调用的方法返回了结果，则直接返回，如果没有，则返回方法本身，告诉程序接着下一步执行.





## 网络连接
### 同步HTTP库
http.clinet和网页做出交互，并写入信息，获得响应

> conn = http.client.HTTPSConnection("google.serper.dev")
>

获得连接，之后可以通过request写入请求

输入参数的：依次是：请求方法（HTTP method） 、 请求的路径（URL path） 、 请求体（body）、 请求头（HTTP headers）。 详细看下http协议

```python
conn.request("GET", "/")
conn.request("POST", "/search", payload, headers)
```

输入请求之后，通过

res = conn.getresponse()

方法获得请求的回复

回复的方法可以通过

`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">status</font>`<font style="color:rgb(25, 27, 31);">、</font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">reason</font>`<font style="color:rgb(25, 27, 31);">、</font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">read()</font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">等方法获得状态码，原因，和内容</font>

<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">这里通过</font>

<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">data = res.read()</font>

<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">直接获得请求的数据内容</font>

<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);"></font>





### 异步aioHTTP库：
能够异步的执行



## 泛型
[https://blog.csdn.net/qq_17275369/article/details/147588238](https://blog.csdn.net/qq_17275369/article/details/147588238)

泛型可以是容器类的泛型，也可以是自定义的泛型，自定义泛型如下图：

1. TypeVar

这个类用来自定义泛型，同时控制泛型的范围：

```python
N = TypeVar("N", bound=Hashable)
```

比如说这个就是表明N代表一个泛型，同时泛型必须是Hashable的子集

这个自定义泛型在实例化的时候需会根据[]输入自动条件泛型类

2. Genratic

## 参数校验：
### 动态参数校验
### 静态参数校验：
#### typing
Typing类里面的很多子类都是可以进行静态参数校验 的，即如果传参出现错误，直接在编辑器中就会报错

1. TypeVar

这个类用来表示泛型，同时控制泛型的范围：

```python
N = TypeVar("N", bound=Hashable)
```

比如说这个就是表明N代表一个泛型，同时泛型必须是Hashable的子集

2. Genratic



限制某个字段的字符串只能在所选之内。

## 数据类型
zip()将两个/多个迭代器压成一个元组迭代器

> name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
>

[].append数组后添加元素

np.concatenate,numpy数组按照特定维度的concat

## langchain
### Runnable接口
runnable接口硅锭实现的类必须有以下方法：

1. 同步类：

正常的invoke，流式的stream, 批处理batch

2. 异步类：

前面会加a

batch在同步中的处理方式是通过线程池来进行IO密集任务的多线程处理，但是对于CPU密集任务而言，python有全局的GIL锁，所以没办法做到效率的提高

abatch是利用异步编程来实现的，也就是通过手动控制携程来进行异步并发操作，所以可以自由控制。

### Command类




