---
title: Brosweruse
urlname: dqg1203o9pko2lrq
date: '2025-11-28 14:31:01'
updated: '2025-12-01 18:37:52'
cover: 'https://cdn.nlark.com/yuque/0/2025/png/43288584/1764583457226-26163554-5264-4b1a-94c8-5b0fa678cab0.png'
description: '主程序入口workspace/browser-use/browser_use/agent/service.pyagent 服务当中的run作为主程序入口，从run函数开始解释loop = asyncio.get_event_loop() agent_run_error: str | None ...'
---
## 主程序入口
1. workspace/browser-use/browser_use/agent/service.py

agent 服务当中的run作为主程序入口，从run函数开始解释

```python
loop = asyncio.get_event_loop()
        agent_run_error: str | None = None  # Initialize error tracking variable
        self._force_exit_telemetry_logged = False  # ADDED: Flag for custom telemetry on force exit
        should_delay_close = False

        # Set up the  signal handler with callbacks specific to this agent
        from browser_use.utils import SignalHandler

        # Define the custom exit callback function for second CTRL+C
        def on_force_exit_log_telemetry():
            self._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
            # NEW: Call the flush method on the telemetry instance
            if hasattr(self, 'telemetry') and self.telemetry:
                self.telemetry.flush()
            self._force_exit_telemetry_logged = True  # Set the flag

        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=on_force_exit_log_telemetry,  # Pass the new telemetrycallback
            exit_on_second_int=True,
        )
        signal_handler.register()
```

包括了

+ 事件循环的异步任务列表的创建
+ 突然ctrl+c中断导致的程序优雅终止
2. 

```python
try:
            await self._log_agent_run()

            self.logger.debug(
                f'🔧 Agent setup: Agent Session ID {self.session_id[-4:]}, Task ID {self.task_id[-4:]}, Browser Session ID {self.browser_session.id[-4:] if self.browser_session else "None"} {"(connecting via CDP)" if (self.browser_session and self.browser_session.cdp_url) else "(launching local browser)"}'
            )

            # Initialize timing for session and task
            self._session_start_time = time.time()
            self._task_start_time = self._session_start_time  # Initialize task start time

            # Only dispatch session events if this is the first run
            if not self.state.session_initialized:
                self.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
                # Emit CreateAgentSessionEvent at the START of run()
                self.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self))

                self.state.session_initialized = True
```

初始化session的时候，创建agentsessionevent这个事件，并将其和其所有的回调函数等等分发到事件总线上

```python
self.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
            # Emit CreateAgentTaskEvent at the START of run()
            self.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self))
```

在这里创建task的时候也将该事件放到bus上

```python
# Log startup message on first step (only if we haven't already done steps)
            self._log_first_step_startup()
            # Start browser session and attach watchdogs
            await self.browser_session.start()
            if self._demo_mode_enabled:
                await self._demo_mode_log(f'Started task: {self.task}', 'info', {'tag': 'task'})
                await self._demo_mode_log(
                    'Demo mode active - follow the side panel for live thoughts and actions.',
                    'info',
                    {'tag': 'status'},
                )
```

这里重要的是启动了broswer_session，同时如果使用了demo模式，则调用_demo_mode_log打印日志，日志的内容打印由broswer_session会话发起动作，打印到浏览器上

```python
# Normally there was no try catch here but the callback can raise an InterruptedError
            try:
                await self._execute_initial_actions()
            except InterruptedError:
                pass
            except Exception as e:
                raise e
```

这里执行了初始化动作（要是有的话），大概内容如下：不细说  


```python
async def _execute_initial_actions(self) -> None:
        # Execute initial actions if provided
        if self.initial_actions and not self.state.follow_up_task:
            self.logger.debug(f'⚡ Executing {len(self.initial_actions)} initial actions...')
            result = await self.multi_act(self.initial_actions)
            # update result 1 to mention that its was automatically loaded
            if result and self.initial_url and result[0].long_term_memory:
                result[0].long_term_memory = f'Found initial url and automatically loaded it. {result[0].long_term_memory}'
            self.state.last_result = result

            # Save initial actions to history as step 0 for rerun capability
            # Skip browser state capture for initial actions (usually just URL navigation)
            if self.settings.flash_mode:
                model_output = self.AgentOutput(
                    evaluation_previous_goal=None,
                    memory='Initial navigation',
                    next_goal=None,
                    action=self.initial_actions,
                )
            else:
                model_output = self.AgentOutput(
                    evaluation_previous_goal='Start',
                    memory=None,
                    next_goal='Initial navigation',
                    action=self.initial_actions,
                )

            metadata = StepMetadata(step_number=0, step_start_time=time.time(), step_end_time=time.time(), step_interval=None)

            # Create minimal browser state history for initial actions
            state_history = BrowserStateHistory(
                url=self.initial_url or '',
                title='Initial Actions',
                tabs=[],
                interacted_element=[None] * len(self.initial_actions),  # No DOM elements needed
                screenshot_path=None,
            )

            history_item = AgentHistory(
                model_output=model_output,
                result=result,
                state=state_history,
                metadata=metadata,
            )

            self.history.add_item(history_item)
            self.logger.debug('📝 Saved initial actions to history as step 0')
            self.logger.debug('Initial actions completed')
```

```python
while self.state.n_steps <= max_steps:
                current_step = self.state.n_steps - 1  # Convert to 0-indexed for step_info

                # Use the consolidated pause state management
                if self.state.paused:
                    self.logger.debug(f'⏸️ Step {self.state.n_steps}: Agent paused, waiting to resume...')
                    await self._external_pause_event.wait()
                    signal_handler.reset()

                # Check if we should stop due to too many failures, if final_response_after_failure is True, we try one last time
                if (self.state.consecutive_failures) >= self.settings.max_failures + int(
                    self.settings.final_response_after_failure
                ):
                    self.logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    agent_run_error = f'Stopped due to {self.settings.max_failures} consecutive failures'
                    break

                # Check control flags before each step
                if self.state.stopped:
                    self.logger.info('🛑 Agent stopped')
                    agent_run_error = 'Agent stopped programmatically'
                    break

                step_info = AgentStepInfo(step_number=current_step, max_steps=max_steps)
                is_done = await self._execute_step(current_step, max_steps, step_info, on_step_start, on_step_end)

                if is_done:
                    # Agent has marked the task as done
                    if self._demo_mode_enabled and self.history.history:
                        final_result_text = self.history.final_result() or 'Task completed'
                        await self._demo_mode_log(f'Final Result: {final_result_text}', 'success', {'tag': 'task'})

                    should_delay_close = True
                    break
            else:
```

如果agent调用的总step数小于max_step，则可以一直调用，如果已经到了，则到后续的处理当中。

同时对一些异常情况（尝试失败次数过多，主动暂停）进行处理，并出动打破循环，进入agent结束阶段

如果agent输出is_done（结束调用）则处理所有信息，并进入结束阶段/

在_execute_step中执行agent调用。

接下来进入到_execute_step这个函数当中

```python
async def _execute_step(
        self,
        step: int,
        max_steps: int,
        step_info: AgentStepInfo,
        on_step_start: AgentHookFunc | None = None,
        on_step_end: AgentHookFunc | None = None,
    ) -> bool:
        """
        Execute a single step with timeout.

        Returns:
            bool: True if task is done, False otherwise
        """
        if on_step_start is not None:
            await on_step_sdatart(self)

        await self._demo_mode_log(
            f'Starting step {step + 1}/{max_steps}',
            'info',
            {'step': step + 1, 'total_steps': max_steps},
        )

        self.logger.debug(f'🚶 Starting step {step + 1}/{max_steps}...')

        try:
            await asyncio.wait_for(
                self.step(step_info),
                timeout=self.settings.step_timeout,
            )
            self.logger.debug(f'✅ Completed step {step + 1}/{max_steps}')
        except TimeoutError:
            # Handle step timeout gracefully
            error_msg = f'Step {step + 1} timed out after {self.settings.step_timeout} seconds'
            self.logger.error(f'⏰ {error_msg}')
            await self._demo_mode_log(error_msg, 'error', {'step': step + 1})
            self.state.consecutive_failures += 1
            self.state.last_result = [ActionResult(error=error_msg)]

        if on_step_end is not None:
            await on_step_end(self)

        if self.history.is_done():
            await self.log_completion()

            # Run judge before done callback if enabled
            if self.settings.use_judge:
                await self._judge_and_log()

            if self.register_done_callback:
                if inspect.iscoroutinefunction(self.register_done_callback):
                    await self.register_done_callback(self.history)
                else:
                    self.register_done_callback(self.history)

            return True

        return False
```

这个函数主要是做agent的单步骤的，包含了

1. 单步step进行的时候对step信息进行打印（输入agent实例，打印的估计是agent本身的信息）
2. 一些单步相关的日志打印（当前步数等等）

之后使用wait_for方法异步调用step方法，并设置最大等待时间（后续可以看看wait_for方法）

最后对单步的异常情况进行处理：

1. time Up
2. 结束的时候可以选择使用LLM对当前step进行评价，并可以调用done的回调函数进行记录？这里的回调函数可以看看

后续进入到step函数当中研究agent的单步调用

```python
@observe(name='agent.step', ignore_output=True, ignore_input=True)
    @time_execution_async('--step')
    async def step(self, step_info: AgentStepInfo | None = None) -> None:
        """Execute one step of the task"""
        # Initialize timing first, before any exceptions can occur

        self.step_start_time = time.time()

        browser_state_summary = None

        try:
            # Phase 1: Prepare context and timing
            browser_state_summary = await self._prepare_context(step_info)

            # Phase 2: Get model output and execute actions
            await self._get_next_action(browser_state_summary)
            await self._execute_actions()

            # Phase 3: Post-processing
            await self._post_process()

        except Exception as e:
            # Handle ALL exceptions in one place
            await self._handle_step_error(e)

        finally:
            await self._finalize(browser_state_summary)
```

每个step的调用根据react可以分为四个部分：

+ 上下文准备 _prepare_context(step_info)
+ 获得当前step的action  _get_next_action
+ 执行action  _execute_actions()
+ 后处理 _post_process()

重点看看工具调用这里，一些操作如何和浏览器交互的

```python
async def _execute_actions(self) -> None:
        """Execute the actions from model output"""
        if self.state.last_model_output is None:
            raise ValueError('No model output to execute actions from')

        result = await self.multi_act(self.state.last_model_output.action)
        self.state.last_result = result
```

调用工具之前，设定好

## 任务记录：
### 搜索豆瓣前top250图书√
按理说11个step就能完成

1. 找到top250页面
2. 发现有页面标签，收集第一个页面
3. 点击第二个页面，收集第二个页面...
4. 调用结束工具

实际上出现问题：

1. 提取信息工具调用出错，重新调用coder去写了个JS来提取信息，成功
2. 点击出错，准备点击到第二个，实际上点击到了最后一个——DOM识别问题还是点击问题？

导致反复调用api，最后50+step才完成问题

剩下到问题不大

观察到的现象：获得下一个页面的操作分为两种

+ 获取直接nvigate 特定url来获得对应页面的书籍
+ 点击下一个页面



### 抖音账号"毒舌电影"2024年7月最后一次发布的电影解说中，宣传了什么商品？×
理想：

+ 导航到抖音首页
+ input

问题：

1. 会出现humaneval，不过后续可以跳开，有点耽误时间
2. 没有登陆，页面点击不进去，导致后续任务失败 		



### 自公元 0 年至100年记录在 NASA “Five Millennium Canon of Solar Eclipses” 中的日全食记录一共有哪些？
1. 遇到了验证码问题，剩下问题绕过了

![](https://cdn.nlark.com/yuque/0/2025/png/43288584/1764583457226-26163554-5264-4b1a-94c8-5b0fa678cab0.png)

2. 页面信息抓取出错，已经导航到了-99-0和0-100两个页面，但是页面内容抓取总是出错
3. 花费很多次数在scroll上面

### 总结：
1. 完成简单任务问题不大
2. extrat工具调用容易出错
3. 时间过多
    1. llm invoke所需时间很多，轮次一多时间就多，浏览器使用时间很短
    2. 减少交互次数，尽量一次性拿到更多信息——只有必须要交互的时候才交互

用到的工具列表如下：

+ write_file——用来写入文件来记录当前已经取得的成果
+ evaluate——写js，找所需的dom元素
+ replace_file——替换，搜索到内容之后再文件中记录`"- [ ] buy milk"` → `"- [x] buy milk"`
+ scroll——滚动屏幕到底部，准备点击控件
+ wait——滚动之后等待页面渲染
+ click——点击dom元素
+ input-抖音中进行搜索

有下载的工具，但是应该没有pdf阅读的工具，如果要阅读官方的PDF可能没办法操作

时间过多的原因以及可以省略的地方

+ scroll是否可以省？

###   

